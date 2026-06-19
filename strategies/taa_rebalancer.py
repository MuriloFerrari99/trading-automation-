"""REBALANCEADOR VIVO do sleeve de DUAL MOMENTUM / TAA defensivo.

Pega o UNICO candidato que GRADUOU no programa de validacao de alpha (rounds 1+2,
ver data/programa_validacao_round2_verdict.txt) e o transforma em ordens de
rebalance para o paper Alpaca. NAO reimplementa o sinal — IMPORTA as funcoes ja
auditadas de simulation.dual_momentum (build_weights / _daily_weights), exatamente
como strategies/beta_rebalancer.py faz com o nucleo do beta. O que este modulo
adiciona e SO a ponte pesquisa->producao.

ESPEC DE PRODUCAO (travada, = config campea do veredito): broad8_L6_N4_AGG.
  - universo de risco: SPY, QQQ, EFA, EEM, VNQ, DBC, GLD, TLT
  - lookback de momentum: 6 meses
  - top-N relativo: 4   (equal-weight 1/4 por slot)
  - filtro absoluto: so fica no ativo de risco se momentum_L6 > momentum(BIL);
    senao o slot vai p/ o defensivo AGG.
  - rebalance MENSAL, long-only, SEM alavancagem (soma dos pesos em [0,1]).
  Metricas auditadas (custo ESTRESSADO 2x): Sharpe 1.17 | CAGR 12.2% | MaxDD
  -13.6% | DSR 0.999 | PBO 0.18 | corr_SPY 0.63 (colapsa p/ 0.118 nos crashes).
  Corta o DD do 60/40 pela metade e entrega crisis-alpha (+8.6% no GFC 2008).

PARIDADE (o gate do Planner): `target_weights_for_date(panel, asof)` DEVE bater,
ao float, com a linha `asof` do vetor de pesos diario que o BACKTEST calcula sobre
o painel inteiro (`production_weight_frame`). O teste tests/test_taa_rebalancer.py
prova isso — e o que garante que o que vai pro paper e exatamente o que foi auditado.

SEM LOOK-AHEAD: o calculo corta o painel em `index <= asof` e as funcoes de peso
ja aplicam .shift(1) (a linha t so usa dados <= t-1). Pegamos a ultima linha (asof)
como o alvo a carregar.

GUARDS: respeita kill switch (via Executor) e PortfolioRiskGuard (halt / per-symbol).
Sem efeito colateral ao importar (nao baixa dados).

PRE-CONDICOES p/ capital real (do proprio veredito): assinatura do Coder; OOS
re-rodado estendendo a janela; revalidar turnover/custo no broker real. Em paper a
flag de captura/trading vivo nasce DESLIGADA (ver strategies/taa_live.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from broker.base import BrokerClient
from core.models import OrderIntent, OrderSide, OrderType
from core.rounding import round_qty
from data.state_repo import StateRepository
from risk.portfolio_guard import PortfolioRiskGuard

# REUTILIZA (importa, NAO edita) o nucleo auditado de sinal/pesos do tribunal.
from simulation.dual_momentum import (
    DEFENSIVE,
    RISK_BROAD,
    _daily_weights,
    build_weights,
    load_panel,
)

logger = logging.getLogger("strategies.taa_rebalancer")

# ---------------------------------------------------------------------------
# PARAMETROS DE PRODUCAO (travados; config campea broad8_L6_N4_AGG do veredito).
# ---------------------------------------------------------------------------
STRATEGY_NAME = "taa_rebalancer"

PROD_RISK_UNIVERSE = list(RISK_BROAD)        # SPY,QQQ,EFA,EEM,VNQ,DBC,GLD,TLT
PROD_LOOKBACK_M = 6
PROD_TOP_N = 4
PROD_DEFENSIVE = "AGG"                        # in DEFENSIVE = ["IEF", "AGG"]
assert PROD_DEFENSIVE in DEFENSIVE            # espec travada (defensivo do campeao)

# Tickers efetivamente negociaveis = universo de risco + o ativo defensivo.
# (BIL e so proxy de cash p/ o filtro de momentum; SPY/IEF entram no painel mas
#  nao recebem peso fora do universo/defensivo.) Todos ETFs => lote inteiro.
TRADABLE = sorted(set(PROD_RISK_UNIVERSE + [PROD_DEFENSIVE]))

# Banda de nao-trade: so negocia um ativo se |peso_alvo - peso_atual| passar disto.
# O rebalance e mensal; drift dentro da banda nao gera ordem (turnover baixo).
DEFAULT_NO_TRADE_BAND = Decimal("0.015")


# ============================================================================
# PESOS-ALVO — a PONTE de paridade pesquisa<->producao.
# ============================================================================
def production_weight_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Vetor de pesos DIARIO da ESTRATEGIA DE PRODUCAO sobre o painel INTEIRO.

    UMA fonte de verdade para backtest e calculo vivo: monta os pesos mensais com
    `build_weights` (config campea) e propaga p/ o diario com `_daily_weights`, que
    ja aplica .shift(1) (a linha t usa apenas dados <= t-1). NAO corta o painel — o
    corte por data e de quem chama (`target_weights_for_date` corta em asof)."""
    w_monthly = build_weights(
        panel,
        risk_universe=PROD_RISK_UNIVERSE,
        lookback_m=PROD_LOOKBACK_M,
        top_n=PROD_TOP_N,
        defensive=PROD_DEFENSIVE,
    )
    return _daily_weights(panel, w_monthly)


def target_weights_for_date(
    panel: pd.DataFrame, asof: pd.Timestamp
) -> dict[str, float]:
    """Pesos-alvo a carregar em `asof` (decisao usando dados <= asof-1), sem
    look-ahead. Corta o painel em `index <= asof` ANTES de calcular; as funcoes de
    peso ainda aplicam .shift(1), entao a linha `asof` so reflete dados <= asof-1.

    Retorna {ticker: peso_alvo} apenas dos pesos NAO-triviais (ex.: {"TLT": 0.25,
    "GLD": 0.25, "AGG": 0.5}). A paridade com o backtest e garantida porque o
    backtest, sobre o painel inteiro, produz na linha `asof` os mesmos numeros."""
    cut = panel.loc[panel.index <= asof]
    if cut.empty:
        return {}
    wf = production_weight_frame(cut)
    if wf.empty:
        return {}
    row = wf.iloc[-1]
    return {str(k): float(v) for k, v in row.items() if abs(float(v)) > 1e-12}


# ============================================================================
# ESTADO / GATILHO MENSAL — sem look-ahead (espelha beta_rebalancer).
# ============================================================================
_LAST_REBALANCE_MONTH_KEY = f"{STRATEGY_NAME}:last_rebalance_month"


def _period_month(asof: pd.Timestamp) -> str:
    """Rotulo de mes 'YYYY-MM' do asof (tz-naive p/ comparacao estavel)."""
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return f"{ts.year:04d}-{ts.month:02d}"


def _rebalance_timing_ready(panel: pd.DataFrame | None, asof: pd.Timestamp) -> bool:
    """True se a janela de TIMING do rebalance ja abriu para o mes de `asof`.

    O backtest decide os pesos no ultimo pregao do mes m e, pelo .shift(1), eles
    so ENTRAM EM VIGOR no 1o pregao de m+1. O caminho vivo so deve rebalancear
    quando os pesos vigentes ja sao os do mes corrente — criterio panel-aware,
    sem look-ahead e robusto a restart: o mes de `asof` tem pelo menos 2 pregoes
    ate (inclusive) `asof`. (Espelha a logica do beta_rebalancer.)"""
    if panel is None or panel.empty:
        return True  # sem painel p/ aferir timing: nao deferimos (uso legado/teste).
    idx = panel.index
    asof_ts = pd.Timestamp(asof)
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    asof_naive = asof_ts.tz_localize(None) if asof_ts.tzinfo is not None else asof_ts
    months = idx_naive.to_period("M")
    m = asof_naive.to_period("M")
    n_days_in_month_upto = int(((months == m) & (idx_naive <= asof_naive)).sum())
    return n_days_in_month_upto >= 2


def is_rebalance_due(
    state: StateRepository | None,
    asof: pd.Timestamp,
    *,
    force: bool = False,
    panel: pd.DataFrame | None = None,
) -> bool:
    """True se um rebalance MENSAL e devido para o mes de `asof`.

    Duas condicoes (ambas sem look-ahead): (1) ESTADO — o mes de `asof` difere do
    mes do ultimo rebalance persistido (idempotencia mensal; 1a vez => devido;
    sobrevive a restart); (2) TIMING — a janela ja abriu (>= 2o pregao do mes),
    aferida por `panel`. `force` ignora ambas (uso operacional/manual)."""
    if force:
        return True
    if not _rebalance_timing_ready(panel, asof):
        return False
    if state is None:
        return True
    last = state.get(_LAST_REBALANCE_MONTH_KEY)
    return last != _period_month(asof)


def mark_rebalanced(state: StateRepository | None, asof: pd.Timestamp) -> None:
    """Persiste que o rebalance do mes de `asof` foi feito (idempotencia mensal)."""
    if state is not None:
        state.set(_LAST_REBALANCE_MONTH_KEY, _period_month(asof))


# ============================================================================
# REBALANCEADOR
# ============================================================================
@dataclass
class RebalancePlan:
    """Resultado do calculo de rebalance (auditavel, antes de executar)."""

    asof: str
    equity: Decimal
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    intents: list[OrderIntent] = field(default_factory=list)
    skipped_in_band: list[str] = field(default_factory=list)   # |dev| < banda
    blocked: list[str] = field(default_factory=list)           # vetado por guard
    notes: list[str] = field(default_factory=list)


class TAARebalancer:
    """Calcula o plano de rebalance do sleeve de TAA a partir do broker.

    NAO submete ordens (isso e o Executor). NAO baixa dados ao importar. O painel
    de precos historico (p/ os pesos) e injetado ou carregado de load_panel()
    (mesma fonte do backtest -> paridade). O broker da o equity e as posicoes
    atuais (mark-to-market = verdade da conta). Long-only, sem alavancagem.
    """

    def __init__(
        self,
        broker: BrokerClient,
        *,
        state: StateRepository | None = None,
        guard: PortfolioRiskGuard | None = None,
        no_trade_band: Decimal = DEFAULT_NO_TRADE_BAND,
        universe: list[str] | None = None,
    ) -> None:
        self._broker = broker
        self._state = state
        self._guard = guard
        self._band = Decimal(str(no_trade_band))
        self._universe = universe or list(TRADABLE)

    # --- pesos atuais a partir do broker -----------------------------------
    def _current_weights(
        self, equity: Decimal
    ) -> tuple[dict[str, float], dict[str, Decimal]]:
        """(peso_atual, preco_atual) por ticker, lidos do broker. Peso =
        valor_de_mercado / equity. So considera simbolos do universo do sleeve."""
        weights: dict[str, float] = {t: 0.0 for t in self._universe}
        prices: dict[str, Decimal] = {}
        if equity <= 0:
            return weights, prices
        positions = {p.symbol: p for p in self._broker.get_positions()}
        for tkr in self._universe:
            pos = positions.get(tkr)
            price = self._safe_price(tkr, pos)
            if price is not None and price > 0:
                prices[tkr] = price
            if pos is None or price is None or price <= 0:
                continue
            mv = Decimal(str(pos.qty)) * price
            weights[tkr] = float(mv / equity)
        return weights, prices

    def _safe_price(self, symbol: str, pos) -> Decimal | None:
        if pos is not None and getattr(pos, "current_price", None) is not None:
            return Decimal(str(pos.current_price))
        try:
            return Decimal(str(self._broker.get_last_price(symbol)))
        except Exception as exc:  # noqa: BLE001 - sem preco -> trata como faltante
            logger.warning("Preco indisponivel p/ %s: %s", symbol, exc)
            return None

    def _resolve_equity(self) -> Decimal:
        """NAV do sleeve = equity da conta (paper, sleeve unico). Fail-safe 0 se
        ilegivel (o chamador trata como rebalance abortado)."""
        try:
            return Decimal(str(self._broker.get_account().equity))
        except Exception:  # noqa: BLE001 - sem equity legivel => sem NAV
            logger.warning("_resolve_equity: equity da conta ilegivel; nav=0.")
            return Decimal("0")

    # --- calculo do plano ---------------------------------------------------
    def compute_plan(
        self,
        panel: pd.DataFrame,
        asof: pd.Timestamp,
        *,
        equity: Decimal | None = None,
        prices: dict[str, Decimal] | None = None,
    ) -> RebalancePlan:
        """Calcula o plano de rebalance para `asof` (ultimo fechamento = ontem).

        `panel` = painel de precos historico (load_panel()); `asof` corta o
        look-ahead. `equity`/`prices` opcionais p/ teste; senao lidos do broker."""
        if equity is None:
            equity = self._resolve_equity()

        target = target_weights_for_date(panel, asof)
        current, broker_prices = self._current_weights(equity)
        prices = {**broker_prices, **(prices or {})}

        plan = RebalancePlan(
            asof=str(pd.Timestamp(asof).date()),
            equity=equity,
            target_weights=target,
            current_weights=current,
        )
        if not target:
            plan.notes.append("painel sem dados ate asof — nenhum alvo calculado.")
            return plan
        if equity <= 0:
            plan.notes.append("equity <= 0 — rebalance abortado (fail-safe).")
            return plan

        for tkr in self._universe:
            self._plan_symbol(
                plan, tkr, target.get(tkr, 0.0), current.get(tkr, 0.0), equity, prices
            )
        # Vende ANTES de comprar: libera caixa antes de deployar (long-only, sem
        # margem). Nao muda quais ordens existem nem suas quantidades — so a ordem
        # de envio -> paridade de carteira final intacta.
        plan.intents.sort(key=lambda i: 0 if i.side == OrderSide.SELL else 1)
        return plan

    def _plan_symbol(
        self,
        plan: RebalancePlan,
        tkr: str,
        w_target: float,
        w_current: float,
        equity: Decimal,
        prices: dict[str, Decimal],
    ) -> None:
        dev = Decimal(str(w_target)) - Decimal(str(w_current))

        # BANDA DE NAO-TRADE: desvio pequeno nao gera ordem (turnover baixo).
        if abs(dev) < self._band:
            plan.skipped_in_band.append(tkr)
            return

        price = prices.get(tkr)
        if price is None or price <= 0:
            plan.blocked.append(f"{tkr} (sem preco)")
            return

        target_notional = Decimal(str(w_target)) * equity
        current_notional = Decimal(str(w_current)) * equity
        delta_notional = target_notional - current_notional
        side = OrderSide.BUY if delta_notional > 0 else OrderSide.SELL

        raw_qty = abs(delta_notional) / price
        qty = round_qty(raw_qty, None, fractional=False)  # ETFs => lote inteiro
        if qty <= 0:
            plan.skipped_in_band.append(tkr)  # delta abaixo de 1 unidade negociavel
            return

        # GUARD: ordens que AUMENTAM exposicao passam pelo can_open (halt global /
        # teto por simbolo). Reducoes (des-risca) sempre passam.
        if self._guard is not None and side == OrderSide.BUY:
            target_symbol_exposure = abs(Decimal(str(w_target)))
            ok, reason = self._guard.can_open(target_symbol_exposure, Decimal("0"))
            if not ok:
                plan.blocked.append(f"{tkr} ({reason})")
                return

        plan.intents.append(
            OrderIntent(
                symbol=tkr,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                strategy=STRATEGY_NAME,
            )
        )

    # --- orquestracao (calculo apenas; execucao e do Executor) --------------
    def plan_if_due(
        self,
        panel: pd.DataFrame,
        asof: pd.Timestamp,
        *,
        force: bool = False,
        equity: Decimal | None = None,
        prices: dict[str, Decimal] | None = None,
    ) -> RebalancePlan | None:
        """Plano de rebalance SE o gatilho mensal disparar; senao None.

        NAO marca o mes como rebalanceado (quem executa chama mark_rebalanced apos
        o Executor confirmar) — assim, se o ciclo cair antes de executar, o
        rebalance volta a ser devido no proximo tick."""
        if not is_rebalance_due(self._state, asof, force=force, panel=panel):
            return None
        return self.compute_plan(panel, asof, equity=equity, prices=prices)


def load_production_panel(force: bool = False) -> pd.DataFrame:
    """Painel de precos da estrategia (MESMA fonte do backtest -> paridade).

    Reusa simulation.dual_momentum.load_panel (yfinance cacheado). Em producao
    real, este e o ponto a trocar por um feed de dados ao vivo equivalente —
    mantendo a IDENTIDADE de calculo de pesos com o backtest."""
    return load_panel(force=force)
