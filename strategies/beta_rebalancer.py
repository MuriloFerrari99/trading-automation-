"""REBALANCEADOR VIVO da estrategia de beta (vol-target + hedge condicional + 2.0x).

Este e o build FINAL antes do track record: pega a ESTRATEGIA DE PRODUCAO TRAVADA
(auditada em simulation/beta_portfolio.py + simulation/beta_v2.py) e a transforma
em ordens de rebalance para o paper Alpaca. NAO reimplementa pesos — IMPORTA as
funcoes ja auditadas. O que este modulo adiciona e SO a ponte pesquisa->producao:

  1. PESOS-ALVO DE HOJE sem look-ahead: corta o painel de precos no fechamento de
     ONTEM (asof) e chama os MESMOS construtores de peso do backtest. Por
     construcao (os construtores ja fazem .shift(1)), a linha `asof` do vetor de
     pesos so usa dados <= asof-1; e aqui ainda cortamos o painel em asof, de modo
     que nenhum preco futuro entra. Ver `target_weights_for_date`.
  2. POSICOES ATUAIS -> pesos atuais (valor de mercado / equity), lidos do broker.
  3. ORDENS DE DIFERENCA (alvo - atual) em NOTIONAL, convertidas a quantidade,
     so negociando o que passar da BANDA DE NAO-TRADE (turnover baixo).
  4. GATILHO MENSAL: rebalanceia 1x/mes, no 2o pregao do mes (r+1) — o dia em que
     os pesos do backtest entram em vigor (os construtores fazem .shift(1) sobre o
     rebalance do 1o pregao r). Combina idempotencia por estado persistido ("o mes
     mudou desde o ultimo rebalance") com um gate de TIMING panel-aware (asof >= 2o
     pregao do mes). Sem isso, o vivo negociaria no dia r contra o alvo do mes
     ANTERIOR e o NAV divergiria do backtest todo mes (MF-2).
  5. GUARDS: respeita kill switch e PortfolioRiskGuard (halt/per-symbol).

ESPEC DE PRODUCAO (travada): vol-target (1/vol, alvo 10% a.a.) + hedge condicional
de cauda (des-risca em bear sustentado) + universo SPY,QQQ,TLT,IEF,GLD,SLV +
BTC/ETH (cap 5%/nome) + ALAVANCAGEM 2.0x. Venue: Alpaca paper.

PARIDADE (o gate central do Planner): `target_weights_for_date(panel, classes, d)`
DEVE bater, dentro de tolerancia, com a linha `d` do vetor de pesos que o BACKTEST
calcula para o painel inteiro. O teste tests/test_beta_rebalancer.py prova isso —
e o que garante que o que vai pro paper e exatamente o que foi auditado.

HONESTIDADE: sem look-ahead (so dados <= ontem); a flag de captura/trading vivo
comeca DESLIGADA (ver strategies/beta_live.py); nenhum efeito colateral ao
importar este modulo.

TAXA DE MARGEM (paper): a alavancagem 2.0x toma ~100% do capital emprestado (gross
medio real ~166% => ~66% emprestado). Em paper a Alpaca nao cobra juro de margem,
entao o custo de financiamento NAO aparece no NAV do paper; o backtest beta_v2 ja
assumiu FINANCING_ANNUAL=5.5% a.a.
(~rf medio + spread) sobre o emprestado e e essa a premissa documentada. Em
producao real, trocar pela taxa efetiva da corretora (Alpaca margin ~ base + spread).
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

# REUTILIZA (importa, NAO edita) o nucleo auditado de pesos/painel.
from simulation.beta_portfolio import UNIVERSE, load_panel
from simulation.beta_v2 import (
    leverage_weights,
    weights_volt_with_conditional_hedge,
)

logger = logging.getLogger("strategies.beta_rebalancer")

# ---------------------------------------------------------------------------
# PARAMETROS DE PRODUCAO (travados; ancorados no veredito do beta_v2).
# ---------------------------------------------------------------------------
STRATEGY_NAME = "beta_rebalancer"

# Alavancagem de producao: 2.0x (ESCOLHA CONSCIENTE do usuario — mais retorno,
# aceitando mais DD). Na fronteira (data/beta_frontier_report.txt, vol-alvo 10% x
# 2.0x) esse nivel entrega CAGR +10.9% / Sharpe 0.95 / MaxDD backtest -25.9%
# (DD ao-vivo estimado -36.2%/-38.8% = max(1.4x backtest, pior crise); pior ano
# -19.5% em 2022; gross medio real ~166%, expo-alvo 20%). A alavancagem e
# PARAMETRO (escala o vetor de pesos por L em leverage_weights) — as funcoes de
# peso auditadas (beta_portfolio/beta_v2) NAO mudam; so este numero. Trade-off de
# AUM registrado: DD ao-vivo > ~-35% afasta alocador, mas o usuario optou pelo
# retorno maior. Os guards do beta (config/risk.py BetaGuardSettings) estao
# recalibrados a ESTE perfil 2.0x — senao o halt dispararia na operacao normal.
PRODUCTION_LEVERAGE = 2.0

# Banda de nao-trade: so negocia um ativo se |peso_alvo - peso_atual| passar disto.
# Mantem o turnover baixo (o rebalance e mensal; drift dentro da banda nao gera
# ordem). 1.5% e um meio-termo entre custo de transacao e aderencia ao alvo.
DEFAULT_NO_TRADE_BAND = Decimal("0.015")

# Mapa ticker-do-backtest (yfinance) -> simbolo-da-corretora (Alpaca).
# Cripto na Alpaca usa "BTC/USD"; o backtest usa "BTC-USD" (yfinance).
_BACKTEST_TO_BROKER = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}
_BROKER_TO_BACKTEST = {v: k for k, v in _BACKTEST_TO_BROKER.items()}

# Lote minimo por simbolo da corretora. Equities/ETFs => inteiro (lot None).
# Cripto => fracionavel com passo pequeno (alinha qty ao lote, sem rejeicao).
_CRYPTO_LOT = Decimal("0.0001")


def to_broker_symbol(backtest_ticker: str) -> str:
    """Ticker do backtest (yfinance) -> simbolo da corretora (Alpaca)."""
    return _BACKTEST_TO_BROKER.get(backtest_ticker, backtest_ticker)


def to_backtest_ticker(broker_symbol: str) -> str:
    """Simbolo da corretora -> ticker do backtest (inverso de to_broker_symbol)."""
    return _BROKER_TO_BACKTEST.get(broker_symbol, broker_symbol)


def _is_crypto_ticker(backtest_ticker: str) -> bool:
    return backtest_ticker in _BACKTEST_TO_BROKER


def beta_sleeve_equity(
    broker: BrokerClient, *, universe: list[str] | None = None
) -> Decimal:
    """Equity do SLEEVE do beta = soma do valor de mercado das posicoes do
    universo do beta (MF-4).

    Os dois books (acoes e beta) dividem UMA conta Alpaca; medir o DD-halt do
    beta sobre o equity TOTAL acopla os books (um tombo das acoes haltaria o
    rebalance do beta e vice-versa). Aqui isolamos: somamos so o mv das posicoes
    cujo simbolo pertence ao universo do beta (na forma do broker). Preco: o
    current_price da posicao (mark-to-market) quando disponivel; senao
    get_last_price; senao o avg_entry_price (fallback conservador).

    Antes do 1o rebalance (sem posicoes do beta) retorna 0 — e o guard fica inerte
    ate o sleeve ter valor real (o PortfolioRiskGuard ja ignora equity<=0)."""
    tickers = universe or [a.ticker for a in UNIVERSE]
    broker_symbols = {to_broker_symbol(t) for t in tickers}
    total = Decimal("0")
    try:
        positions = broker.get_positions()
    except Exception:  # noqa: BLE001 - sem posicoes legiveis => sleeve 0 (inerte)
        logger.warning("beta_sleeve_equity: falha ao ler posicoes; sleeve=0.")
        return Decimal("0")
    for pos in positions:
        if pos.symbol not in broker_symbols:
            continue
        price = getattr(pos, "current_price", None)
        if price is None or Decimal(str(price)) <= 0:
            try:
                price = broker.get_last_price(pos.symbol)
            except Exception:  # noqa: BLE001 - sem preco vivo => usa entrada
                price = pos.avg_entry_price
        total += Decimal(str(pos.qty)) * Decimal(str(price))
    return total


def beta_sleeve_pnl(
    broker: BrokerClient, *, universe: list[str] | None = None
) -> Decimal:
    """P&L NAO-realizado das posicoes do SLEEVE do beta (mark-to-market).

    P&L = soma de qty * (preco_atual - preco_medio_de_entrada) sobre as posicoes
    cujo simbolo pertence ao universo do beta. Preco: current_price (mtm) quando
    disponivel; senao get_last_price; senao o avg_entry_price (P&L 0 nesse nome).
    Em conta FLAT (sem posicoes do beta) retorna 0."""
    tickers = universe or [a.ticker for a in UNIVERSE]
    broker_symbols = {to_broker_symbol(t) for t in tickers}
    pnl = Decimal("0")
    try:
        positions = broker.get_positions()
    except Exception:  # noqa: BLE001 - sem posicoes legiveis => P&L 0
        logger.warning("beta_sleeve_pnl: falha ao ler posicoes; pnl=0.")
        return Decimal("0")
    for pos in positions:
        if pos.symbol not in broker_symbols:
            continue
        entry = Decimal(str(pos.avg_entry_price))
        price = getattr(pos, "current_price", None)
        if price is None or Decimal(str(price)) <= 0:
            try:
                price = broker.get_last_price(pos.symbol)
            except Exception:  # noqa: BLE001 - sem preco vivo => sem P&L nesse nome
                price = entry
        pnl += Decimal(str(pos.qty)) * (Decimal(str(price)) - entry)
    return pnl


def resolve_sleeve_capital(
    broker: BrokerClient, *, settings=None
) -> Decimal | None:
    """Caixa ABSOLUTO e FIXO alocado ao sleeve do beta, ou None p/ paper so-beta.

    NOVO-3 (simplificacao da raiz): so existe UM caso em que o sleeve tem capital
    proprio — quando BETA_SLEEVE_CAPITAL (absoluto, USD) e setado explicitamente
    (> 0). Esse valor e CAIXA FIXO no tempo: o NAV do sleeve = capital_fixo + P&L
    (sem double-count, porque o caixa fixo NAO inclui P&L). E o caso de coexistir
    com o book de acoes (carvar um sleeve limpo).

    Sem capital absoluto (o DEFAULT, paper SO-BETA), retorna None: NAO existe um
    "alocado" separado — o beta E a conta inteira, entao o NAV correto e o equity
    da conta (broker.get_account().equity), que ja contem o P&L UMA vez. Compor
    `pct*equity + P&L` aqui contava o P&L DUAS vezes (NOVO-3) — por isso o ramo
    pct*equity foi REMOVIDO. Ver resolve_sleeve_nav, que e a base de NAV de fato
    consumida pelo guard e pelo sizing."""
    from config.risk import get_beta_guard_settings

    bs = settings or get_beta_guard_settings()
    if bs.sleeve_capital and Decimal(str(bs.sleeve_capital)) > 0:
        return Decimal(str(bs.sleeve_capital))
    return None  # paper so-beta: sem caixa separado; usar o equity da conta.


def beta_sleeve_nav(
    broker: BrokerClient,
    allocated_capital: Decimal,
    *,
    universe: list[str] | None = None,
) -> Decimal:
    """NAV de um sleeve com caixa FIXO = capital ALOCADO (constante) + P&L do beta.

    SO vale quando `allocated_capital` e um CAIXA ABSOLUTO E FIXO (constante no
    tempo) — o caso de coexistir com o book de acoes. Como o caixa fixo NAO inclui
    P&L, somar o P&L uma vez da o NAV correto (sem double-count):
      - boot flat: caixa=alocado, posicoes=0   -> NAV = alocado            (>0, guard sadio)
      - apos comprar X de notional: caixa=alocado-X, mv=X -> NAV = alocado (compra nao muda equity)
      - posicoes andam +dP: mv = X+dP, caixa inalterado   -> NAV = alocado + dP

    ATENCAO (NOVO-3): NAO passe aqui pct*equity da conta — equity ja contem o P&L,
    e somar P&L de novo conta DUAS vezes. Em paper so-beta NAO use esta funcao;
    use o equity da conta direto (resolve_sleeve_nav cuida disso)."""
    return Decimal(str(allocated_capital)) + beta_sleeve_pnl(broker, universe=universe)


def resolve_sleeve_nav(
    broker: BrokerClient,
    *,
    settings=None,
    universe: list[str] | None = None,
) -> Decimal:
    """NAV do sleeve do beta — a UNICA base de NAV consumida pelo guard e pelo sizing.

    Esta funcao colapsa o que antes eram duas chamadas (resolve_sleeve_capital +
    beta_sleeve_nav) numa so primitiva CORRETA, eliminando o double-count do
    NOVO-3 (que vinha de compor pct*equity com +P&L):

      - CAPITAL ABSOLUTO setado (BETA_SLEEVE_CAPITAL > 0): NAV = caixa_fixo + P&L
        do beta. O caixa e constante, entao o P&L entra UMA vez. Isola o sleeve do
        book de acoes (coexistencia futura).
      - DEFAULT (paper SO-BETA, sem capital absoluto): NAV = broker.get_account()
        .equity. O beta E a conta inteira; o equity da conta JA e o NAV correto e
        ja contem o P&L UMA vez. Reutiliza a MESMA semantica do PortfolioRiskGuard
        das acoes (que mede sobre o equity da conta). Sem somar P&L, sem pct*equity.

    Em conta flat (boot/go-live) ambos os ramos dao um NAV POSITIVO (capital
    absoluto, ou o equity da conta com caixa), entao o guard nasce sadio (NOVO-1).
    Se o equity da conta for ilegivel no ramo so-beta, retorna 0 (o chamador trata
    como sleeve sem capital -> guard inerte / rebalance abortado)."""
    capital = resolve_sleeve_capital(broker, settings=settings)
    if capital is not None:
        # Caixa absoluto FIXO: NAV = caixa + P&L (uma vez). Sem double-count.
        return beta_sleeve_nav(broker, capital, universe=universe)
    # Paper so-beta: o equity da conta JA e o NAV correto (P&L contado uma vez).
    try:
        return Decimal(str(broker.get_account().equity))
    except Exception:  # noqa: BLE001 - sem equity legivel => sem NAV resolvido
        logger.warning("resolve_sleeve_nav: equity da conta ilegivel; nav=0.")
        return Decimal("0")


# ============================================================================
# PESOS-ALVO — a PONTE de paridade pesquisa<->producao.
# ============================================================================
def production_weight_frame(
    panel: pd.DataFrame,
    classes: dict[str, str],
    *,
    leverage: float = PRODUCTION_LEVERAGE,
) -> pd.DataFrame:
    """Vetor de pesos da ESTRATEGIA DE PRODUCAO sobre o painel INTEIRO.

    UMA fonte de verdade tanto p/ o backtest quanto p/ o calculo vivo:
      nucleo  = vol-target + hedge condicional (weights_volt_with_conditional_hedge,
                que ja e o vol-target confirmado com o de-risk de cauda da espec)
      overlay = alavancagem 2.0x (leverage_weights), com financiamento ja modelado
                no backtest (beta_v2). Aqui so escala o vetor de pesos.

    Cada construtor importado ja garante a ausencia de look-ahead (.shift(1)):
    a linha t usa apenas dados <= t-1. Esta funcao NAO corta o painel — ela
    devolve o frame inteiro; o corte por data e responsabilidade de quem chama
    (`target_weights_for_date` corta em asof p/ o caminho vivo)."""
    base = weights_volt_with_conditional_hedge(panel, classes)
    return leverage_weights(base, leverage)


def target_weights_for_date(
    panel: pd.DataFrame,
    classes: dict[str, str],
    asof: pd.Timestamp,
    *,
    leverage: float = PRODUCTION_LEVERAGE,
) -> dict[str, float]:
    """Pesos-alvo de HOJE (decisao tomada no fechamento de ONTEM), SEM look-ahead.

    `asof` = ultimo fechamento disponivel (ontem). Cortamos o painel em
    `index <= asof` ANTES de calcular os pesos — assim nenhum preco posterior a
    `asof` participa. Os construtores ainda aplicam .shift(1) internamente, de
    modo que a linha `asof` do vetor resultante so reflete dados <= asof-1.
    Pegamos a ULTIMA linha (asof) como o alvo a carregar a partir de hoje.

    Retorna {ticker_backtest: peso_alvo} (ex.: {"SPY": 0.18, "BTC-USD": 0.05}).
    A paridade com o backtest e garantida porque o backtest, sobre o painel
    inteiro, produz na linha `asof` exatamente este mesmo numero (o teste prova).
    """
    cut = panel.loc[panel.index <= asof]
    if cut.empty:
        return {}
    wf = production_weight_frame(cut, classes, leverage=leverage)
    if wf.empty:
        return {}
    row = wf.iloc[-1]
    return {str(k): float(v) for k, v in row.items()}


# ============================================================================
# ESTADO / GATILHO MENSAL — sem look-ahead.
# ============================================================================
_LAST_REBALANCE_MONTH_KEY = f"{STRATEGY_NAME}:last_rebalance_month"


def _period_month(asof: pd.Timestamp) -> str:
    """Rotulo de mes 'YYYY-MM' do asof (tz-naive p/ comparacao estavel)."""
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return f"{ts.year:04d}-{ts.month:02d}"


def _rebalance_timing_ready(panel: pd.DataFrame, asof: pd.Timestamp) -> bool:
    """True se a janela de TIMING do rebalance ja abriu para o mes de `asof` (MF-2).

    O backtest decide os pesos no 1o pregao do mes `r` (_rebalance_mask) mas, por
    causa do .shift(1) herdado dos construtores, esses pesos so ENTRAM EM VIGOR
    (e so sao negociados) no 2o pregao do mes `r+1`, usando dados de fechamento
    ate `r`. Logo, o caminho vivo so pode rebalancear a partir de `r+1` — antes
    disso (em `r`) o vetor de pesos ainda reflete o mes ANTERIOR e negociar levaria
    ao alvo VELHO, divergindo do backtest todo mes.

    Criterio panel-aware, sem look-ahead e robusto a restart: o mes de `asof` tem
    pelo menos 2 pregoes ATE (inclusive) `asof`. Isso e verdade exatamente a partir
    de `r+1` (e segue verdadeiro o mes inteiro — se o bot ficou fora no `r+1` e
    voltou em `r+5`, ainda rebalanceia, pois o estado mensal ainda esta em aberto).
    """
    if panel is None or panel.empty:
        # Sem painel para aferir o timing: nao deferimos (compat. legado/testes
        # que so exercitam o gatilho mensal por estado). O caminho vivo SEMPRE
        # passa o painel (run_beta_rebalance_cycle), entao o defer real vale la.
        return True
    idx = panel.index
    asof_ts = pd.Timestamp(asof)
    # Normaliza tz p/ comparacao estavel de periodo-mes (painel real e tz-aware
    # UTC; defendemos contra um painel tz-naive tambem).
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

    DUAS condicoes (ambas sem look-ahead):
      1. ESTADO: o mes de `asof` difere do mes do ultimo rebalance persistido
         (idempotencia mensal; 1a vez sem estado => devido). Sobrevive a restart.
      2. TIMING (MF-2): a janela do rebalance ja abriu — `asof` e >= o 2o pregao
         do mes (`r+1`), igual ao backtest (pesos entram em vigor em r+1). Aferido
         por `panel`; quando `panel` e None (uso legado/manual), o timing nao e
         imposto. `force` ignora AMBAS as condicoes (uso operacional/manual)."""
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
    leverage: float
    target_weights: dict[str, float]          # por ticker do backtest
    current_weights: dict[str, float]         # por ticker do backtest
    intents: list[OrderIntent] = field(default_factory=list)
    skipped_in_band: list[str] = field(default_factory=list)   # |dev| < banda
    blocked: list[str] = field(default_factory=list)           # vetado por guard
    notes: list[str] = field(default_factory=list)


class BetaRebalancer:
    """Calcula o plano de rebalance da estrategia de beta a partir do broker.

    NAO submete ordens (isso e o Executor). NAO baixa dados ao importar. O painel
    de precos historico (p/ os pesos) e injetado ou carregado de load_panel()
    (mesma fonte do backtest -> paridade). O broker da o equity e as posicoes
    atuais (mark-to-market = verdade da conta).
    """

    def __init__(
        self,
        broker: BrokerClient,
        *,
        state: StateRepository | None = None,
        guard: PortfolioRiskGuard | None = None,
        leverage: float = PRODUCTION_LEVERAGE,
        no_trade_band: Decimal = DEFAULT_NO_TRADE_BAND,
        universe: list[str] | None = None,
        sleeve_capital: Decimal | None = None,
    ) -> None:
        self._broker = broker
        self._state = state
        self._guard = guard
        self._leverage = leverage
        self._band = Decimal(str(no_trade_band))
        # Universo em tickers do backtest (yfinance). Default = UNIVERSE auditado.
        self._universe = universe or [a.ticker for a in UNIVERSE]
        self._classes = {a.ticker: a.klass for a in UNIVERSE}
        # MF-4 (latente) + NOVO-3: DIMENSIONAR sobre o NAV do SLEEVE do beta, NAO
        # sobre o equity TOTAL da conta. `sleeve_capital` (se passado) e um CAIXA
        # ABSOLUTO E FIXO => NAV = caixa + P&L (uma vez). `None` => resolve do
        # config no compute_plan: capital absoluto, ou o equity da conta direto em
        # paper so-beta (sem double-count do P&L; ver resolve_sleeve_nav).
        self._sleeve_capital = (
            Decimal(str(sleeve_capital)) if sleeve_capital is not None else None
        )

    # --- pesos atuais a partir do broker -----------------------------------
    def _current_weights(self, equity: Decimal) -> tuple[dict[str, float], dict[str, Decimal]]:
        """(peso_atual, preco_atual) por ticker do backtest, do broker.

        Peso = valor_de_mercado_da_posicao / equity. So considera simbolos do
        universo da estrategia (ignora posicoes de outras estrategias). Preco do
        broker quando disponivel (current_price; fallback get_last_price)."""
        weights: dict[str, float] = {t: 0.0 for t in self._universe}
        prices: dict[str, Decimal] = {}
        if equity <= 0:
            return weights, prices
        positions = {p.symbol: p for p in self._broker.get_positions()}
        for tkr in self._universe:
            sym = to_broker_symbol(tkr)
            pos = positions.get(sym)
            price = self._safe_price(sym, pos)
            if price is not None and price > 0:
                prices[tkr] = price
            if pos is None or price is None or price <= 0:
                continue
            mv = Decimal(str(pos.qty)) * price
            weights[tkr] = float(mv / equity)
        return weights, prices

    def _safe_price(self, broker_symbol: str, pos) -> Decimal | None:
        if pos is not None and getattr(pos, "current_price", None) is not None:
            return Decimal(str(pos.current_price))
        try:
            return Decimal(str(self._broker.get_last_price(broker_symbol)))
        except Exception as exc:  # noqa: BLE001 - sem preco -> trata como faltante
            logger.warning("Preco indisponivel p/ %s: %s", broker_symbol, exc)
            return None

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
        look-ahead. `equity`/`prices` opcionais p/ teste; senao lidos do broker.

        MF-4 (latente) + NOVO-3: quando `equity` nao e injetado, dimensiona sobre o
        NAV do SLEEVE do beta. Em paper SO-BETA (default), o NAV do sleeve E o
        equity da conta (P&L contado UMA vez) — identico ao de hoje, sem o
        double-count do NOVO-3. Com um capital ABSOLUTO injetado/config (book de
        acoes junto), o NAV = caixa_fixo + P&L (isola o sleeve da conta).
        """
        if equity is None:
            if self._sleeve_capital is not None:
                # Caixa absoluto FIXO injetado: NAV = caixa + P&L (uma vez).
                equity = beta_sleeve_nav(self._broker, self._sleeve_capital)
            else:
                # Resolve do config: capital absoluto + P&L, ou equity da conta
                # (paper so-beta). Uma fonte de NAV, sem double-count.
                equity = resolve_sleeve_nav(self._broker)

        target = target_weights_for_date(panel, self._classes, asof, leverage=self._leverage)
        current, broker_prices = self._current_weights(equity)
        prices = {**broker_prices, **(prices or {})}

        plan = RebalancePlan(
            asof=str(pd.Timestamp(asof).date()),
            equity=equity,
            leverage=self._leverage,
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
            self._plan_symbol(plan, tkr, target.get(tkr, 0.0), current.get(tkr, 0.0), equity, prices)

        # RESERVA DE CAIXA PARA CRIPTO (correcao de EXECUCAO, NAO de estrategia):
        # cripto na Alpaca e NAO-MARGINAVEL (cash-only). A 2.0x o book de acoes
        # consome margem E o caixa (colateral), zerando o caixa antes da cripto ->
        # a ordem de cripto e rejeitada server-side. As POSICOES-ALVO e a
        # alavancagem 2.0x ficam IDENTICAS ao backtest; so a ORDEM em que as
        # ordens saem muda. Financiamos a cripto com o CAIXA PRIMEIRO ordenando:
        #   1) SELLs (liberam caixa e margem),
        #   2) BUYs de CRIPTO (reservam o caixa enquanto ele existe),
        #   3) BUYs de ACOES (deployam no caixa restante + margem).
        # O Executor envia as intents nesta ordem -> a cripto cabe no caixa e as
        # acoes tomam margem. Mesmo NAV, mesma carteira final -> paridade intacta.
        plan.intents = self._order_for_cash_reservation(plan.intents)
        self._note_crypto_cash_shortfall(plan, equity, prices)
        return plan

    def _order_for_cash_reservation(self, intents: list[OrderIntent]) -> list[OrderIntent]:
        """Reordena as intents p/ financiar a cripto (cash-only) ANTES das acoes.

        Ordem estavel em 3 baldes: SELLs -> BUYs de cripto -> BUYs de acoes. NAO
        altera quais ordens existem nem suas quantidades (a estrategia/pesos ficam
        intactos) — so a SEQUENCIA de envio, p/ o caixa nao ser consumido pelas
        acoes (marginaveis) antes da cripto (nao-marginavel)."""
        sells = [i for i in intents if i.side == OrderSide.SELL]
        crypto_buys = [
            i for i in intents
            if i.side == OrderSide.BUY and _is_crypto_ticker(to_backtest_ticker(i.symbol))
        ]
        stock_buys = [
            i for i in intents
            if i.side == OrderSide.BUY and not _is_crypto_ticker(to_backtest_ticker(i.symbol))
        ]
        return sells + crypto_buys + stock_buys

    def _note_crypto_cash_shortfall(
        self, plan: RebalancePlan, equity: Decimal, prices: dict[str, Decimal]
    ) -> None:
        """Trata GRACIOSO o caso (nao esperado, cripto-alvo ~20% << caixa 100%) em
        que o caixa NAO-MARGINAVEL nao cobre as compras de cripto. Apenas REGISTRA
        uma nota auditavel (nao quebra, nao re-dimensiona) — o pre-trade do Executor
        e o broker ainda barram a sobra. Em operacao normal a cripto cabe no caixa,
        entao esta nota nunca aparece; serve de fail-safe explicito."""
        try:
            account = self._broker.get_account()
            cash = Decimal(str(account.non_marginable_buying_power))
        except Exception:  # noqa: BLE001 - sem conta legivel => nao bloqueia o plano
            return
        crypto_buy_notional = Decimal("0")
        for i in plan.intents:
            if i.side != OrderSide.BUY:
                continue
            if not _is_crypto_ticker(to_backtest_ticker(i.symbol)):
                continue
            price = prices.get(to_backtest_ticker(i.symbol))
            if price is not None and price > 0:
                crypto_buy_notional += i.qty * Decimal(str(price))
        if crypto_buy_notional > cash:
            plan.notes.append(
                f"cripto-alvo (compra ~{crypto_buy_notional}) excede o caixa "
                f"nao-marginavel ({cash}); parte da cripto pode ser barrada "
                "(cash-only). Esperado apenas em caixa anomalo."
            )

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

        fractional = _is_crypto_ticker(tkr)
        lot = _CRYPTO_LOT if fractional else None
        raw_qty = abs(delta_notional) / price
        qty = round_qty(raw_qty, lot, fractional=fractional)
        if qty <= 0:
            plan.skipped_in_band.append(tkr)  # delta abaixo de 1 unidade negociavel
            return

        # GUARD: ordens que AUMENTAM exposicao do simbolo passam pelo can_open
        # (halt global / teto por simbolo). Reducoes (des-risca) sempre passam.
        if self._guard is not None and side == OrderSide.BUY:
            target_symbol_exposure = abs(Decimal(str(w_target)))
            ok, reason = self._guard.can_open(target_symbol_exposure, Decimal("0"))
            if not ok:
                plan.blocked.append(f"{tkr} ({reason})")
                return

        plan.intents.append(
            OrderIntent(
                symbol=to_broker_symbol(tkr),
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                strategy=STRATEGY_NAME,
            )
        )

    # --- orquestracao do rebalance (calculo apenas; execucao e do Executor) -
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

        NAO marca o mes como rebalanceado (quem executa chama mark_rebalanced
        apos o Executor confirmar). Assim, se o ciclo cair antes de executar, o
        rebalance volta a ser devido no proximo tick.

        MF-2: passa `panel` ao gatilho p/ deferir o rebalance ao 2o pregao do mes
        (r+1), o dia em que os pesos do backtest entram em vigor."""
        if not is_rebalance_due(self._state, asof, force=force, panel=panel):
            return None
        return self.compute_plan(panel, asof, equity=equity, prices=prices)


def load_production_panel(force: bool = False) -> tuple[pd.DataFrame, dict[str, str]]:
    """Painel de precos da estrategia (MESMA fonte do backtest -> paridade).

    Reusa simulation.beta_portfolio.load_panel (yfinance cacheado). Em producao
    real, este e o ponto a trocar por um feed de dados ao vivo equivalente —
    mantendo a IDENTIDADE de calculo de pesos com o backtest."""
    return load_panel(force=force)
