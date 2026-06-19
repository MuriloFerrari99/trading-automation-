"""Entrypoint do MVP (Fase 1).

Monta a aplicacao (config -> broker -> dados -> agentes -> orquestrador ->
monitor) e inicia o Monitor agendado.

Uso:
    uv run python main.py            # inicia o Monitor (loop agendado)
    uv run python main.py --once     # roda um unico ciclo e sai (diagnostico)
    uv run python main.py --check    # so valida conexao/conta e sai
"""

from __future__ import annotations

import argparse
import logging
import sys

from agents.executor import Executor
from agents.monitor import Monitor
from agents.planner import Planner
from broker.base import BrokerClient
from config.risk import get_risk_settings
from config.settings import LiveTradingBlockedError, get_settings
from config.watchlist import load_watchlist
from core.kill_switch import KillSwitch
from core.market_clock import MarketClock
from data.audit_log import AuditLog
from data.db import Database
from data.order_repo import OrderRepository
from data.position_repo import PositionRepository
from data.signal_repo import SignalRepository
from data.state_repo import StateRepository
from data.trade_logger import TradeLogger
from orchestration.base import AgentOrchestrator
from feedback.decision_log import DecisionLog
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence
from orchestration.factory import build_orchestrator
from orchestration.reconcile import reconcile
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard
from strategies.ladder_buys import LadderBuysStrategy
from strategies.signals.base import SignalService
from strategies.signals.congress import CongressTradingProvider, StaticCongressSource
from strategies.signals.smart_money import SmartMoneyProvider, StaticSmartMoneySource
from strategies.trailing_stop import TrailingStopStrategy
from strategies.wheel import WheelStrategy

logger = logging.getLogger("main")

MONITOR_INTERVAL_MINUTES = 10


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def build_broker(settings) -> BrokerClient:
    # Import tardio: so carrega o SDK quando realmente formos a corretora real.
    from broker.alpaca_broker import AlpacaBroker

    return AlpacaBroker(settings)


def _build_risk_manager(broker, state, audit, *, sizer=None) -> RiskManager:
    """Monta o RiskManager fixando o equity de inicio do dia (persistido).

    `sizer` (DynamicSizer) opcional liga o sizing por conviccao: quando presente,
    a fracao de risco-por-trade passa a ser dirigida pela confianca da decisao.
    """
    from datetime import datetime, timezone
    from decimal import Decimal

    rs = get_risk_settings()
    equity = broker.get_account().equity
    today = datetime.now(timezone.utc).date().isoformat()
    key = f"risk_start_equity:{today}"
    stored = state.get_decimal(key)
    if stored is None:
        state.set_decimal(key, equity)
        start_equity = equity
    else:
        start_equity = stored

    # Pico (high-water) persistido GLOBALMENTE (nao por dia): o gate de max
    # drawdown e pico-a-vale e deve sobreviver a restart intradiario.
    peak_key = "risk_peak_equity"
    stored_peak = state.get_decimal(peak_key)
    peak_equity = max(stored_peak or Decimal("0"), start_equity, equity)
    state.set_decimal(peak_key, peak_equity)

    guard = PortfolioRiskGuard(
        start_equity=start_equity if start_equity > 0 else Decimal("1"),
        daily_loss_pct=rs.daily_loss_limit_pct,
        max_dd_pct=rs.max_drawdown_pct,
        max_per_symbol_pct=rs.max_per_symbol_pct,
        max_heat_pct=rs.max_portfolio_heat_pct,
        peak_equity=peak_equity if peak_equity > 0 else None,
        on_peak_update=lambda p: state.set_decimal(peak_key, p),
    )
    audit.write(
        "system", "risk_init",
        payload={"start_equity": str(start_equity), "peak_equity": str(peak_equity)},
    )
    return RiskManager(rs, guard, sizer=sizer)


def _build_beta_guard(broker, state, audit):
    """PortfolioRiskGuard calibrado ao perfil de BETA (ancorado no beta_v2).

    ADITIVO: separado do guard das estrategias de acoes (_build_risk_manager).

    MF-4 + NOVO-1: o envelope de risco do beta e ISOLADO do das acoes. Tem chaves
    de estado PROPRIAS ('beta:risk_*') e mede DD/perda-diaria sobre o NAV do
    SLEEVE do beta = CAPITAL ALOCADO (caixa) + P&L das posicoes do beta — NAO
    sobre o market value das posicoes (que e 0 no boot da conta flat, o estado de
    go-live) e NAO sobre o equity TOTAL da conta.

    POR QUE (NOVO-1, regressao do fix do MF-4): antes a base era o mv das posicoes
    (beta_sleeve_equity). Em conta flat isso e 0; o codigo coagia start=1 e chamava
    update(0) => daily_pl=(0-1)/1=-100% => HALT fantasma e pegajoso => a 1a
    rebalanceada bloqueava tudo e o book nunca era construido. Agora a base e o NAV
    do sleeve: no boot flat, NAV = capital alocado (positivo) => guard sadio/inerte
    (daily_pl=0), e a 1a rebalanceada GERA ordens. O halt so engaja apos perdas
    REAIS do sleeve alem do limite. O start/peak por dia/high-water sobrevivem a
    restart."""
    from datetime import datetime, timezone
    from decimal import Decimal

    from config.risk import get_beta_guard_settings
    from strategies.beta_rebalancer import resolve_sleeve_nav

    bs = get_beta_guard_settings()
    # NOVO-3: NAV do sleeve numa SO primitiva (sem double-count). Em paper so-beta
    # (default) e o equity da conta direto (P&L contado UMA vez); com capital
    # absoluto e caixa_fixo + P&L. No boot flat o NAV e POSITIVO -> guard sadio.
    sleeve_nav = resolve_sleeve_nav(broker, settings=bs)  # NAV DO BETA, nao da conta
    today = datetime.now(timezone.utc).date().isoformat()

    # Chaves PROPRIAS do beta (isolam o envelope do guard das acoes).
    start_key = f"beta:risk_start_equity:{today}"
    peak_key = "beta:risk_peak_equity"

    # start_equity = NAV do sleeve no inicio do dia (persistido). No boot flat,
    # NAV = equity da conta / capital absoluto (positivo) -> guard NASCE sadio.
    #
    # NOVO-4 (saneamento): um valor persistido <= 0 e LIXO (residuo da versao
    # buggada do NOVO-1, que gravava 0). Coagir esse 0 p/ start=1 NEUTRALIZA o gate
    # de perda diaria do dia (denominador 1). Em vez disso, RE-DERIVAMOS do NAV do
    # boot e LIMPAMOS a chave stale (re-grava o NAV correto). So tratamos None como
    # "primeiro boot do dia" (grava o NAV); <= 0 e tratado como ausente.
    start_equity = state.get_decimal(start_key)
    if start_equity is None or start_equity <= 0:
        if start_equity is not None and start_equity <= 0:
            logger.warning(
                "beta_guard: start_equity persistido invalido (%s) p/ %s; "
                "re-derivando do NAV do boot (%s) e limpando a chave stale.",
                start_equity, start_key, sleeve_nav,
            )
        state.set_decimal(start_key, sleeve_nav)
        start_equity = sleeve_nav

    stored_peak = state.get_decimal(peak_key)
    peak_equity = max(stored_peak or Decimal("0"), start_equity, sleeve_nav)
    state.set_decimal(peak_key, peak_equity)

    guard = PortfolioRiskGuard(
        # NAV do sleeve e POSITIVO no boot (equity da conta / capital absoluto),
        # e start_equity foi re-derivado se vinha <= 0 (NOVO-4). Mantemos o
        # fallback p/ 1 SO p/ o caso degenerado de NAV de boot 0 (conta sem caixa
        # nem posicoes) — onde o gate diario fica inerte por nao haver capital.
        start_equity=start_equity if start_equity > 0 else Decimal("1"),
        daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct,
        max_per_symbol_pct=bs.max_per_symbol_pct,
        max_heat_pct=bs.max_portfolio_heat_pct,
        peak_equity=peak_equity if peak_equity > 0 else None,
        on_peak_update=lambda p: state.set_decimal(peak_key, p),
    )
    # So engaja o halt no boot com um NAV REALMENTE resolvido (> 0). Um NAV <= 0
    # significa conta ILEGIVEL (resolve_sleeve_nav engoliu uma falha transitoria
    # de get_account e devolveu 0) OU conta vazia — em ambos o gate diario fica
    # INERTE, nao halta. Sem esta guarda, um hiccup do broker no boot dispara
    # daily_pl=(0-1)/1=-100% e CONGELA o book pela sessao (regressao NOVO-1 pelo
    # caminho da excecao). O 1o ciclo com NAV real engaja o guard normalmente.
    if sleeve_nav > 0:
        guard.update(sleeve_nav)
    audit.write(
        "system", "beta_guard_init",
        payload={
            "max_dd": str(bs.max_drawdown_pct),
            "sleeve_nav": str(sleeve_nav),
            "start_equity": str(start_equity),
            "halt": guard.trading_halted,
        },
    )
    return guard


class _BetaRebalanceOrchestrator(AgentOrchestrator):
    """Wrapper ADITIVO: delega o ciclo ao orquestrador base e, em seguida, roda o
    rebalance mensal do beta (no-op quando nao e devido / flag off).

    Mantem o contrato AgentOrchestrator intacto — para o Monitor e so um
    orquestrador. NAO altera o pipeline de acoes existente; apenas adiciona um
    passo que so age uma vez por mes e so quando a flag esta ligada."""

    def __init__(self, inner, broker, executor, *, state=None, guard=None) -> None:
        self._inner = inner
        self._broker = broker
        self._executor = executor
        self._state = state
        self._guard = guard

    def run_cycle(self):
        result = self._inner.run_cycle()
        try:
            from strategies.beta_live import run_beta_rebalance_cycle
            from strategies.beta_rebalancer import resolve_sleeve_nav

            # MF-4 + NOVO-1 + NOVO-3: atualiza o DD-halt do beta com o NAV do
            # SLEEVE. Em paper SO-BETA (default) o NAV E o equity da conta (P&L
            # contado UMA vez) — sem o double-count do NOVO-3, entao o guard ve o
            # drawdown REAL (1x). Com capital absoluto, NAV = caixa_fixo + P&L
            # (isola o sleeve do book de acoes). Em conta flat o NAV e positivo
            # (sem halt fantasma).
            if self._guard is not None:
                # Mesma defesa do boot: NAV <= 0 = leitura transitoria falha (ou
                # conta vazia); NAO alimenta o guard, senao daily_pl=-100% halta o
                # book num hiccup do broker no meio da sessao.
                _nav = resolve_sleeve_nav(self._broker)
                if _nav > 0:
                    self._guard.update(_nav)
            run_beta_rebalance_cycle(
                self._broker, self._executor, state=self._state, guard=self._guard
            )
        except Exception:  # noqa: BLE001 - rebalance aditivo nao derruba o ciclo base
            logger.exception("Ciclo de rebalance de beta falhou (ignorado).")
        return result


def build_app(
    broker: BrokerClient, orchestrator_name: str = "local"
) -> tuple[Monitor, AgentOrchestrator]:
    db = Database()
    state = StateRepository(db)
    trade_logger = TradeLogger(db)
    signal_repo = SignalRepository(db)
    order_repo = OrderRepository(db)
    position_repo = PositionRepository(db)
    audit = AuditLog(db)
    kill_switch = KillSwitch()
    watchlist = load_watchlist()

    # Reconciliacao no boot: broker = fonte de verdade, ANTES de qualquer ciclo.
    try:
        reconcile(broker, order_repo, position_repo, audit)
    except Exception:
        logger.exception("Falha na reconciliacao de boot; seguindo com cautela.")

    # Sinais (Nivel 2): providers atras de fontes swappable. Por padrao usam
    # fontes estaticas VAZIAS (nenhum sinal real) — seguro e pronto p/ plugar
    # uma fonte real (Capital Trades, 13F, etc.) sem mexer no resto.
    signal_service = SignalService(
        [
            CongressTradingProvider(StaticCongressSource()),
            SmartMoneyProvider(StaticSmartMoneySource()),
        ]
    )

    # Sizing por conviccao (opcional, por config). Liga o "cerebro" ao caminho
    # vivo: a confianca (regime + sinais + ML champion) dirige o risco-por-trade
    # via DynamicSizer, sempre DENTRO dos caps do RiskManager. Sem FIMATHE aqui.
    rs = get_risk_settings()
    sizer = None
    enricher = None
    if rs.conviction_sizing:
        from config.risk import build_dynamic_sizer
        from integration.enricher import DecisionEnricher
        from ml.model_store import load_promoted_classifier

        sizer = build_dynamic_sizer(rs)
        champion = load_promoted_classifier()  # None => ML em shadow (no-op)
        enricher = DecisionEnricher(sizer=sizer, classifier=champion)
        logger.info(
            "Sizing por conviccao ATIVO (max %.1f%%/trade, floor %.2f) | ML champion: %s",
            float(rs.conviction_max_risk_pct) * 100,
            rs.conviction_confidence_floor,
            "carregado" if champion is not None else "ausente (shadow)",
        )

    # Camada de risco: circuit breakers + sizing. start_equity do dia persistido.
    risk_manager = _build_risk_manager(broker, state, audit, sizer=sizer)

    # MF-1: quando o book de beta esta vivo, o universo do beta e GERIDO pelo
    # rebalanceador (vol-target/rebalance, SEM stop por ativo). O TrailingStop do
    # pipeline-base protege TODO long do broker — INCLUSIVE o book do beta — e o
    # liquidaria numa queda de 10%, corrompendo a tese e o track record. Isentamos
    # esses simbolos (na FORMA DO BROKER, ex.: 'BTC/USD') do trailing. Com a flag
    # off, o set fica vazio e o comportamento e o legado.
    beta_excluded_symbols: set[str] = set()
    try:
        from strategies.beta_live import beta_live_enabled as _beta_on

        if _beta_on():
            from strategies.beta_rebalancer import UNIVERSE as _BETA_UNIVERSE
            from strategies.beta_rebalancer import to_broker_symbol as _to_broker

            beta_excluded_symbols = {_to_broker(a.ticker) for a in _BETA_UNIVERSE}
    except Exception:  # noqa: BLE001 - resolucao do universo do beta nunca quebra o boot
        logger.exception("Falha ao resolver universo do beta p/ isentar trailing (ignorado).")

    # WheelStrategy so age em ativos com `wheel` na watchlist E se o gate de
    # elegibilidade (nivel de opcoes + liquidez) passar em runtime.
    # TrailingStop protege TODO long sem protecao (default da config de risco),
    # EXCETO os simbolos do book de beta (MF-1).
    strategies = [
        TrailingStopStrategy(
            get_risk_settings().default_trailing_stop_pct,
            excluded_symbols=beta_excluded_symbols,
        ),
        LadderBuysStrategy(),
        WheelStrategy(),
    ]
    planner = Planner(
        broker,
        state,
        watchlist,
        strategies,
        signal_service=signal_service,
        signal_repo=signal_repo,
        risk_manager=risk_manager,
        audit=audit,
        enricher=enricher,
        assumed_stop_pct=rs.assumed_stop_pct,
    )
    executor = Executor(
        broker, trade_logger, kill_switch, order_repo=order_repo, audit=audit
    )

    # Camada de decisao (loop de feedback + gate por regime). Registra TODA
    # decisao e veta combos estrategia@regime com edge negativo comprovado.
    # Compartilha a conexao do Database (tabela `decisions` no mesmo SQLite).
    decision_log = DecisionLog(connection=db.conn)
    intelligence = DecisionIntelligence(
        decision_log, DecisionPolicy(), price_provider=broker.get_last_price
    )

    # Orquestrador selecionado por config (factory). Default "bus": pipeline de
    # agentes que FECHA o loop de feedback, classifica regime por barras reais e
    # reconcilia periodicamente. As deps abaixo so sao usadas pelo modo "bus".
    orchestrator = build_orchestrator(
        orchestrator_name,
        planner,
        executor,
        intelligence=intelligence,
        broker=broker,
        decision_log=decision_log,
        symbols=watchlist.symbols(),
        order_repo=order_repo,
        position_repo=position_repo,
        audit=audit,
    )

    clock = MarketClock(broker)

    # --- Plug ADITIVO da estrategia de beta, atras de FLAG (desligada por padrao).
    # Enquanto BETA_LIVE_ENABLED estiver off, NADA disto roda: sem rebalance, sem
    # captura, sem efeito no sistema Alpaca existente. Ver strategies/beta_live.py.
    on_schedule_start = None
    beta_orchestrator = orchestrator
    try:
        from strategies.beta_live import (
            beta_live_enabled,
            schedule_eod_capture,
            schedule_panel_refresh,
        )

        if beta_live_enabled():
            beta_guard = _build_beta_guard(broker, state, audit)
            # (a) Rebalance mensal: roda no tick do Monitor (no-op quando nao e
            #     devido), executando as ordens de diferenca pelo Executor.
            beta_orchestrator = _BetaRebalanceOrchestrator(
                orchestrator, broker, executor, state=state, guard=beta_guard
            )

            # (b) Jobs agendados quando o Monitor inicia (ambos atras da flag):
            #     - captura EOD pos-fechamento (snapshot de NAV);
            #     - NOVO-2: refresh diario do painel de precos (pre-mercado), p/ o
            #       asof avancar e o gatilho mensal nao congelar no cache velho.
            def _on_start(sched):
                schedule_eod_capture(sched, broker, db_conn=db.conn, state=state)
                schedule_panel_refresh(sched)

            on_schedule_start = _on_start
            logger.info(
                "BETA LIVE ATIVO (flag BETA_LIVE_ENABLED): rebalance mensal + captura EOD "
                "+ refresh diario de painel ligados."
            )
    except Exception:  # noqa: BLE001 - plug aditivo nunca quebra o boot do sistema base
        logger.exception("Falha ao montar o plug de beta (ignorado); sistema base segue.")

    monitor = Monitor(
        beta_orchestrator,
        clock,
        interval_minutes=MONITOR_INTERVAL_MINUTES,
        run_when_closed=watchlist.has_crypto(),  # cripto opera 24/7
        on_schedule_start=on_schedule_start,
    )
    return monitor, beta_orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sistema de automacao de investimentos (paper)")
    parser.add_argument("--once", action="store_true", help="roda um unico ciclo e sai")
    parser.add_argument("--check", action="store_true", help="valida conexao/conta e sai")
    args = parser.parse_args(argv)

    _configure_logging()

    try:
        settings = get_settings()
    except LiveTradingBlockedError as exc:
        logger.error("GUARD DE SEGURANCA: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Falha ao carregar configuracao: %s", exc)
        return 2

    logger.info(
        "Modo paper trading. Endpoint: %s | orquestrador: %s",
        settings.alpaca_endpoint, settings.orchestrator,
    )

    broker = build_broker(settings)

    if args.check:
        account = broker.get_account()
        logger.info(
            "Conta OK | equity=%s cash=%s buying_power=%s options_level=%s | mercado_aberto=%s",
            account.equity, account.cash, account.buying_power,
            account.options_level, broker.is_market_open(),
        )
        return 0

    monitor, _ = build_app(broker, settings.orchestrator)

    if args.once:
        monitor.tick()
        return 0

    monitor.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
