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


def _build_risk_manager(broker, state, audit) -> RiskManager:
    """Monta o RiskManager fixando o equity de inicio do dia (persistido)."""
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
    guard = PortfolioRiskGuard(
        start_equity=start_equity if start_equity > 0 else Decimal("1"),
        daily_loss_pct=rs.daily_loss_limit_pct,
        max_dd_pct=rs.max_drawdown_pct,
        max_per_symbol_pct=rs.max_per_symbol_pct,
        max_heat_pct=rs.max_portfolio_heat_pct,
    )
    audit.write("system", "risk_init", payload={"start_equity": str(start_equity)})
    return RiskManager(rs, guard)


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

    # Camada de risco: circuit breakers + sizing. start_equity do dia persistido.
    risk_manager = _build_risk_manager(broker, state, audit)

    # WheelStrategy so age em ativos com `wheel` na watchlist E se o gate de
    # elegibilidade (nivel de opcoes + liquidez) passar em runtime.
    strategies = [TrailingStopStrategy(), LadderBuysStrategy(), WheelStrategy()]
    planner = Planner(
        broker,
        state,
        watchlist,
        strategies,
        signal_service=signal_service,
        signal_repo=signal_repo,
        risk_manager=risk_manager,
        audit=audit,
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

    # Orquestrador selecionado por config (factory). Trocar p/ OpenSquad depois
    # e so mudar ORCHESTRATOR no .env (e fornecer um OrchestratorBridge).
    orchestrator = build_orchestrator(
        orchestrator_name, planner, executor, intelligence=intelligence
    )

    clock = MarketClock(broker)
    monitor = Monitor(orchestrator, clock, interval_minutes=MONITOR_INTERVAL_MINUTES)
    return monitor, orchestrator


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
