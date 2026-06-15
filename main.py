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
from config.settings import LiveTradingBlockedError, get_settings
from config.watchlist import load_watchlist
from core.kill_switch import KillSwitch
from core.market_clock import MarketClock
from data.db import Database
from data.state_repo import StateRepository
from data.trade_logger import TradeLogger
from orchestration.local_orchestrator import LocalOrchestrator
from strategies.trailing_stop import TrailingStopStrategy

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


def build_app(broker: BrokerClient) -> tuple[Monitor, LocalOrchestrator]:
    db = Database()
    state = StateRepository(db)
    trade_logger = TradeLogger(db)
    kill_switch = KillSwitch()
    watchlist = load_watchlist()

    strategies = [TrailingStopStrategy()]
    planner = Planner(broker, state, watchlist, strategies)
    executor = Executor(broker, trade_logger, kill_switch)
    orchestrator = LocalOrchestrator(planner, executor)

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

    logger.info("Modo paper trading. Endpoint: %s", settings.alpaca_endpoint)

    broker = build_broker(settings)

    if args.check:
        account = broker.get_account()
        logger.info(
            "Conta OK | equity=%s cash=%s buying_power=%s options_level=%s | mercado_aberto=%s",
            account.equity, account.cash, account.buying_power,
            account.options_level, broker.is_market_open(),
        )
        return 0

    monitor, _ = build_app(broker)

    if args.once:
        monitor.tick()
        return 0

    monitor.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
