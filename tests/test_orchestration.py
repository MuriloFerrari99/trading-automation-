"""Testes de integracao do ciclo: Planner -> Orchestrator -> Executor + Monitor."""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from agents.monitor import Monitor
from agents.planner import Planner
from core.market_clock import MarketClock
from orchestration.local_orchestrator import LocalOrchestrator
from strategies.trailing_stop import TrailingStopStrategy


def _build(broker, state, trade_logger, kill_switch, watchlist):
    planner = Planner(broker, state, watchlist, [TrailingStopStrategy()])
    executor = Executor(broker, trade_logger, kill_switch)
    return LocalOrchestrator(planner, executor)


def test_full_cycle_places_native_trailing(broker, state, trade_logger, kill_switch, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "120")
    orch = _build(broker, state, trade_logger, kill_switch, watchlist)

    # primeiro ciclo: coloca a trailing stop nativa (fica aberta no broker)
    r1 = orch.run_cycle()
    assert r1.intents_count == 1
    assert r1.executed_count == 1
    assert len(broker.get_open_orders()) == 1  # protecao colocada

    # segundo ciclo: ja protegido => nao reenvia
    r2 = orch.run_cycle()
    assert r2.intents_count == 0


def test_monitor_skips_when_market_closed(broker, state, trade_logger, kill_switch, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "50")  # dispararia se rodasse
    broker.set_market_open(False)
    orch = _build(broker, state, trade_logger, kill_switch, watchlist)
    monitor = Monitor(orch, MarketClock(broker), interval_minutes=10)

    result = monitor.tick()
    assert result is None  # ciclo pulado
    assert broker.submitted == []  # nada executado com mercado fechado


def test_monitor_runs_when_market_open(broker, state, trade_logger, kill_switch, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "50")
    broker.set_market_open(True)
    orch = _build(broker, state, trade_logger, kill_switch, watchlist)
    monitor = Monitor(orch, MarketClock(broker), interval_minutes=10)

    result = monitor.tick()
    assert result is not None
    assert result.executed_count == 1
