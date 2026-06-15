"""Testes da costura de orquestracao plugavel (factory + OpenSquad adapter)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agents.executor import Executor
from agents.planner import Planner
from config.watchlist import Watchlist, WatchlistItem
from data.db import Database
from feedback.decision_log import DecisionLog
from intelligence.engine import DecisionIntelligence
from orchestration.bus_orchestrator import BusOrchestrator
from orchestration.factory import build_orchestrator
from orchestration.local_orchestrator import LocalOrchestrator
from orchestration.opensquad_orchestrator import (
    OpenSquadOrchestrator,
    OrchestratorBridge,
)
from strategies.trailing_stop import TrailingStopStrategy


@pytest.fixture
def planner_executor(broker, state, trade_logger, kill_switch):
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    planner = Planner(broker, state, wl, [TrailingStopStrategy()])
    executor = Executor(broker, trade_logger, kill_switch)
    return planner, executor


@pytest.fixture
def bus_deps(broker):
    """Dependencias extras que o orquestrador 'bus' (default) exige."""
    log = DecisionLog(connection=Database(":memory:").conn)
    intel = DecisionIntelligence(log, price_provider=broker.get_last_price)
    return dict(broker=broker, intelligence=intel, decision_log=log, symbols=["AAPL"])


# --- Factory ----------------------------------------------------------------
def test_factory_default_is_bus(planner_executor, bus_deps):
    planner, executor = planner_executor
    orch = build_orchestrator(None, planner, executor, **bus_deps)
    assert isinstance(orch, BusOrchestrator)


def test_factory_bus_without_deps_raises(planner_executor):
    planner, executor = planner_executor
    with pytest.raises(ValueError) as exc:
        build_orchestrator("bus", planner, executor)
    assert "bus" in str(exc.value).lower()


def test_factory_local_explicit(planner_executor):
    planner, executor = planner_executor
    assert isinstance(build_orchestrator("local", planner, executor), LocalOrchestrator)


def test_factory_opensquad(planner_executor):
    planner, executor = planner_executor
    orch = build_orchestrator("opensquad", planner, executor)
    assert isinstance(orch, OpenSquadOrchestrator)


def test_factory_unknown_raises(planner_executor):
    planner, executor = planner_executor
    with pytest.raises(ValueError):
        build_orchestrator("crewai", planner, executor)


# --- OpenSquad adapter ------------------------------------------------------
def test_opensquad_without_bridge_raises_clear_error(planner_executor):
    planner, executor = planner_executor
    orch = OpenSquadOrchestrator(planner, executor)  # sem bridge
    with pytest.raises(NotImplementedError) as exc:
        orch.run_cycle()
    assert "bridge" in str(exc.value).lower()


class _ApproveAllBridge(OrchestratorBridge):
    def should_run_cycle(self) -> bool:
        return True

    def approve(self, intents):
        return intents


class _RejectAllBridge(OrchestratorBridge):
    def should_run_cycle(self) -> bool:
        return True

    def approve(self, intents):
        return []  # checkpoint humano reprova tudo


def test_opensquad_with_bridge_runs_and_executes(broker, state, trade_logger, kill_switch):
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "50")  # dispara trailing
    planner = Planner(broker, state, wl, [TrailingStopStrategy()])
    executor = Executor(broker, trade_logger, kill_switch)

    orch = OpenSquadOrchestrator(planner, executor, bridge=_ApproveAllBridge())
    result = orch.run_cycle()
    assert result.intents_count == 1
    assert result.executed_count == 1


def test_opensquad_checkpoint_can_reject(broker, state, trade_logger, kill_switch):
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "50")
    planner = Planner(broker, state, wl, [TrailingStopStrategy()])
    executor = Executor(broker, trade_logger, kill_switch)

    orch = OpenSquadOrchestrator(planner, executor, bridge=_RejectAllBridge())
    result = orch.run_cycle()
    # planejou 1, mas o checkpoint reprovou => nada executado
    assert result.intents_count == 1
    assert result.executed_count == 0
    assert broker.submitted == []
