"""Pipeline bus end-to-end e fechamento do loop pelo FeedbackAgent."""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from agents.feedback_agent import FeedbackAgent
from agents.planner import Planner
from broker.fake_broker import FakeBroker
from config.watchlist import Watchlist, WatchlistItem
from core.kill_switch import KillSwitch
from data.db import Database
from data.state_repo import StateRepository
from data.trade_logger import TradeLogger
from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, OutcomeStatus
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence
from orchestration.bus import InMemoryBus
from orchestration.bus_orchestrator import assemble_bus_orchestrator
from strategies.trailing_stop import TrailingStopStrategy


def test_bus_pipeline_runs_end_to_end():
    broker = FakeBroker(cash=Decimal("100000"), prices={"AAPL": Decimal("105")})
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    db = Database(":memory:")
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    planner = Planner(broker, StateRepository(db), wl, [TrailingStopStrategy()])
    executor = Executor(broker, TradeLogger(db), KillSwitch("/tmp/__no_kill_bus__"))
    decision_log = DecisionLog(connection=db.conn)
    intel = DecisionIntelligence(decision_log, DecisionPolicy(), price_provider=broker.get_last_price)

    orch = assemble_bus_orchestrator(
        broker=broker, planner=planner, executor=executor, intelligence=intel,
        decision_log=decision_log, symbols=["AAPL"],
    )
    result = orch.run_cycle()

    # Trailing nativo colocado, aprovado pela decisao (sem historico => permite).
    assert result.intents_count == 1
    assert result.executed_count == 1
    assert len(broker.get_open_orders()) == 1
    # A decisao foi registrada no log (loop de feedback).
    assert len(decision_log.all_records()) == 1


def test_feedback_agent_closes_loop_on_sell():
    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    # decisao de COMPRA aberta a 100
    dec_id = log.record(
        Decision(strategy="t", symbol="AAPL", action=DecisionAction.BUY,
                 regime=MarketRegime.UNKNOWN, reference_price=Decimal("100"))
    )
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    # chega um fill de VENDA a 110 (lucro)
    bus.publish("fills", {"symbol": "AAPL", "side": "sell",
                          "filled_qty": Decimal("10"), "fill_price": Decimal("110")})
    agent.step(bus)

    rows = log.all_records()
    row = next(r for r in rows if r["id"] == dec_id)
    assert row["outcome_status"] == OutcomeStatus.WIN.value
    assert Decimal(row["exit_price"]) == Decimal("110")
    assert float(row["return_pct"]) > 0


def test_feedback_agent_marks_loss():
    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    log.record(
        Decision(strategy="t", symbol="AAPL", action=DecisionAction.BUY,
                 regime=MarketRegime.UNKNOWN, reference_price=Decimal("100"))
    )
    bus = InMemoryBus()
    bus.publish("fills", {"symbol": "AAPL", "side": "sell",
                          "filled_qty": Decimal("10"), "fill_price": Decimal("90")})
    FeedbackAgent(log).step(bus)
    assert log.all_records()[0]["outcome_status"] == OutcomeStatus.LOSS.value
