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


def test_feedback_uses_real_buy_fill_price_as_entry():
    """P&L medido do fill REAL da compra (102), nao do reference_price (100)."""
    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    dec_id = log.record(
        Decision(strategy="t", symbol="AAPL", action=DecisionAction.BUY,
                 regime=MarketRegime.UNKNOWN, reference_price=Decimal("100"),
                 client_order_id="coid-1")
    )
    bus = InMemoryBus()
    agent = FeedbackAgent(log)

    # Fill de COMPRA a 102 (diferente do reference 100) => fixa entry real.
    bus.publish("fills", {"symbol": "AAPL", "side": "buy", "filled_qty": Decimal("10"),
                          "fill_price": Decimal("102"), "client_order_id": "coid-1"})
    agent.step(bus)
    row = next(r for r in log.all_records() if r["id"] == dec_id)
    assert Decimal(row["entry_price"]) == Decimal("102")
    assert row["outcome_status"] == OutcomeStatus.OPEN.value  # ainda aberta

    # Venda a 112 => fecha medindo de 102 (e nao de 100).
    bus.publish("fills", {"symbol": "AAPL", "side": "sell",
                          "filled_qty": Decimal("10"), "fill_price": Decimal("112")})
    agent.step(bus)
    row = next(r for r in log.all_records() if r["id"] == dec_id)
    assert row["outcome_status"] == OutcomeStatus.WIN.value
    assert abs(float(row["return_pct"]) - (10.0 / 102.0)) < 1e-9


def test_decision_agent_uses_market_data_bars_for_regime():
    """Regime sai de UNKNOWN ja no 1o ciclo quando ha barras reais no bus."""
    from agents.bus_agents import DecisionAgent
    from core.models import OrderIntent, OrderSide

    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    # SEM price_provider: so as barras do bus podem tirar o regime de UNKNOWN.
    intel = DecisionIntelligence(log, DecisionPolicy())
    agent = DecisionAgent(intel)

    # Serie em tendencia de alta clara (> slow+1 amostras).
    closes = [Decimal(str(100 + i)) for i in range(40)]
    bus = InMemoryBus()
    bus.publish("market_data", {"AAPL": closes})
    bus.publish("intents", [OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy="t")])
    bus.publish("signals", [])
    agent.step(bus)

    row = log.all_records()[-1]
    assert row["regime"] == MarketRegime.TREND_UP.value


def test_reconcile_agent_runs_every_n_cycles():
    from agents.reconcile_agent import ReconcileAgent
    from data.audit_log import AuditLog
    from data.order_repo import OrderRepository
    from data.position_repo import PositionRepository

    db = Database(":memory:")
    broker = FakeBroker(cash=Decimal("100000"), prices={"AAPL": Decimal("100")})
    agent = ReconcileAgent(
        broker, OrderRepository(db), PositionRepository(db), AuditLog(db), every_n_cycles=3
    )
    bus = InMemoryBus()

    calls = []
    import agents.reconcile_agent as ra
    orig = ra.reconcile
    ra.reconcile = lambda *a, **k: calls.append(1)
    try:
        for _ in range(7):
            agent.step(bus)
    finally:
        ra.reconcile = orig
    # ciclos 3 e 6 disparam (a cada 3); boot ja reconciliou antes.
    assert len(calls) == 2
