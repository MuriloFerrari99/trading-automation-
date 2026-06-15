"""Testes do ciclo de vida da ordem (write-before-network) e auditoria."""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from core.models import OrderIntent, OrderSide
from data.audit_log import AuditLog
from data.order_repo import FILLED, PENDING_SUBMIT, SUBMITTED, OrderRepository


def test_pending_recorded_before_fill(db, broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    order_repo = OrderRepository(db)
    audit = AuditLog(db)
    ex = Executor(broker, trade_logger, kill_switch, order_repo=order_repo, audit=audit)

    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("5"), strategy="t")
    result = ex.execute(intent)
    assert result is not None

    row = order_repo.get(intent.client_order_id)
    assert row is not None
    assert row["status"] == FILLED  # FakeBroker preenche na hora
    assert row["broker_order_id"] is not None
    assert row["filled_qty"] == "5"


def test_audit_trail_records_actor(db, broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    order_repo = OrderRepository(db)
    audit = AuditLog(db)
    ex = Executor(broker, trade_logger, kill_switch, order_repo=order_repo, audit=audit)

    ex.execute(OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy="t"))
    events = audit.recent()
    actors = {e["actor"] for e in events}
    event_names = {e["event"] for e in events}
    assert "executor" in actors
    assert "order_pending" in event_names
    assert "order_submitted" in event_names


def test_failed_submit_marks_rejected(db, broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    order_repo = OrderRepository(db)
    audit = AuditLog(db)

    def always_fail(intent):
        raise RuntimeError("API down")

    broker.submit_order = always_fail  # type: ignore[method-assign]
    ex = Executor(
        broker, trade_logger, kill_switch,
        order_repo=order_repo, audit=audit, max_retries=2, backoff_seconds=0,
    )
    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy="t")
    result = ex.execute(intent)
    assert result is None
    row = order_repo.get(intent.client_order_id)
    assert row["status"] == "REJECTED"  # gravado como pendente e depois rejeitado


def test_trailing_stop_stays_open_not_filled(db, broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    order_repo = OrderRepository(db)
    audit = AuditLog(db)
    ex = Executor(broker, trade_logger, kill_switch, order_repo=order_repo, audit=audit)

    from core.models import OrderType

    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.SELL, qty=Decimal("10"),
        order_type=OrderType.TRAILING_STOP, trail_percent=Decimal("10"), strategy="trailing_stop",
    )
    result = ex.execute(intent)
    assert result is not None
    # trailing nativo fica ABERTO no broker (nao preenche imediatamente)
    assert order_repo.get(intent.client_order_id)["status"] == SUBMITTED
    assert len(broker.get_open_orders()) == 1
    _ = PENDING_SUBMIT  # constante exportada
