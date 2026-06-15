"""Testes de reconciliacao broker = fonte de verdade."""

from __future__ import annotations

from decimal import Decimal

from broker.base import BrokerOrder
from broker.fake_broker import FakeBroker
from data.audit_log import AuditLog
from data.order_repo import FILLED, OrderRepository
from data.position_repo import PositionRepository
from orchestration.reconcile import reconcile


def _repos(db):
    return OrderRepository(db), PositionRepository(db), AuditLog(db)


def test_reconcile_syncs_broker_positions(db):
    broker = FakeBroker()
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    order_repo, pos_repo, audit = _repos(db)

    report = reconcile(broker, order_repo, pos_repo, audit)
    assert report.positions_synced == 1
    assert "AAPL" in pos_repo.symbols()
    # broker tinha posicao que o local nao conhecia => ORPHAN_POSITION
    assert ("ORPHAN_POSITION", "AAPL") in report.issues


def test_reconcile_removes_stale_local_position(db):
    broker = FakeBroker()  # broker sem posicoes
    order_repo, pos_repo, audit = _repos(db)
    # local acha que tem TSLA, mas o broker nao
    from core.models import Position

    pos_repo.upsert(Position(symbol="TSLA", qty=Decimal("5"), avg_entry_price=Decimal("200")))

    report = reconcile(broker, order_repo, pos_repo, audit)
    assert ("STALE_LOCAL_POSITION", "TSLA") in report.issues
    assert "TSLA" not in pos_repo.symbols()  # corrigido a favor do broker


def test_reconcile_updates_filled_order(db):
    broker = FakeBroker()
    order_repo, pos_repo, audit = _repos(db)
    # ordem local pendente; o broker reporta como filled
    from core.models import OrderIntent, OrderSide

    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"), strategy="t")
    order_repo.record_pending(intent)
    broker.seed_open_order(
        BrokerOrder(
            broker_order_id="b1", client_order_id=intent.client_order_id,
            symbol="AAPL", side="buy", qty=Decimal("10"), filled_qty=Decimal("10"),
            status="filled", order_type="market",
        )
    )

    report = reconcile(broker, order_repo, pos_repo, audit)
    assert report.orders_updated == 1
    assert order_repo.get(intent.client_order_id)["status"] == FILLED


def test_reconcile_marks_unknown_local_order_rejected(db):
    broker = FakeBroker()  # nao conhece a ordem
    order_repo, pos_repo, audit = _repos(db)
    from core.models import OrderIntent, OrderSide

    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy="t")
    order_repo.record_pending(intent)

    report = reconcile(broker, order_repo, pos_repo, audit)
    assert ("UNKNOWN_LOCAL_ORDER", intent.client_order_id) in report.issues
    assert order_repo.get(intent.client_order_id)["status"] == "REJECTED"
    # auditoria registrada
    assert any(e["event"] == "reconcile_on_boot" for e in audit.recent())
