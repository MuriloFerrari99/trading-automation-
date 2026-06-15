"""Testes de idempotencia: client_order_id determinístico e nao-duplicacao."""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from core.idempotency import make_client_order_id
from core.models import OrderIntent, OrderSide


def test_client_order_id_is_deterministic():
    a = make_client_order_id("trailing_stop", "AAPL", "sell", "2026-06-15T10:00:00+00:00")
    b = make_client_order_id("trailing_stop", "AAPL", "sell", "2026-06-15T10:00:00+00:00")
    c = make_client_order_id("trailing_stop", "AAPL", "sell", "2026-06-15T10:00:01+00:00")
    assert a == b  # mesmo evento => mesmo id
    assert a != c  # decisao em instante diferente => id diferente
    assert a.startswith("bot-")


def test_order_intent_autofills_client_order_id():
    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy="t")
    assert intent.client_order_id is not None
    # reconstruir com o MESMO created_at reproduz o id
    again = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"),
        strategy="t", created_at=intent.created_at,
    )
    assert again.client_order_id == intent.client_order_id


def test_resubmit_same_intent_is_not_duplicated(broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    ex = Executor(broker, trade_logger, kill_switch)
    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("5"), strategy="t")

    first = ex.execute(intent)
    assert first is not None
    # reenvio da MESMA intencao (mesmo client_order_id) => idempotente, nao reexecuta
    second = ex.execute(intent)
    assert second is None
    assert len(broker.submitted) == 1  # uma unica ordem chegou ao broker
