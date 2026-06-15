"""Testes do FakeBroker e do fluxo de auditoria com TradeLogger."""

from __future__ import annotations

from decimal import Decimal

from broker.fake_broker import FakeBroker
from core.models import OrderIntent, OrderSide
from data.db import Database
from data.trade_logger import TradeLogger


def test_buy_creates_position_and_spends_cash():
    broker = FakeBroker(cash=Decimal("10000"), prices={"AAPL": Decimal("100")})
    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"))
    result = broker.submit_order(intent)

    assert result.status == "filled"
    assert result.filled_avg_price == Decimal("100")
    pos = broker.get_position("AAPL")
    assert pos is not None
    assert pos.qty == Decimal("10")
    assert broker.get_account().cash == Decimal("9000")


def test_sell_reduces_position():
    broker = FakeBroker(cash=Decimal("0"), prices={"AAPL": Decimal("100")})
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("90"))
    broker.submit_order(OrderIntent(symbol="AAPL", side=OrderSide.SELL, qty=Decimal("10")))
    assert broker.get_position("AAPL") is None
    assert broker.get_account().cash == Decimal("1000")


def test_market_open_flag():
    broker = FakeBroker(market_open=False)
    assert broker.is_market_open() is False
    broker.set_market_open(True)
    assert broker.is_market_open() is True


def test_trade_logged_to_sqlite():
    broker = FakeBroker(prices={"AAPL": Decimal("100")})
    db = Database(":memory:")
    logger = TradeLogger(db)

    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("5"), strategy="trailing_stop"
    )
    result = broker.submit_order(intent)
    row_id = logger.log_execution(intent, result)

    assert row_id > 0
    recent = logger.recent()
    assert len(recent) == 1
    entry = recent[0]
    assert entry["symbol"] == "AAPL"
    assert entry["side"] == "buy"
    assert entry["strategy"] == "trailing_stop"
    assert entry["qty"] == "5"
    db.close()
