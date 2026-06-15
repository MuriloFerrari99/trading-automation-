"""Testes da estrategia Trailing Stop (nativo Alpaca)."""

from __future__ import annotations

from decimal import Decimal

from broker.base import BrokerOrder
from core.models import OrderSide, OrderType
from strategies.base import StrategyContext
from strategies.trailing_stop import TrailingStopStrategy


def _ctx(broker, state, watchlist):
    return StrategyContext(broker=broker, state=state, watchlist=watchlist)


def test_no_position_no_intents(broker, state, watchlist):
    broker.set_price("AAPL", "100")
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []


def test_places_native_trailing_stop_when_holding(broker, state, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side == OrderSide.SELL
    assert intent.order_type == OrderType.TRAILING_STOP
    assert intent.qty == Decimal("10")
    assert intent.trail_percent == Decimal("10")  # 0.10 -> 10%


def test_does_not_duplicate_when_trailing_already_open(broker, state, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    # ja existe uma trailing stop aberta no broker p/ AAPL
    broker.seed_open_order(
        BrokerOrder(
            broker_order_id="t1", client_order_id="x", symbol="AAPL", side="sell",
            qty=Decimal("10"), filled_qty=Decimal("0"), status="new",
            order_type="trailing_stop",
        )
    )
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []  # ja protegido, nao duplica


def test_full_protection_flow_places_once(broker, state, watchlist):
    """Em ciclos seguidos, coloca a trailing UMA vez e nao reenvia."""
    from agents.executor import Executor
    from core.kill_switch import KillSwitch
    from data.db import Database
    from data.trade_logger import TradeLogger

    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    ks = KillSwitch("/tmp/__no_kill__")
    ex = Executor(broker, TradeLogger(Database(":memory:")), ks)
    strat = TrailingStopStrategy()

    # ciclo 1: coloca a trailing
    intents = strat.evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    ex.execute(intents[0])
    # ciclo 2: ja existe trailing aberta no broker => nao reenvia
    assert strat.evaluate(_ctx(broker, state, watchlist)) == []
