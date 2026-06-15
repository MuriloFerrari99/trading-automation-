"""Testes da estrategia Trailing Stop."""

from __future__ import annotations

from decimal import Decimal

from core.models import OrderSide
from strategies.base import StrategyContext
from strategies.trailing_stop import TrailingStopStrategy, high_water_key


def _ctx(broker, state, watchlist):
    return StrategyContext(broker=broker, state=state, watchlist=watchlist)


def test_no_position_no_intents(broker, state, watchlist):
    broker.set_price("AAPL", "100")
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []


def test_no_trigger_above_stop(broker, state, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")  # subiu; stop = 105*0.9 = 94.5
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []
    # high-water deve ter sido registrado em 105
    assert state.get_decimal(high_water_key("AAPL")) == Decimal("105")


def test_stop_follows_price_up_then_triggers(broker, state, watchlist):
    strat = TrailingStopStrategy()
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))

    # Sobe a 120 -> high-water 120, stop 108. Sem disparo.
    broker.set_price("AAPL", "120")
    assert strat.evaluate(_ctx(broker, state, watchlist)) == []
    assert state.get_decimal(high_water_key("AAPL")) == Decimal("120")

    # Cai para 110 (> stop 108). Ainda sem disparo; stop NAO desce.
    broker.set_price("AAPL", "110")
    assert strat.evaluate(_ctx(broker, state, watchlist)) == []
    assert state.get_decimal(high_water_key("AAPL")) == Decimal("120")

    # Cai para 107 (< stop 108). Dispara venda da posicao inteira.
    broker.set_price("AAPL", "107")
    intents = strat.evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    assert intents[0].side == OrderSide.SELL
    assert intents[0].qty == Decimal("10")
    assert intents[0].strategy == "trailing_stop"
    # apos disparo o high-water e resetado
    assert state.get_decimal(high_water_key("AAPL")) is None


def test_high_water_initialized_from_entry(broker, state, watchlist):
    # preco atual abaixo da entrada => stop baseado na entrada
    broker.seed_position("AAPL", qty=Decimal("5"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "89")  # stop = 100*0.9 = 90; 89 < 90 dispara
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    assert intents[0].side == OrderSide.SELL
