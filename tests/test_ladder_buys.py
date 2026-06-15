"""Testes da estrategia Ladder Buys."""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.watchlist import LadderConfig, LadderRung, Watchlist, WatchlistItem
from core.models import OrderSide
from strategies.base import StrategyContext
from strategies.ladder_buys import LadderBuysStrategy, anchor_key, rung_key


@pytest.fixture
def ladder_watchlist() -> Watchlist:
    return Watchlist(
        items=[
            WatchlistItem(
                symbol="MSFT",
                ladder=LadderConfig(
                    anchor_price=Decimal("100"),
                    rungs=[
                        LadderRung(drop_pct=Decimal("0.20"), qty=Decimal("5")),
                        LadderRung(drop_pct=Decimal("0.30"), qty=Decimal("10")),
                    ],
                ),
            )
        ]
    )


def _ctx(broker, state, watchlist):
    return StrategyContext(broker=broker, state=state, watchlist=watchlist)


def test_no_trigger_above_first_rung(broker, state, ladder_watchlist):
    broker.set_price("MSFT", "85")  # -15%, acima do 1o degrau (-20% => 80)
    intents = LadderBuysStrategy().evaluate(_ctx(broker, state, ladder_watchlist))
    assert intents == []
    assert state.get_decimal(anchor_key("MSFT")) == Decimal("100")


def test_first_rung_triggers(broker, state, ladder_watchlist):
    broker.set_price("MSFT", "80")  # -20% exatamente => dispara 1o degrau
    intents = LadderBuysStrategy().evaluate(_ctx(broker, state, ladder_watchlist))
    assert len(intents) == 1
    assert intents[0].side == OrderSide.BUY
    assert intents[0].qty == Decimal("5")
    assert intents[0].strategy == "ladder_buys"
    assert state.get(rung_key("MSFT", 0)) == "filled"
    assert state.get(rung_key("MSFT", 1)) is None


def test_deep_drop_triggers_both_rungs(broker, state, ladder_watchlist):
    broker.set_price("MSFT", "65")  # -35% => dispara os dois degraus
    intents = LadderBuysStrategy().evaluate(_ctx(broker, state, ladder_watchlist))
    assert len(intents) == 2
    assert {i.qty for i in intents} == {Decimal("5"), Decimal("10")}


def test_rung_is_idempotent_across_cycles(broker, state, ladder_watchlist):
    strat = LadderBuysStrategy()
    broker.set_price("MSFT", "80")
    first = strat.evaluate(_ctx(broker, state, ladder_watchlist))
    assert len(first) == 1
    # mesmo preco no proximo ciclo: nao recompra o mesmo degrau
    second = strat.evaluate(_ctx(broker, state, ladder_watchlist))
    assert second == []


def test_rungs_fire_progressively(broker, state, ladder_watchlist):
    strat = LadderBuysStrategy()
    broker.set_price("MSFT", "80")  # 1o degrau
    assert len(strat.evaluate(_ctx(broker, state, ladder_watchlist))) == 1
    broker.set_price("MSFT", "70")  # -30% => 2o degrau
    second = strat.evaluate(_ctx(broker, state, ladder_watchlist))
    assert len(second) == 1
    assert second[0].qty == Decimal("10")


def test_anchor_defaults_to_first_observed_price(broker, state):
    wl = Watchlist(
        items=[
            WatchlistItem(
                symbol="MSFT",
                ladder=LadderConfig(
                    rungs=[LadderRung(drop_pct=Decimal("0.20"), qty=Decimal("5"))]
                ),
            )
        ]
    )
    broker.set_price("MSFT", "200")  # ancora inicial = 200
    strat = LadderBuysStrategy()
    assert strat.evaluate(_ctx(broker, state, wl)) == []
    assert state.get_decimal(anchor_key("MSFT")) == Decimal("200")
    broker.set_price("MSFT", "160")  # -20% de 200 => dispara
    assert len(strat.evaluate(_ctx(broker, state, wl))) == 1


def test_reset_rearms_ladder(broker, state, ladder_watchlist):
    strat = LadderBuysStrategy()
    broker.set_price("MSFT", "80")
    assert len(strat.evaluate(_ctx(broker, state, ladder_watchlist))) == 1
    LadderBuysStrategy.reset("MSFT", state)
    assert state.get_decimal(anchor_key("MSFT")) is None
    # re-arma: dispara de novo
    assert len(strat.evaluate(_ctx(broker, state, ladder_watchlist))) == 1
