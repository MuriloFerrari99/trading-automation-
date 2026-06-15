"""Testes da Wheel Strategy: gate de elegibilidade, puts e covered calls."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agents.executor import Executor
from broker.fake_broker import FakeBroker
from config.watchlist import Watchlist, WatchlistItem, WheelConfig
from core.models import OptionType, OrderSide
from strategies.base import StrategyContext
from strategies.wheel import WheelStrategy, open_call_key, open_put_key


@pytest.fixture
def wheel_watchlist() -> Watchlist:
    return Watchlist(
        items=[
            WatchlistItem(
                symbol="KO",
                wheel=WheelConfig(otm_pct=Decimal("0.10"), contracts=1, min_dte=20, max_dte=45),
            )
        ]
    )


def _ctx(broker, state, watchlist):
    return StrategyContext(broker=broker, state=state, watchlist=watchlist)


# --- GATE -------------------------------------------------------------------
def test_gate_blocks_without_options_level(state, wheel_watchlist):
    broker = FakeBroker(cash=Decimal("100000"), prices={"KO": Decimal("60")}, options_level=0)
    intents = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))
    assert intents == []  # nivel 0 => Wheel desabilitada


def test_gate_blocks_without_liquidity(state, wheel_watchlist):
    # nivel ok, mas cash insuficiente p/ garantir o put (~54*100 = 5400)
    broker = FakeBroker(cash=Decimal("1000"), prices={"KO": Decimal("60")}, options_level=2)
    intents = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))
    assert intents == []


# --- PUT --------------------------------------------------------------------
def test_sells_cash_secured_put_when_eligible(state, wheel_watchlist):
    broker = FakeBroker(cash=Decimal("100000"), prices={"KO": Decimal("60")}, options_level=2)
    intents = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side == OrderSide.SELL
    assert intent.contract.option_type == OptionType.PUT
    # strike ~10% abaixo de 60 = 54
    assert intent.contract.strike == Decimal("54")
    assert state.get(open_put_key("KO")) is not None


def test_put_not_resold_next_cycle(state, wheel_watchlist):
    broker = FakeBroker(cash=Decimal("100000"), prices={"KO": Decimal("60")}, options_level=2)
    strat = WheelStrategy()
    assert len(strat.evaluate(_ctx(broker, state, wheel_watchlist))) == 1
    assert strat.evaluate(_ctx(broker, state, wheel_watchlist)) == []  # idempotente


# --- COVERED CALL -----------------------------------------------------------
def test_sells_covered_call_when_holding_shares(state, wheel_watchlist):
    broker = FakeBroker(cash=Decimal("0"), prices={"KO": Decimal("60")}, options_level=2)
    # assignment: passou a deter 100 acoes a custo 54
    broker.seed_position("KO", qty=Decimal("100"), avg_entry_price=Decimal("54"))
    intents = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))
    assert len(intents) == 1
    intent = intents[0]
    assert intent.contract.option_type == OptionType.CALL
    # strike ~10% acima do custo 54 = 59.4 -> arredonda p/ 59
    assert intent.contract.strike == Decimal("59")
    assert state.get(open_call_key("KO")) is not None


# --- EXECUTOR (defesa em profundidade) --------------------------------------
def test_executor_blocks_option_without_level(state, trade_logger, kill_switch, wheel_watchlist):
    # gera a intencao com nivel ok...
    broker = FakeBroker(cash=Decimal("100000"), prices={"KO": Decimal("60")}, options_level=2)
    intent = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))[0]
    # ...mas no momento da execucao a conta perdeu o nivel
    broker.set_options_level(0)
    ex = Executor(broker, trade_logger, kill_switch)
    result = ex.execute(intent)
    assert result is None
    assert broker.submitted_options == []


def test_executor_executes_and_audits_option(state, trade_logger, kill_switch, wheel_watchlist):
    broker = FakeBroker(cash=Decimal("100000"), prices={"KO": Decimal("60")}, options_level=2)
    intent = WheelStrategy().evaluate(_ctx(broker, state, wheel_watchlist))[0]
    ex = Executor(broker, trade_logger, kill_switch)
    result = ex.execute(intent)
    assert result is not None
    assert result.status == "filled"
    assert len(broker.submitted_options) == 1
    logged = trade_logger.recent()
    assert len(logged) == 1
    assert logged[0]["strategy"] == "wheel"
