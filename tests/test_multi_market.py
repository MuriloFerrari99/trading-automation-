"""Generalizacao multi-mercado: cripto 24/7, gating por pregao, tick/lote."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agents.monitor import Monitor
from agents.planner import Planner
from config.watchlist import Watchlist, WatchlistItem
from core.market_clock import MarketClock, asset_tradable_now
from core.models import OrderIntent, OrderSide
from core.rounding import round_price, round_qty
from orchestration.base import CycleResult
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard
from strategies.base import Strategy


# --- helpers de tradabilidade ----------------------------------------------
def test_asset_tradable_now():
    assert asset_tradable_now("crypto", equity_market_open=False) is True
    assert asset_tradable_now("crypto", equity_market_open=True) is True
    assert asset_tradable_now("equity", equity_market_open=False) is False
    assert asset_tradable_now("equity", equity_market_open=True) is True


# --- watchlist --------------------------------------------------------------
def test_watchlist_parses_asset_class_and_steps(tmp_path):
    from config.watchlist import load_watchlist

    yaml_text = """
symbols:
  - symbol: BTCUSD
    asset_class: crypto
    fractional: true
    tick_size: 0.01
    lot_size: 0.0001
    trailing_stop_pct: 12.0
  - symbol: AAPL
"""
    p = tmp_path / "wl.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    wl = load_watchlist(p)
    btc = wl.get("BTCUSD")
    assert btc.is_crypto and btc.fractional
    assert btc.tick_size == Decimal("0.01") and btc.lot_size == Decimal("0.0001")
    assert wl.get("AAPL").asset_class == "equity"
    assert wl.has_crypto() is True


def test_invalid_asset_class_rejected():
    with pytest.raises(ValueError):
        WatchlistItem(symbol="X", asset_class="forex")


# --- rounding ---------------------------------------------------------------
def test_round_qty_to_lot_and_fractional():
    assert round_qty(Decimal("0.123456"), Decimal("0.0001"), fractional=True) == Decimal("0.1234")
    assert round_qty(Decimal("7.9"), None) == Decimal("7")  # acoes inteiras
    assert round_qty(Decimal("0.5"), None, fractional=True) == Decimal("0.5")


def test_round_price_to_tick():
    assert round_price(Decimal("100.027"), Decimal("0.01")) == Decimal("100.03")
    assert round_price(None, Decimal("0.01")) is None
    assert round_price(Decimal("100.02"), None) == Decimal("100.02")


# --- Monitor 24/7 -----------------------------------------------------------
class _StubClock(MarketClock):
    def __init__(self, is_open):
        self._is_open = is_open

    def info(self):
        from broker.base import MarketClockInfo

        return MarketClockInfo(is_open=self._is_open)


class _StubOrch:
    def __init__(self):
        self.ran = 0

    def run_cycle(self):
        self.ran += 1
        return CycleResult(intents=[], results=[], signals=[])


def test_monitor_skips_when_closed_equity_only():
    orch = _StubOrch()
    m = Monitor(orch, _StubClock(is_open=False), run_when_closed=False)
    assert m.tick() is None
    assert orch.ran == 0


def test_monitor_runs_when_closed_if_crypto():
    orch = _StubOrch()
    m = Monitor(orch, _StubClock(is_open=False), run_when_closed=True)
    assert m.tick() is not None
    assert orch.ran == 1


# --- gating de estrategia por pregao ---------------------------------------
class _BuyEverything(Strategy):
    name = "buyer"

    def evaluate(self, ctx):
        out = []
        for item in ctx.watchlist.items:
            from core.market_clock import asset_tradable_now as _t

            if _t(item.asset_class, ctx.equity_market_open):
                out.append(OrderIntent(symbol=item.symbol, side=OrderSide.BUY,
                                       qty=Decimal("1"), strategy=self.name))
        return out


def test_crypto_trades_when_equity_closed(broker, state):
    broker.set_market_open(False)  # pregao de acoes fechado
    broker.set_price("BTCUSD", "50000")
    broker.set_price("AAPL", "100")
    wl = Watchlist(items=[
        WatchlistItem(symbol="BTCUSD", asset_class="crypto", fractional=True),
        WatchlistItem(symbol="AAPL"),
    ])
    rm = RiskManager(
        __import__("config.risk", fromlist=["RiskSettings"]).RiskSettings(_env_file=None),
        PortfolioRiskGuard(Decimal("100000")),
    )
    planner = Planner(broker, state, wl, [_BuyEverything()], risk_manager=rm)
    intents = planner.plan()
    symbols = [i.symbol for i in intents]
    assert "BTCUSD" in symbols   # cripto opera 24/7
    assert "AAPL" not in symbols  # acao barrada fora do pregao
