"""Integracao Planner + camada de risco."""

from __future__ import annotations

from decimal import Decimal

from agents.planner import Planner
from config.risk import RiskSettings
from config.watchlist import Watchlist, WatchlistItem
from core.models import OrderIntent, OrderSide
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard
from strategies.base import Strategy


class _BuyStrategy(Strategy):
    name = "buyer"

    def __init__(self, qty):
        self._qty = qty

    def evaluate(self, ctx):
        return [OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=self._qty, strategy=self.name)]


def _settings():
    return RiskSettings(
        _env_file=None, RISK_PER_TRADE_PCT=Decimal("0.01"),
        MAX_PER_SYMBOL_PCT=Decimal("0.20"), MAX_PORTFOLIO_HEAT_PCT=Decimal("0.10"),
        DAILY_LOSS_LIMIT_PCT=Decimal("0.03"), MAX_DRAWDOWN_PCT=Decimal("0.20"),
    )


def _planner(broker, state, strategy, guard):
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    rm = RiskManager(_settings(), guard)
    return Planner(broker, state, wl, [strategy], risk_manager=rm)


def test_planner_caps_buy_qty(broker, state):
    broker.set_price("AAPL", "100")  # equity = cash 100k => max 20% => 200 shares
    planner = _planner(broker, state, _BuyStrategy(Decimal("500")), PortfolioRiskGuard(Decimal("100000")))
    intents = planner.plan()
    assert len(intents) == 1
    assert intents[0].qty == Decimal("200")  # capado pelo teto por simbolo


def test_planner_vetoes_buy_when_halted(broker, state):
    broker.set_price("AAPL", "100")
    guard = PortfolioRiskGuard(Decimal("100000"), daily_loss_pct=Decimal("0.03"))
    guard.update(Decimal("90000"))  # -10% => halt
    planner = _planner(broker, state, _BuyStrategy(Decimal("10")), guard)
    assert planner.plan() == []  # risco crescente vetado sob halt
