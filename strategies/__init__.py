"""Estrategias de trading (modulos independentes e testaveis)."""

from strategies.base import Strategy, StrategyContext
from strategies.ladder_buys import LadderBuysStrategy
from strategies.trailing_stop import TrailingStopStrategy

__all__ = [
    "Strategy",
    "StrategyContext",
    "LadderBuysStrategy",
    "TrailingStopStrategy",
]
