"""Estrategias de trading (modulos independentes e testaveis)."""

from strategies.base import Strategy, StrategyContext
from strategies.trailing_stop import TrailingStopStrategy

__all__ = ["Strategy", "StrategyContext", "TrailingStopStrategy"]
