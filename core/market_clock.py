"""Relogio de mercado.

O Monitor so deve operar durante o pregao. A fonte de verdade e o clock da
corretora (broker.get_clock()), que ja considera feriados e early-close. Esta
classe expoe is_open + next_open/next_close para agendamento e visibilidade.
"""

from __future__ import annotations

from datetime import datetime

from broker.base import BrokerClient, MarketClockInfo


def asset_tradable_now(asset_class: str, equity_market_open: bool) -> bool:
    """Um ativo pode operar agora?

    Cripto opera 24/7 (sempre True); acoes respeitam o pregao (clock da corretora).
    """
    return True if asset_class == "crypto" else equity_market_open


class MarketClock:
    def __init__(self, broker: BrokerClient) -> None:
        self._broker = broker

    def info(self) -> MarketClockInfo:
        return self._broker.get_clock()

    def is_open(self) -> bool:
        return self._broker.is_market_open()

    def next_open(self) -> datetime | None:
        return self.info().next_open

    def next_close(self) -> datetime | None:
        return self.info().next_close
