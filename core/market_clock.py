"""Relogio de mercado.

O Monitor so deve operar durante o pregao. A fonte de verdade e o clock da
corretora (broker.is_market_open()), exposto aqui atras de uma interface fina
e testavel.
"""

from __future__ import annotations

from broker.base import BrokerClient


class MarketClock:
    def __init__(self, broker: BrokerClient) -> None:
        self._broker = broker

    def is_open(self) -> bool:
        return self._broker.is_market_open()
