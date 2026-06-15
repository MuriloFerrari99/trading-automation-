"""Agente de ingestao de dados de mercado (doc 05 §1, camada 1).

Busca barras recentes (fechamentos) dos ativos da watchlist no broker e publica
um snapshot normalizado no topico `market_data`. Nao toma decisoes. O
DecisionAgent consome esse snapshot para classificar o regime com barras reais
(em vez de amostrar 1 preco por ciclo).

Agente sem inbox: e disparado pelo scheduler/orquestrador via step().
"""

from __future__ import annotations

from decimal import Decimal

from agents.base import BaseAgent
from broker.base import BrokerClient
from orchestration.bus import MessageBus

MARKET_DATA = "market_data"


class IngestionAgent(BaseAgent):
    def __init__(
        self,
        broker: BrokerClient,
        symbols: list[str],
        *,
        bars_limit: int = 60,
    ) -> None:
        super().__init__("ingestion")  # sem inbox
        self._broker = broker
        self._symbols = [s.upper() for s in symbols]
        self._bars_limit = bars_limit

    def step(self, bus: MessageBus) -> None:
        snapshot: dict[str, list[Decimal]] = {}
        for symbol in self._symbols:
            try:
                snapshot[symbol] = self._broker.get_bars(symbol, self._bars_limit)
            except Exception:
                self.log.exception("Falha ao buscar barras de %s", symbol)
                snapshot[symbol] = []
        bus.publish(MARKET_DATA, snapshot)
        self.log.debug("Publicado market_data de %d ativo(s).", len(snapshot))
