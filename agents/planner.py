"""Agente Planejador.

Consome dados de mercado (via estrategias) e produz intencoes de ordem.
No MVP roda a(s) estrategia(s) configurada(s) e agrega as OrderIntent.
Nao executa nada — apenas decide.
"""

from __future__ import annotations

import logging

from broker.base import BrokerClient
from config.watchlist import Watchlist
from core.models import OrderIntent
from data.state_repo import StateRepository
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("agent.planner")


class Planner:
    def __init__(
        self,
        broker: BrokerClient,
        state: StateRepository,
        watchlist: Watchlist,
        strategies: list[Strategy],
    ) -> None:
        self._broker = broker
        self._state = state
        self._watchlist = watchlist
        self._strategies = strategies

    def plan(self) -> list[OrderIntent]:
        ctx = StrategyContext(
            broker=self._broker, state=self._state, watchlist=self._watchlist
        )
        intents: list[OrderIntent] = []
        for strategy in self._strategies:
            try:
                produced = strategy.evaluate(ctx)
            except Exception:  # uma estrategia com erro nao derruba as demais
                logger.exception("Estrategia '%s' falhou ao avaliar", strategy.name)
                continue
            if produced:
                logger.info("Estrategia '%s' gerou %d intencao(oes)", strategy.name, len(produced))
            intents.extend(produced)
        return intents
