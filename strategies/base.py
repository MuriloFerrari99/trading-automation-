"""Contrato de estrategia.

Cada estrategia e um modulo independente e testavel que, dado um contexto de
mercado, produz uma lista de `OrderIntent`. A estrategia NUNCA submete ordens
diretamente — quem executa e o Executor, depois de validacao e kill switch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from broker.base import BrokerClient
from config.watchlist import Watchlist
from core.models import OptionOrderIntent, OrderIntent
from data.state_repo import StateRepository

# Uma estrategia pode emitir intencoes de acoes (OrderIntent) e/ou de opcoes
# (OptionOrderIntent). O Executor despacha por tipo.
TradeIntent = OrderIntent | OptionOrderIntent


@dataclass
class StrategyContext:
    """Tudo que uma estrategia precisa para decidir, sem acoplar a corretora."""

    broker: BrokerClient
    state: StateRepository
    watchlist: Watchlist


class Strategy(ABC):
    #: nome curto, usado em auditoria (campo `strategy` da OrderIntent)
    name: str = "base"

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> list[TradeIntent]:
        """Avalia o mercado e retorna intencoes de ordem (possivelmente vazio)."""
