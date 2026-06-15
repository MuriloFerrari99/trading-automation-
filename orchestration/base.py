"""Interface de orquestracao de agentes.

A logica de negocio (Monitor/Planner/Executor) depende apenas desta interface.
Hoje usamos o LocalOrchestrator (single-process, chamadas diretas). Quando a
doc do OpenSquad chegar, basta implementar AgentOrchestrator por cima dele sem
reescrever agentes nem estrategias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.models import OrderIntent, OrderResult


@dataclass
class CycleResult:
    """Resultado de um ciclo de orquestracao."""

    intents: list[OrderIntent]
    results: list[OrderResult]

    @property
    def intents_count(self) -> int:
        return len(self.intents)

    @property
    def executed_count(self) -> int:
        return len(self.results)


class AgentOrchestrator(ABC):
    @abstractmethod
    def run_cycle(self) -> CycleResult:
        """Roda um ciclo completo: planejar -> executar -> retornar resultado."""
