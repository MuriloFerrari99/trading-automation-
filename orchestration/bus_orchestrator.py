"""BusOrchestrator — orquestracao no modelo bus + step (doc 05 §4).

Implementa a MESMA interface AgentOrchestrator (run_cycle -> CycleResult) que o
LocalOrchestrator, entao cai direto no Monitor existente sem mudar nada. A
diferenca e o paradigma interno: em vez de chamar planner/executor direto, ele
faz cada AGENTE dar um `step()` sobre o MessageBus, na ordem do pipeline.

E aditivo: o LocalOrchestrator (run_cycle direto) continua existindo. Para usar
este, monte via `assemble_bus_orchestrator(...)` e injete no Monitor.
"""

from __future__ import annotations

import logging

from agents.bus_agents import DecisionAgent, ExecutorAgent, PlannerAgent
from agents.feedback_agent import FeedbackAgent
from agents.ingestion import IngestionAgent
from orchestration.base import AgentOrchestrator, CycleResult
from orchestration.bus import InMemoryBus, MessageBus

logger = logging.getLogger("orchestrator.bus")


class BusOrchestrator(AgentOrchestrator):
    def __init__(
        self,
        bus: MessageBus,
        ingestion: IngestionAgent,
        planner_agent: PlannerAgent,
        decision_agent: DecisionAgent,
        executor_agent: ExecutorAgent,
        feedback_agent: FeedbackAgent,
        *,
        reconcile_agent=None,
    ) -> None:
        self._bus = bus
        self._planner_agent = planner_agent
        self._decision_agent = decision_agent
        self._executor_agent = executor_agent
        # Ordem do pipeline: [reconcile periodico] -> ingestao -> planner ->
        # decisao -> execucao -> feedback. O reconcile vem primeiro (broker =
        # verdade) e so dispara a cada N ciclos; e opcional (None em testes).
        self._pipeline = [
            ingestion, planner_agent, decision_agent, executor_agent, feedback_agent,
        ]
        if reconcile_agent is not None:
            self._pipeline.insert(0, reconcile_agent)

    def register(self, agent) -> None:
        """Adiciona um agente ao fim do pipeline (modelo do doc 05)."""
        self._pipeline.append(agent)

    def run_cycle(self) -> CycleResult:
        for agent in self._pipeline:
            agent.step(self._bus)
        result = CycleResult(
            intents=list(self._decision_agent.last_approved),
            results=list(self._executor_agent.last_results),
            signals=list(self._planner_agent.last_signals),
        )
        logger.debug(
            "Ciclo (bus): %d sinais, %d aprovadas, %d executadas.",
            result.signals_count, result.intents_count, result.executed_count,
        )
        return result


def assemble_bus_orchestrator(
    *,
    broker,
    planner,
    executor,
    intelligence,
    decision_log,
    symbols: list[str],
    bars_limit: int = 60,
    order_repo=None,
    position_repo=None,
    audit=None,
    reconcile_every: int = 6,
) -> BusOrchestrator:
    """Monta o pipeline de agentes sobre um InMemoryBus.

    Se order_repo/position_repo/audit forem fornecidos, inclui o ReconcileAgent
    periodico (broker = verdade a cada `reconcile_every` ciclos)."""
    bus = InMemoryBus()
    reconcile_agent = None
    if order_repo is not None and position_repo is not None and audit is not None:
        from agents.reconcile_agent import ReconcileAgent

        reconcile_agent = ReconcileAgent(
            broker, order_repo, position_repo, audit, every_n_cycles=reconcile_every
        )
    return BusOrchestrator(
        bus,
        IngestionAgent(broker, symbols, bars_limit=bars_limit),
        PlannerAgent(planner),
        DecisionAgent(intelligence),
        ExecutorAgent(executor),
        FeedbackAgent(decision_log),
        reconcile_agent=reconcile_agent,
    )
