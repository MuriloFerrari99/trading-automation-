"""Orquestrador local (single-process).

Implementacao simples da interface AgentOrchestrator: coordena Planner e
Executor por chamadas diretas, dentro do mesmo processo. Suficiente para o MVP
e respeita o contrato para troca futura pelo OpenSquad.

A "fila" de intencoes entre Planner e Executor e a propria lista retornada
pelo Planner; o ponto de acoplamento e unico (este orquestrador), o que
facilita inserir mensageria/eventos depois sem tocar nos agentes.
"""

from __future__ import annotations

import logging

from agents.executor import Executor
from agents.planner import Planner
from orchestration.base import AgentOrchestrator, CycleResult

logger = logging.getLogger("orchestrator.local")


class LocalOrchestrator(AgentOrchestrator):
    def __init__(self, planner: Planner, executor: Executor) -> None:
        self._planner = planner
        self._executor = executor

    def run_cycle(self) -> CycleResult:
        # 1) Sinais (Nivel 2): coletados e registrados como SUGESTAO. Nao
        #    passam pelo Executor — a separacao e proposital e estrutural.
        signals = self._planner.gather_signals()

        # 2) Estrategias (Nivel 1): geram intencoes a partir de dados de mercado.
        intents = self._planner.plan()
        if not intents:
            logger.debug("Nenhuma intencao gerada neste ciclo.")
            return CycleResult(intents=[], results=[], signals=signals)

        # 3) Execucao: somente intencoes de estrategia chegam ao Executor.
        results = self._executor.execute_many(intents)
        return CycleResult(intents=intents, results=results, signals=signals)
