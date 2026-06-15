"""Fabrica de orquestradores.

Permite trocar a implementacao de orquestracao por CONFIG (uma linha / variavel
de ambiente ORCHESTRATOR), sem tocar no main nem nos agentes — exatamente a
costura desacoplada que o briefing pediu para plugar o OpenSquad depois.
"""

from __future__ import annotations

from agents.executor import Executor
from agents.planner import Planner
from intelligence.engine import DecisionIntelligence
from orchestration.base import AgentOrchestrator
from orchestration.local_orchestrator import LocalOrchestrator
from orchestration.opensquad_orchestrator import OpenSquadOrchestrator, OrchestratorBridge

LOCAL = "local"
OPENSQUAD = "opensquad"


def build_orchestrator(
    name: str | None,
    planner: Planner,
    executor: Executor,
    *,
    bridge: OrchestratorBridge | None = None,
    intelligence: DecisionIntelligence | None = None,
) -> AgentOrchestrator:
    key = (name or LOCAL).strip().lower()
    if key == LOCAL:
        return LocalOrchestrator(planner, executor, intelligence=intelligence)
    if key == OPENSQUAD:
        return OpenSquadOrchestrator(planner, executor, bridge=bridge)
    raise ValueError(
        f"Orquestrador desconhecido: {name!r}. Use {LOCAL!r} ou {OPENSQUAD!r}."
    )
