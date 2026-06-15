"""Camada de orquestracao (desacoplada para plugar o OpenSquad depois)."""

from orchestration.base import AgentOrchestrator, CycleResult
from orchestration.factory import build_orchestrator
from orchestration.local_orchestrator import LocalOrchestrator
from orchestration.opensquad_orchestrator import (
    OpenSquadOrchestrator,
    OrchestratorBridge,
)

__all__ = [
    "AgentOrchestrator",
    "CycleResult",
    "LocalOrchestrator",
    "OpenSquadOrchestrator",
    "OrchestratorBridge",
    "build_orchestrator",
]
