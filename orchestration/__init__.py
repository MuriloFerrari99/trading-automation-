"""Camada de orquestracao (desacoplada para plugar o OpenSquad depois)."""

from orchestration.base import AgentOrchestrator, CycleResult
from orchestration.local_orchestrator import LocalOrchestrator

__all__ = ["AgentOrchestrator", "CycleResult", "LocalOrchestrator"]
