"""Fabrica de orquestradores.

Permite trocar a implementacao de orquestracao por CONFIG (uma linha / variavel
de ambiente ORCHESTRATOR), sem tocar no main nem nos agentes — exatamente a
costura desacoplada que o briefing pediu para plugar o OpenSquad depois.

Default: "bus" — pipeline de agentes (ingestao -> planner -> decisao -> execucao
-> feedback) com reconciliacao periodica. E o unico modo que FECHA o loop de
feedback e classifica regime por barras reais. "local" e "opensquad" continuam
disponiveis para diagnostico/integracao futura.
"""

from __future__ import annotations

from agents.executor import Executor
from agents.planner import Planner
from intelligence.engine import DecisionIntelligence
from orchestration.base import AgentOrchestrator
from orchestration.bus_orchestrator import assemble_bus_orchestrator
from orchestration.local_orchestrator import LocalOrchestrator
from orchestration.opensquad_orchestrator import OpenSquadOrchestrator, OrchestratorBridge

BUS = "bus"
LOCAL = "local"
OPENSQUAD = "opensquad"


def build_orchestrator(
    name: str | None,
    planner: Planner,
    executor: Executor,
    *,
    bridge: OrchestratorBridge | None = None,
    intelligence: DecisionIntelligence | None = None,
    broker=None,
    decision_log=None,
    symbols: list[str] | None = None,
    order_repo=None,
    position_repo=None,
    audit=None,
    reconcile_every: int = 6,
    bars_limit: int = 60,
) -> AgentOrchestrator:
    key = (name or BUS).strip().lower()
    if key == BUS:
        missing = [
            n
            for n, v in (
                ("broker", broker),
                ("intelligence", intelligence),
                ("decision_log", decision_log),
                ("symbols", symbols),
            )
            if v is None
        ]
        if missing:
            raise ValueError(
                f"Orquestrador 'bus' requer {', '.join(missing)}. "
                f"Use 'local' se nao puder fornece-los."
            )
        return assemble_bus_orchestrator(
            broker=broker,
            planner=planner,
            executor=executor,
            intelligence=intelligence,
            decision_log=decision_log,
            symbols=symbols,
            bars_limit=bars_limit,
            order_repo=order_repo,
            position_repo=position_repo,
            audit=audit,
            reconcile_every=reconcile_every,
        )
    if key == LOCAL:
        return LocalOrchestrator(planner, executor, intelligence=intelligence)
    if key == OPENSQUAD:
        return OpenSquadOrchestrator(planner, executor, bridge=bridge)
    raise ValueError(
        f"Orquestrador desconhecido: {name!r}. Use {BUS!r}, {LOCAL!r} ou {OPENSQUAD!r}."
    )
