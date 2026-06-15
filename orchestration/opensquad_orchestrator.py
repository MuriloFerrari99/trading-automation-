"""Adapter de orquestracao externa (OpenSquad) — costura de plug.

CONTEXTO HONESTO (ver docs/research/05-arquitetura-agentes.md, secao 5):
o OpenSquad e um framework TS/Node usado via CLI / slash-commands / MCP, e
NAO expoe um SDK Python publico para registrar/despachar agentes
programaticamente. Portanto NAO da para "embutir" o OpenSquad sem a doc/SDK
ou o contrato MCP dele.

O que entregamos aqui sem inventar a API do OpenSquad:
- `OrchestratorBridge`: o contrato MINIMO que um orquestrador externo precisa
  satisfazer para dirigir o ciclo de trading. Quando a doc do OpenSquad chegar,
  implementa-se um bridge concreto por cima do MCP/CLI dele — sem tocar em
  agentes, estrategias ou broker.
- `OpenSquadOrchestrator`: implementa a interface estavel `AgentOrchestrator`.
  Sem um bridge concreto, falha com mensagem clara (em vez de fingir integrar).
  Com um bridge, delega o ciclo e respeita o CHECKPOINT DE APROVACAO — conceito
  central do OpenSquad (pipeline com aprovacao humana).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agents.executor import Executor
from agents.planner import Planner
from orchestration.base import AgentOrchestrator, CycleResult
from strategies.base import TradeIntent

logger = logging.getLogger("orchestrator.opensquad")


class OrchestratorBridge(ABC):
    """Contrato que um orquestrador externo (OpenSquad via MCP/CLI) deve cumprir.

    Esta NAO e a API do OpenSquad — e a abstracao que ESTE adapter exige. Os
    dois metodos mapeiam o modelo do OpenSquad (pipeline com checkpoints):
    decidir QUANDO rodar e APROVAR o que sera executado.
    """

    @abstractmethod
    def should_run_cycle(self) -> bool:
        """Decide se o ciclo deve rodar agora (ex: squad disparou o step)."""

    @abstractmethod
    def approve(self, intents: list[TradeIntent]) -> list[TradeIntent]:
        """Recebe as intencoes planejadas e retorna as APROVADAS para execucao.

        Permite o checkpoint de aprovacao (humano ou por politica) antes de
        qualquer ordem chegar ao Executor.
        """


class OpenSquadOrchestrator(AgentOrchestrator):
    """Orquestrador externo plugavel. Mesma interface do LocalOrchestrator."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        *,
        bridge: OrchestratorBridge | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._bridge = bridge

    def run_cycle(self) -> CycleResult:
        if self._bridge is None:
            raise NotImplementedError(
                "OpenSquadOrchestrator requer um OrchestratorBridge concreto. "
                "O OpenSquad e TS/Node (CLI/MCP) e nao expoe SDK Python publico "
                "para despachar agentes; veja docs/research/05-arquitetura-agentes.md. "
                "Forneca a doc do SDK / contrato MCP do OpenSquad para implementar "
                "o bridge. Ate la, use o orquestrador 'local'."
            )

        # Sinais (Nivel 2): coletados/sugeridos como no fluxo local.
        signals = self._planner.gather_signals()

        if not self._bridge.should_run_cycle():
            logger.info("Bridge externo decidiu nao rodar o ciclo agora.")
            return CycleResult(intents=[], results=[], signals=signals)

        intents = self._planner.plan()
        # Checkpoint de aprovacao: so o que o bridge aprovar chega ao Executor.
        approved = self._bridge.approve(intents)
        if len(approved) != len(intents):
            logger.info(
                "Checkpoint: %d de %d intencao(oes) aprovada(s) para execucao.",
                len(approved), len(intents),
            )
        results = self._executor.execute_many(approved)
        return CycleResult(intents=intents, results=results, signals=signals)
