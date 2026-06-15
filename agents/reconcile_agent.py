"""ReconcileAgent — reconciliacao periodica dentro do pipeline (doc 05 §7).

Roda como PRIMEIRO passo do pipeline, mas so dispara `reconcile()` a cada N
ciclos (broker = fonte de verdade). O boot ja reconcilia uma vez em main.py;
este agente garante que partial fills/cancelamentos que ocorrem DEPOIS do submit
sejam re-sincronizados durante o dia, sem precisar reiniciar o processo.

Sem inbox: e disparado pelo orquestrador via step().
"""

from __future__ import annotations

from agents.base import BaseAgent
from broker.base import BrokerClient
from data.audit_log import AuditLog
from data.order_repo import OrderRepository
from data.position_repo import PositionRepository
from orchestration.bus import MessageBus
from orchestration.reconcile import reconcile


class ReconcileAgent(BaseAgent):
    def __init__(
        self,
        broker: BrokerClient,
        order_repo: OrderRepository,
        position_repo: PositionRepository,
        audit: AuditLog,
        *,
        every_n_cycles: int = 6,
    ) -> None:
        super().__init__("reconcile")  # sem inbox
        self._broker = broker
        self._orders = order_repo
        self._positions = position_repo
        self._audit = audit
        self._every = max(1, every_n_cycles)
        self._count = 0

    def step(self, bus: MessageBus) -> None:
        self._count += 1
        if self._count % self._every != 0:
            return
        try:
            reconcile(self._broker, self._orders, self._positions, self._audit)
        except Exception:  # reconciliacao nunca deve derrubar o ciclo
            self.log.exception("Falha na reconciliacao periodica (ciclo %d)", self._count)
