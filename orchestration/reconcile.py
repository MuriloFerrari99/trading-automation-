"""Reconciliacao broker = fonte de verdade.

Ao iniciar (e periodicamente), ANTES de qualquer decisao nova, o bot reconcilia
seu estado local contra o que a Alpaca reporta (doc 05 §7, doc 02 §7.5):

1. Busca posicoes reais e ordens abertas no broker.
2. Compara com o estado local (positions, orders pendentes/submetidas).
3. Resolve divergencias a favor do BROKER: snapshot de posicoes corrigido;
   ordens locais pendentes que o broker ja preencheu/cancelou sao atualizadas;
   ordens locais que o broker nao conhece sao marcadas como rejeitadas.
4. Loga toda divergencia no audit_log.

Divergencia recorrente e sintoma de bug de idempotencia ou partial fill mal
tratado — por isso tudo vai para auditoria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from broker.base import BrokerClient
from core.models import Position
from data.audit_log import SYSTEM, AuditLog
from data.order_repo import CANCELED, FILLED, PARTIALLY_FILLED, OrderRepository
from data.position_repo import PositionRepository

logger = logging.getLogger("reconcile")

_BROKER_FILLED = {"filled"}
_BROKER_PARTIAL = {"partially_filled"}
_BROKER_DEAD = {"canceled", "expired", "rejected", "done_for_day"}


@dataclass
class ReconcileReport:
    positions_synced: int = 0
    orders_updated: int = 0
    issues: list[tuple[str, str]] = field(default_factory=list)  # (tipo, detalhe)

    def add_issue(self, kind: str, detail: str) -> None:
        self.issues.append((kind, detail))


def reconcile(
    broker: BrokerClient,
    order_repo: OrderRepository,
    position_repo: PositionRepository,
    audit: AuditLog,
) -> ReconcileReport:
    report = ReconcileReport()

    # 1) Posicoes: broker vence.
    broker_positions: list[Position] = broker.get_positions()
    local_symbols = position_repo.symbols()
    broker_symbols = {p.symbol for p in broker_positions}

    for sym in broker_symbols - local_symbols:
        report.add_issue("ORPHAN_POSITION", sym)  # broker tem, local nao conhecia
    for sym in local_symbols - broker_symbols:
        report.add_issue("STALE_LOCAL_POSITION", sym)  # local tinha, broker nao

    position_repo.replace_all_from_broker(broker_positions)
    report.positions_synced = len(broker_positions)

    # 2) Ordens locais que o bot acha em aberto: confronta com o broker.
    for local in order_repo.pending_or_submitted():
        cid = local["client_order_id"]
        broker_order = _safe_lookup(broker, cid)
        if broker_order is None:
            # Broker nao conhece a ordem que o bot achava pendente.
            order_repo.mark_rejected(cid)
            report.orders_updated += 1
            report.add_issue("UNKNOWN_LOCAL_ORDER", cid)
            continue
        status = (broker_order.status or "").lower()
        if status in _BROKER_FILLED:
            order_repo.mark_fill(
                cid, filled_qty=broker_order.filled_qty,
                filled_avg_price=None, status=FILLED,
            )
            report.orders_updated += 1
        elif status in _BROKER_PARTIAL:
            order_repo.mark_fill(
                cid, filled_qty=broker_order.filled_qty,
                filled_avg_price=None, status=PARTIALLY_FILLED,
            )
            report.orders_updated += 1
        elif status in _BROKER_DEAD:
            order_repo.mark_fill(
                cid, filled_qty=broker_order.filled_qty,
                filled_avg_price=None, status=CANCELED,
            )
            report.orders_updated += 1

    # 3) Auditoria.
    audit.write(
        SYSTEM, "reconcile_on_boot",
        payload={
            "positions_synced": report.positions_synced,
            "orders_updated": report.orders_updated,
            "issues": report.issues,
        },
    )
    if report.issues:
        logger.warning("Reconciliacao encontrou divergencias: %s", report.issues)
    else:
        logger.info(
            "Reconciliacao OK: %d posicoes, %d ordens atualizadas.",
            report.positions_synced, report.orders_updated,
        )
    return report


def _safe_lookup(broker: BrokerClient, cid: str | None):
    if cid is None:
        return None
    try:
        return broker.get_order_by_client_id(cid)
    except Exception:
        return None
