"""Repositorio do ciclo de vida das ordens (tabela `orders`).

Padrao de escrita seguro (doc 02 §7.4 / doc 05 §7): a intencao e gravada como
PENDING_SUBMIT ANTES da chamada de rede; se o processo cair, a reconciliacao
sabe que existia uma ordem pendente. Depois marca-se SUBMITTED (com o id do
broker) e, conforme os fills chegam, FILLED / PARTIALLY_FILLED.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from core.models import OptionOrderIntent, OrderIntent, OrderType
from data.db import Database

# Estados do ciclo de vida.
PENDING_SUBMIT = "PENDING_SUBMIT"
SUBMITTED = "SUBMITTED"
FILLED = "FILLED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
REJECTED = "REJECTED"
CANCELED = "CANCELED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrderRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def record_pending(self, intent: OrderIntent | OptionOrderIntent) -> str:
        """Grava a ordem como PENDING_SUBMIT ANTES de qualquer chamada de rede.

        Idempotente: se o client_order_id ja existe (retry), nao duplica.
        Retorna o client_order_id.
        """
        cid = intent.client_order_id
        if isinstance(intent, OptionOrderIntent):
            symbol = intent.contract.occ_symbol
            order_type = "option"
            limit_price = None
            stop_price = None
        else:
            symbol = intent.symbol
            order_type = intent.order_type.value
            limit_price = str(intent.limit_price) if intent.limit_price is not None else None
            stop_price = str(intent.stop_price) if intent.stop_price is not None else None

        now = _now()
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO orders
                    (client_order_id, ts, symbol, side, qty, order_type,
                     limit_price, stop_price, strategy, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO NOTHING
                """,
                (
                    cid, now, symbol, intent.side.value, str(intent.qty), order_type,
                    limit_price, stop_price, intent.strategy, PENDING_SUBMIT, now,
                ),
            )
        return cid

    def mark_submitted(self, client_order_id: str, broker_order_id: str, status: str = SUBMITTED) -> None:
        self._update(client_order_id, broker_order_id=broker_order_id, status=status)

    def mark_fill(
        self,
        client_order_id: str,
        *,
        filled_qty: Decimal,
        filled_avg_price: Decimal | None,
        status: str,
    ) -> None:
        self._update(
            client_order_id,
            status=status,
            filled_qty=str(filled_qty),
            filled_avg_price=str(filled_avg_price) if filled_avg_price is not None else None,
        )

    def mark_rejected(self, client_order_id: str) -> None:
        self._update(client_order_id, status=REJECTED)

    def _update(self, client_order_id: str, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE orders SET {sets}, updated_at = ? WHERE client_order_id = ?",
                (*values, _now(), client_order_id),
            )

    def get(self, client_order_id: str) -> dict | None:
        cur = self._db.conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def pending_or_submitted(self) -> list[dict]:
        """Ordens que o bot acredita estarem em aberto (p/ reconciliacao)."""
        cur = self._db.conn.execute(
            "SELECT * FROM orders WHERE status IN (?, ?, ?)",
            (PENDING_SUBMIT, SUBMITTED, PARTIALLY_FILLED),
        )
        return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def is_trailing(intent: OrderIntent | OptionOrderIntent) -> bool:
        return (
            isinstance(intent, OrderIntent)
            and intent.order_type == OrderType.TRAILING_STOP
        )
