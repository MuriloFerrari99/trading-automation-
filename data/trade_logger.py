"""Auditoria de trades.

Requisito inegociavel do briefing: todo trade executado deve ser logado
(timestamp, ativo, lado, qtd, preco, estrategia que originou) para auditoria.

Esta classe escreve tanto no SQLite (tabela trade_log) quanto via logging
padrao do Python, dando uma trilha redundante.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.models import OrderIntent, OrderResult
from data.db import Database

logger = logging.getLogger("trade_audit")


class TradeLogger:
    """Registra trades no SQLite e no log de aplicacao."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def log_execution(
        self,
        intent: OrderIntent,
        result: OrderResult,
    ) -> int:
        """Registra um trade executado a partir da intencao e do resultado.

        Retorna o id da linha em trade_log.
        """
        ts = datetime.now(timezone.utc).isoformat()
        price = result.filled_avg_price
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_log
                    (ts, symbol, side, qty, price, strategy, broker_order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    intent.symbol,
                    intent.side.value,
                    str(intent.qty),
                    str(price) if price is not None else None,
                    intent.strategy,
                    result.broker_order_id,
                    result.status,
                ),
            )
            row_id = int(cur.lastrowid)
        logger.info(
            "TRADE %s %s qty=%s price=%s strategy=%s order_id=%s status=%s",
            intent.side.value.upper(),
            intent.symbol,
            intent.qty,
            price if price is not None else "n/a",
            intent.strategy,
            result.broker_order_id,
            result.status,
        )
        return row_id

    def recent(self, limit: int = 50) -> list[dict]:
        """Retorna os ultimos trades logados (mais recentes primeiro)."""
        cur = self._db.conn.execute(
            "SELECT * FROM trade_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]
