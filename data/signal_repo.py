"""Repositorio de sinais (tabela `signals`).

Persiste sugestoes geradas pelos SignalProviders para auditoria e revisao
humana. Sinais NUNCA viram ordens automaticamente — este repositorio e o
ponto onde um humano (ou um futuro fluxo de aprovacao) os consulta.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.models import Signal
from data.db import Database


class SignalRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, signal: Signal) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals (ts, symbol, side, source, confidence, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    signal.symbol,
                    signal.side.value,
                    signal.source,
                    str(signal.confidence),
                    signal.note,
                    signal.created_at.isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def record_many(self, signals: list[Signal]) -> int:
        return sum(1 for s in signals if self.record(s))

    def recent(self, limit: int = 50) -> list[dict]:
        cur = self._db.conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]
