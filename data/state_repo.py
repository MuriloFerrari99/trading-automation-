"""Repositorio de estado chave/valor (tabela `state`).

Usado pelas estrategias para persistir estado entre ciclos do Monitor — por
exemplo, o high-water mark (maxima observada) do Trailing Stop por ativo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from data.db import Database


class StateRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str) -> str | None:
        cur = self._db.conn.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set(self, key: str, value: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at
                """,
                (key, value, ts),
            )

    def delete(self, key: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM state WHERE key = ?", (key,))

    def get_decimal(self, key: str) -> Decimal | None:
        v = self.get(key)
        return Decimal(v) if v is not None else None

    def set_decimal(self, key: str, value: Decimal) -> None:
        self.set(key, str(value))
