"""Repositorio de posicoes (tabela `positions`).

Snapshot local das posicoes, reconciliado contra o broker (que e a fonte de
verdade). As estrategias leem posicoes ao vivo do broker; este snapshot serve
para auditoria, reconciliacao e deteccao de divergencias.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.models import Position
from data.db import Database


class PositionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, position: Position) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO positions (symbol, qty, avg_entry_price, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty = excluded.qty,
                    avg_entry_price = excluded.avg_entry_price,
                    updated_at = excluded.updated_at
                """,
                (position.symbol, str(position.qty), str(position.avg_entry_price), now),
            )

    def delete(self, symbol: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol.upper(),))

    def symbols(self) -> set[str]:
        cur = self._db.conn.execute("SELECT symbol FROM positions")
        return {row["symbol"] for row in cur.fetchall()}

    def all(self) -> list[dict]:
        cur = self._db.conn.execute("SELECT * FROM positions ORDER BY symbol")
        return [dict(row) for row in cur.fetchall()]

    def replace_all_from_broker(self, positions: list[Position]) -> None:
        """Substitui o snapshot local pelo que o broker reporta (broker vence)."""
        broker_symbols = {p.symbol for p in positions}
        for p in positions:
            self.upsert(p)
        # Remove posicoes locais que o broker nao reporta mais (stale).
        for sym in self.symbols() - broker_symbols:
            self.delete(sym)
