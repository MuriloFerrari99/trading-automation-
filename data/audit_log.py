"""Trilha de auditoria (tabela `audit_log`).

Registra o "porque" de cada decisao/ordem com o ATOR que a produziu. Sem isso
e impossivel explicar por que o bot fez um trade (debugging e compliance) —
erro comum nº9 do doc 05.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from data.db import Database

# Atores padrao.
PLANNER = "planner"
EXECUTOR = "executor"
MONITOR = "monitor"
SYSTEM = "system"


class AuditLog:
    def __init__(self, db: Database) -> None:
        self._db = db

    def write(
        self,
        actor: str,
        event: str,
        *,
        symbol: str | None = None,
        payload: dict | None = None,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, default=str) if payload else None
        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO audit_log (ts, actor, event, symbol, payload) VALUES (?, ?, ?, ?, ?)",
                (ts, actor, event, symbol, payload_json),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 100) -> list[dict]:
        cur = self._db.conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]
