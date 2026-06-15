"""Persistencia do loop de feedback: a tabela `decisions`.

Denormalizada de proposito (decisao + contexto + resultado numa linha): e
simples, suficiente para avaliacao, e mantem a Camada 0 desacoplada de db.py.
Cria a propria tabela com CREATE TABLE IF NOT EXISTS no mesmo arquivo SQLite do
resto do sistema (data/trading.sqlite por padrao), em modo WAL para conviver
com a conexao principal.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from feedback.models import Decision, Outcome, OutcomeStatus

DEFAULT_DB_PATH = Path("data/trading.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,   -- quando a decisao foi tomada (ISO UTC)
    strategy        TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,   -- buy | sell | hold | skip
    regime          TEXT    NOT NULL,   -- trend_up | trend_down | range | high_vol | unknown
    reference_price TEXT,
    signal_strength REAL    NOT NULL DEFAULT 0,
    context         TEXT,               -- JSON livre (features/sinais)
    client_order_id TEXT,
    -- resultado (preenchido quando conhecido):
    outcome_status  TEXT    NOT NULL DEFAULT 'open',
    entry_price     TEXT,
    exit_price      TEXT,
    realized_pnl    TEXT,
    return_pct      REAL,
    closed_at       TEXT,
    outcome_note    TEXT,
    -- lote (para matching FIFO de vendas contra compras):
    qty             TEXT,               -- qty da ordem; setada no fill de compra
    remaining_qty   TEXT                -- qty ainda aberta; 0 => totalmente fechada
);
CREATE INDEX IF NOT EXISTS idx_decisions_strategy ON decisions(strategy);
CREATE INDEX IF NOT EXISTS idx_decisions_regime   ON decisions(regime);
CREATE INDEX IF NOT EXISTS idx_decisions_coid     ON decisions(client_order_id);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol   ON decisions(symbol);
"""

# Colunas adicionadas depois da v1 — migração idempotente para bancos existentes.
_MIGRATION_COLUMNS = (("qty", "TEXT"), ("remaining_qty", "TEXT"))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dec_to_str(v: Decimal | None) -> str | None:
    return None if v is None else str(v)


class DecisionLog:
    """Grava decisoes e anexa resultados. Fonte do dataset de aprendizado."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            self._conn = connection
            self._owns_conn = False
        else:
            path = Path(db_path)
            if str(path) != ":memory:" and path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._owns_conn = True
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._conn:
            self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Adiciona colunas novas a bancos pre-existentes (ALTER idempotente).
        CREATE TABLE IF NOT EXISTS nao altera tabela ja criada — daqui vem a
        coluna de lote em DBs antigos."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(decisions)")}
        with self._conn:
            for name, coltype in _MIGRATION_COLUMNS:
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {coltype}")

    # ------------------------------ escrita ------------------------------ #

    def record(self, decision: Decision) -> int:
        """Grava uma decisao (status inicial 'open' se virou trade; senao o
        chamador pode anexar 'skipped' logo em seguida). Retorna o id da linha."""
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO decisions (
                    ts, strategy, symbol, action, regime, reference_price,
                    signal_strength, context, client_order_id, outcome_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.created_at.astimezone(timezone.utc).isoformat(),
                    decision.strategy,
                    decision.symbol,
                    decision.action.value,
                    decision.regime.value,
                    _dec_to_str(decision.reference_price),
                    float(decision.signal_strength),
                    json.dumps(decision.context, default=str, ensure_ascii=False),
                    decision.client_order_id,
                    # decisoes que nao operam ja nascem com desfecho 'skipped'.
                    OutcomeStatus.SKIPPED.value
                    if decision.action.value in ("hold", "skip")
                    else OutcomeStatus.OPEN.value,
                ),
            )
            return int(cur.lastrowid)

    def attach_outcome(self, decision_id: int, outcome: Outcome) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE decisions SET
                    outcome_status = ?, entry_price = ?, exit_price = ?,
                    realized_pnl = ?, return_pct = ?, closed_at = ?, outcome_note = ?
                WHERE id = ?
                """,
                (
                    outcome.status.value,
                    _dec_to_str(outcome.entry_price),
                    _dec_to_str(outcome.exit_price),
                    _dec_to_str(outcome.realized_pnl),
                    outcome.return_pct,
                    (outcome.closed_at or datetime.now(timezone.utc)).isoformat()
                    if outcome.status not in (OutcomeStatus.OPEN, OutcomeStatus.SKIPPED)
                    else (outcome.closed_at.isoformat() if outcome.closed_at else None),
                    outcome.note,
                    decision_id,
                ),
            )

    def set_entry_price_by_client_order_id(self, coid: str, entry_price: Decimal) -> int:
        """Fixa o entry_price REAL (preco de fill da compra) nas decisoes abertas
        daquele client_order_id. Retorna quantas linhas foram atualizadas.

        Isso torna o P&L do loop de feedback fiel ao fill real (e nao ao
        reference_price da hora da decisao)."""
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE decisions SET entry_price = ?
                WHERE client_order_id = ? AND outcome_status = ?
                """,
                (_dec_to_str(entry_price), coid, OutcomeStatus.OPEN.value),
            )
            return cur.rowcount

    def set_qty_by_client_order_id(self, coid: str, qty: Decimal) -> int:
        """Fixa qty/remaining_qty (lote) nas decisoes ABERTAS do client_order_id,
        apenas enquanto ainda nao setado (idempotente em retry/reconexao).
        Retorna quantas linhas foram atualizadas."""
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE decisions SET qty = ?, remaining_qty = ?
                WHERE client_order_id = ? AND outcome_status = ? AND qty IS NULL
                """,
                (_dec_to_str(qty), _dec_to_str(qty), coid, OutcomeStatus.OPEN.value),
            )
            return cur.rowcount

    def open_buy_lots(self, symbol: str) -> list[dict]:
        """Lotes de COMPRA ainda abertos do simbolo, em ordem FIFO (mais antigo
        primeiro). Base do matching lote-a-lote de vendas contra compras."""
        cur = self._conn.execute(
            """
            SELECT * FROM decisions
            WHERE upper(symbol) = upper(?) AND action = 'buy' AND outcome_status = ?
            ORDER BY ts ASC, id ASC
            """,
            (symbol, OutcomeStatus.OPEN.value),
        )
        return [dict(r) for r in cur.fetchall()]

    def reduce_remaining(self, decision_id: int, new_remaining: Decimal) -> None:
        """Abate a qty ainda aberta de um lote parcialmente vendido (segue OPEN)."""
        with self._conn:
            self._conn.execute(
                "UPDATE decisions SET remaining_qty = ? WHERE id = ?",
                (_dec_to_str(new_remaining), decision_id),
            )

    def attach_outcome_by_client_order_id(self, coid: str, outcome: Outcome) -> int:
        """Anexa resultado a decisao(oes) ligada(s) a um client_order_id.
        Retorna quantas linhas foram atualizadas."""
        with self._conn:
            cur = self._conn.execute(
                "SELECT id FROM decisions WHERE client_order_id = ?", (coid,)
            )
            ids = [int(r["id"]) for r in cur.fetchall()]
        for did in ids:
            self.attach_outcome(did, outcome)
        return len(ids)

    # ------------------------------ leitura ------------------------------ #

    def all_records(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM decisions ORDER BY ts ASC, id ASC")
        return [dict(r) for r in cur.fetchall()]

    def open_decisions(self) -> list[dict]:
        """Decisoes que viraram trade e ainda estao abertas — uteis para o
        Monitor reconciliar e anexar o resultado quando fecharem."""
        cur = self._conn.execute(
            "SELECT * FROM decisions WHERE outcome_status = ? ORDER BY ts ASC",
            (OutcomeStatus.OPEN.value,),
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()
