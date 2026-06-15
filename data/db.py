"""Persistencia local em SQLite.

Centraliza a conexao e o schema. Tabelas:
- trade_log: auditoria imutavel de todo trade executado (requisito do briefing).
- orders:    intencoes/ordens submetidas e seu status.
- positions: snapshot das posicoes (cache local, fonte de verdade e a corretora).
- state:     pares chave/valor para estado dos agentes (ex: trailing high-water).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_DB_PATH = Path("data/trading.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,   -- ISO-8601 UTC
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             TEXT    NOT NULL,   -- Decimal serializado como texto
    price           TEXT,              -- preco de execucao (pode ser nulo p/ market pendente)
    strategy        TEXT    NOT NULL,
    broker_order_id TEXT,
    status          TEXT    NOT NULL DEFAULT 'submitted'
);

CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             TEXT    NOT NULL,
    order_type      TEXT    NOT NULL,
    limit_price     TEXT,
    stop_price      TEXT,
    strategy        TEXT    NOT NULL,
    broker_order_id TEXT,
    status          TEXT    NOT NULL DEFAULT 'accepted'
);

CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT    PRIMARY KEY,
    qty             TEXT    NOT NULL,
    avg_entry_price TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key             TEXT    PRIMARY KEY,
    value           TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

-- Sinais/sugestoes (Nivel 2). NUNCA executam automaticamente: sao apenas
-- input/sugestao para o Planejador e ficam aqui para revisao humana e auditoria.
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,   -- quando foi registrado
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    source          TEXT    NOT NULL,   -- ex: congress, smart_money
    confidence      TEXT    NOT NULL,
    note            TEXT,
    created_at      TEXT    NOT NULL    -- timestamp do proprio sinal
);
"""


class Database:
    """Wrapper fino sobre sqlite3 com schema garantido na inicializacao."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        if self._path.parent and str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: o scheduler pode rodar em thread separada.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def close(self) -> None:
        self._conn.close()
