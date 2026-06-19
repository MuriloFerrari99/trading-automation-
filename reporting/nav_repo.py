"""Tabela `nav_history` — curva de equity mark-to-market, AUDITAVEL e IMUTAVEL.

Esta tabela e o ATIVO CENTRAL do track record (Fase 1 do PLAN_BETA_DISCIPLINADO).
Um track record so vale para captacao de AUM se for *a prova de cherry-picking*:
registrado desde o 1o dia, sem reescrita conveniente de historico. Por isso o
desenho aqui prioriza INTEGRIDADE acima de comodidade.

PADRAO DE PERSISTENCIA (igual a feedback/decision_log.py)
---------------------------------------------------------
- Cria a propria tabela com CREATE TABLE IF NOT EXISTS no MESMO data/trading.sqlite.
- Aceita ou um db_path (abre conexao propria) ou uma `connection` compartilhada
  com o resto do sistema (Database.conn) — em WAL para conviver com a conexao
  principal sem travar.

DOIS PRINCIPIOS DE INTEGRIDADE (nao-negociaveis)
------------------------------------------------
1. APPEND-ONLY DO HISTORICO SELADO. Ha *um snapshot oficial por dia* (coluna
   `date` UNIQUE). Durante o PROPRIO dia, re-rodar a captura faz UPSERT do
   snapshot daquele dia (corrige um valor intradiario para o EOD final) — isso e
   idempotencia, nao reescrita de historico. Mas assim que existe QUALQUER linha
   de uma data POSTERIOR, o dia anterior esta SELADO: tentar reescreve-lo levanta
   ImmutableHistoryError. Ou seja: nunca se altera um dia ja "fechado pelo dia
   seguinte", e nunca se faz backfill de uma data no meio/passado.

2. CADEIA DE HASH (estilo blockchain). Cada linha guarda:
       row_hash = sha256( payload_canonico | prev_hash )
   onde prev_hash e o row_hash da linha imediatamente anterior (genesis = "").
   Alterar OU remover qualquer linha passada quebra o hash dela e/ou o
   encadeamento de todas as seguintes — detectavel por reporting/verify_chain.py.
   Isto torna a adulteracao *evidente para um auditor externo* mesmo que alguem
   tenha acesso de escrita direto ao SQLite (o atacante teria de recomputar TODA
   a cadeia a partir do ponto adulterado, e ainda assim um export/commit diario
   anterior do CSV — backup WORM — denunciaria a divergencia).

COMO UM AUDITOR EXTERNO VERIFICA (resumo; detalhe em verify_chain.py)
--------------------------------------------------------------------
    uv run python -m reporting.verify_chain
recomputa a cadeia inteira a partir dos campos crus e compara com os row_hash
gravados. Verde => nenhuma linha foi alterada/removida/inserida fora de ordem.
Cruzando com o backup append-only externo (ex.: commit git diario do CSV), a
prova fica independente do banco vivo.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("data/trading.sqlite")

# Genesis: prev_hash da PRIMEIRA linha. String fixa e publica (faz parte da prova).
GENESIS_PREV_HASH = ""

SCHEMA = """
CREATE TABLE IF NOT EXISTS nav_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,            -- ISO-8601 UTC do registro
    date               TEXT    NOT NULL UNIQUE,     -- YYYY-MM-DD (1 snapshot oficial/dia)
    equity             TEXT    NOT NULL,            -- NAV mark-to-market (Decimal->texto)
    cash               TEXT    NOT NULL,
    long_market_value  TEXT    NOT NULL,
    gross_exposure     TEXT    NOT NULL,            -- valor bruto de mercado / equity
    net_exposure       TEXT    NOT NULL,            -- exposicao liquida / equity
    regime             TEXT    NOT NULL DEFAULT 'unknown',
    bench_spy          TEXT,                        -- preco SPY (total-return) do dia
    bench_6040         TEXT,                        -- NAV sintetico 60/40 do dia
    source             TEXT    NOT NULL DEFAULT 'eod',  -- 'eod' | 'intraday'
    prev_hash          TEXT    NOT NULL,
    row_hash           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history(date);
"""

# Campos que entram no hash, NESTA ORDEM. Mudar a ordem/conjunto invalida cadeias
# antigas — por isso e congelado e documentado (parte do contrato de auditoria).
_HASH_FIELDS = (
    "date",
    "equity",
    "cash",
    "long_market_value",
    "gross_exposure",
    "net_exposure",
    "regime",
    "bench_spy",
    "bench_6040",
    "source",
)


class ImmutableHistoryError(RuntimeError):
    """Tentativa de reescrever um dia ja SELADO (existe data posterior)."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(v) -> str:
    """Serializacao canonica e estavel de um campo para o hash.

    None -> "" (string vazia, distinta e reproduzivel). Numeros/Decimais sao
    convertidos pelo chamador para str ANTES de chegar aqui, de modo que o hash
    sempre veja exatamente o texto que foi/sera gravado no banco.
    """
    return "" if v is None else str(v)


def compute_row_hash(fields: dict, prev_hash: str) -> str:
    """Hash canonico de uma linha encadeado ao anterior.

    row_hash = sha256( "date=...|equity=...|...|prev=<prev_hash>" ).
    Determinista e reproduzivel por um terceiro a partir SO dos campos crus.
    """
    parts = [f"{name}={_norm(fields.get(name))}" for name in _HASH_FIELDS]
    parts.append(f"prev={_norm(prev_hash)}")
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class NavSnapshot:
    """Um ponto da curva de equity (linha de nav_history)."""

    date: str
    equity: str
    cash: str
    long_market_value: str
    gross_exposure: str
    net_exposure: str
    regime: str = "unknown"
    bench_spy: str | None = None
    bench_6040: str | None = None
    source: str = "eod"
    ts: str | None = None
    prev_hash: str | None = None
    row_hash: str | None = None

    def hash_fields(self) -> dict:
        return {
            "date": self.date,
            "equity": self.equity,
            "cash": self.cash,
            "long_market_value": self.long_market_value,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "regime": self.regime,
            "bench_spy": self.bench_spy,
            "bench_6040": self.bench_6040,
            "source": self.source,
        }


def _to_str(v) -> str:
    """Decimal/float/int -> texto; usado para normalizar entradas numericas."""
    return str(v)


def _opt_str(v) -> str | None:
    return None if v is None else str(v)


class NavHistoryRepo:
    """Grava e le a curva de equity append-only com cadeia de hash."""

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

    # ------------------------------ leitura ------------------------------ #

    def last_row(self) -> dict | None:
        """Ultima linha por (date, id) — a 'ponta' da cadeia."""
        cur = self._conn.execute(
            "SELECT * FROM nav_history ORDER BY date DESC, id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_by_date(self, day: str) -> dict | None:
        cur = self._conn.execute("SELECT * FROM nav_history WHERE date = ?", (day,))
        row = cur.fetchone()
        return dict(row) if row else None

    def max_sealed_date(self) -> str | None:
        """Maior `date` ja gravada. Qualquer data <= esta (exceto ela propria) e
        considerada SELADA (ha pelo menos um dia posterior)."""
        cur = self._conn.execute("SELECT MAX(date) AS d FROM nav_history")
        row = cur.fetchone()
        return row["d"] if row and row["d"] else None

    def all_rows(self) -> list[dict]:
        """Toda a serie, em ordem cronologica (a ordem da cadeia)."""
        cur = self._conn.execute(
            "SELECT * FROM nav_history ORDER BY date ASC, id ASC"
        )
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM nav_history").fetchone()["n"])

    # ------------------------------ escrita ------------------------------ #

    def record_snapshot(
        self,
        *,
        day: str | date | None = None,
        equity,
        cash,
        long_market_value,
        gross_exposure,
        net_exposure,
        regime: str = "unknown",
        bench_spy=None,
        bench_6040=None,
        source: str = "eod",
    ) -> dict:
        """Grava (ou atualiza, se for o MESMO dia ainda nao selado) o snapshot do dia.

        Idempotencia diaria: chamar duas vezes no mesmo `day` NAO cria duas linhas
        — a 2a faz UPSERT do snapshot do dia (recomputando o hash, ja que o dia
        ainda nao foi selado por um dia posterior).

        Imutabilidade: se `day` for ANTERIOR a maior data ja gravada (i.e., o dia
        ja foi selado por um dia posterior, ou seria um backfill no passado),
        levanta ImmutableHistoryError. Nunca altera linhas de dias anteriores.

        Aceita valores como Decimal/float/int/str; tudo e normalizado para texto
        ANTES do hash, para o hash refletir exatamente o que e gravado.
        """
        day_str = _coerce_date(day)

        existing = self.get_by_date(day_str)
        max_date = self.max_sealed_date()

        # Bloqueio de reescrita de historico selado / backfill no passado.
        if existing is None and max_date is not None and day_str < max_date:
            raise ImmutableHistoryError(
                f"Backfill proibido: {day_str} e anterior ao ultimo dia gravado "
                f"({max_date}). nav_history e append-only por dia."
            )
        if existing is not None and max_date is not None and day_str < max_date:
            # O dia ja existe E ja foi selado por um dia posterior -> imutavel.
            raise ImmutableHistoryError(
                f"Dia {day_str} ja foi selado (existe data posterior {max_date}); "
                f"reescrever quebraria a cadeia de hash do track record."
            )

        # prev_hash = hash da linha imediatamente anterior NA CADEIA. Quando
        # atualizamos o dia corrente (UPSERT), o prev e o da linha anterior a ele.
        prev_hash = self._prev_hash_for(day_str, exclude_existing=existing is not None)

        snap = NavSnapshot(
            date=day_str,
            equity=_to_str(equity),
            cash=_to_str(cash),
            long_market_value=_to_str(long_market_value),
            gross_exposure=_to_str(gross_exposure),
            net_exposure=_to_str(net_exposure),
            regime=regime,
            bench_spy=_opt_str(bench_spy),
            bench_6040=_opt_str(bench_6040),
            source=source,
        )
        row_hash = compute_row_hash(snap.hash_fields(), prev_hash)
        ts = _utcnow_iso()

        with self._conn:
            # UPSERT por `date` (UNIQUE). No mesmo dia, atualiza tudo MENOS o id.
            self._conn.execute(
                """
                INSERT INTO nav_history (
                    ts, date, equity, cash, long_market_value, gross_exposure,
                    net_exposure, regime, bench_spy, bench_6040, source,
                    prev_hash, row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    ts = excluded.ts,
                    equity = excluded.equity,
                    cash = excluded.cash,
                    long_market_value = excluded.long_market_value,
                    gross_exposure = excluded.gross_exposure,
                    net_exposure = excluded.net_exposure,
                    regime = excluded.regime,
                    bench_spy = excluded.bench_spy,
                    bench_6040 = excluded.bench_6040,
                    source = excluded.source,
                    prev_hash = excluded.prev_hash,
                    row_hash = excluded.row_hash
                """,
                (
                    ts, snap.date, snap.equity, snap.cash, snap.long_market_value,
                    snap.gross_exposure, snap.net_exposure, snap.regime,
                    snap.bench_spy, snap.bench_6040, snap.source,
                    prev_hash, row_hash,
                ),
            )
        return self.get_by_date(day_str)

    def _prev_hash_for(self, day_str: str, *, exclude_existing: bool) -> str:
        """row_hash da linha que ANTECEDE `day_str` na cadeia.

        Para um dia NOVO (append na ponta): e o row_hash da ultima linha.
        Para um UPSERT do dia corrente: e o row_hash da ultima linha com date <
        day_str (ignora a propria linha do dia, que sera regravada).
        """
        if exclude_existing:
            cur = self._conn.execute(
                "SELECT row_hash FROM nav_history WHERE date < ? "
                "ORDER BY date DESC, id DESC LIMIT 1",
                (day_str,),
            )
        else:
            cur = self._conn.execute(
                "SELECT row_hash FROM nav_history ORDER BY date DESC, id DESC LIMIT 1"
            )
        row = cur.fetchone()
        return row["row_hash"] if row else GENESIS_PREV_HASH

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()


def _coerce_date(day: str | date | None) -> str:
    """Normaliza o dia para 'YYYY-MM-DD'. None => hoje (UTC)."""
    if day is None:
        return datetime.now(timezone.utc).date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    # ja string: aceita 'YYYY-MM-DD' ou ISO completo (corta o tempo).
    return str(day)[:10]
