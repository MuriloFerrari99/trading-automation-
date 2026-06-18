"""Consultas e agregacoes para o dashboard.

Tudo aqui le o SQLite em modo SOMENTE-LEITURA (nao escreve, nao migra schema) e
devolve estruturas JSON-serializaveis (dict/list de tipos primitivos). Os
valores Decimal sao armazenados como TEXT no banco; convertemos para float
apenas na fronteira de apresentacao, com tolerancia a campos nulos/ausentes.

Nenhuma dependencia de rede ou da corretora: este modulo funciona offline a
partir do banco. O enriquecimento ao vivo (preco atual, P&L nao-realizado) e
opcional e injetado de fora (ver `dashboard.broker_view`).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/trading.sqlite")

# Desfechos que representam um trade fechado com P&L conhecido.
_CLOSED_STATUSES = ("win", "loss", "breakeven")


def _connect_ro(db_path: Path | str) -> sqlite3.Connection:
    """Abre o SQLite em modo somente-leitura (nao cria arquivo se faltar)."""
    path = Path(db_path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# --------------------------------------------------------------------------- #
# Blocos individuais
# --------------------------------------------------------------------------- #


def _closed_decisions(conn: sqlite3.Connection) -> list[dict]:
    """Decisoes que viraram trade e fecharam (win/loss/breakeven), com P&L."""
    if not _table_exists(conn, "decisions"):
        return []
    rows = _rows(
        conn,
        """
        SELECT id, ts, strategy, symbol, action, regime, reference_price,
               outcome_status, entry_price, exit_price, realized_pnl,
               return_pct, closed_at, outcome_note
        FROM decisions
        WHERE outcome_status IN (?, ?, ?)
        ORDER BY COALESCE(closed_at, ts) DESC, id DESC
        """,
        _CLOSED_STATUSES,
    )
    for r in rows:
        r["realized_pnl"] = _to_float(r.get("realized_pnl"))
        r["return_pct"] = _to_float(r.get("return_pct"))
        r["entry_price"] = _to_float(r.get("entry_price"))
        r["exit_price"] = _to_float(r.get("exit_price"))
    return rows


def _open_decisions(conn: sqlite3.Connection) -> list[dict]:
    if not _table_exists(conn, "decisions"):
        return []
    rows = _rows(
        conn,
        """
        SELECT id, ts, strategy, symbol, action, regime, reference_price,
               entry_price, signal_strength
        FROM decisions
        WHERE outcome_status = 'open'
        ORDER BY ts DESC, id DESC
        """,
    )
    for r in rows:
        r["reference_price"] = _to_float(r.get("reference_price"))
        r["entry_price"] = _to_float(r.get("entry_price"))
        r["signal_strength"] = _to_float(r.get("signal_strength"))
    return rows


def _recent_trades(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Execucoes registradas em trade_log (auditoria imutavel)."""
    if not _table_exists(conn, "trade_log"):
        return []
    rows = _rows(
        conn,
        """
        SELECT id, ts, symbol, side, qty, price, strategy, broker_order_id, status
        FROM trade_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    for r in rows:
        r["qty"] = _to_float(r.get("qty"))
        r["price"] = _to_float(r.get("price"))
    return rows


def _positions_local(conn: sqlite3.Connection) -> list[dict]:
    """Snapshot local de posicoes (cache; corretora e a fonte de verdade)."""
    if not _table_exists(conn, "positions"):
        return []
    rows = _rows(
        conn,
        "SELECT symbol, qty, avg_entry_price, updated_at FROM positions ORDER BY symbol",
    )
    for r in rows:
        r["qty"] = _to_float(r.get("qty"))
        r["avg_entry_price"] = _to_float(r.get("avg_entry_price"))
    return rows


def _open_orders(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    if not _table_exists(conn, "orders"):
        return []
    rows = _rows(
        conn,
        """
        SELECT ts, symbol, side, qty, order_type, limit_price, stop_price,
               strategy, status, filled_qty, filled_avg_price, updated_at
        FROM orders
        WHERE status IN ('PENDING_SUBMIT', 'SUBMITTED', 'PARTIALLY_FILLED')
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    for r in rows:
        r["qty"] = _to_float(r.get("qty"))
        r["filled_qty"] = _to_float(r.get("filled_qty"))
        r["filled_avg_price"] = _to_float(r.get("filled_avg_price"))
    return rows


def _recent_audit(conn: sqlite3.Connection, limit: int = 40) -> list[dict]:
    if not _table_exists(conn, "audit_log"):
        return []
    rows = _rows(
        conn,
        "SELECT ts, actor, event, symbol, payload FROM audit_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    for r in rows:
        payload = r.get("payload")
        if payload:
            with contextlib.suppress(ValueError, TypeError):
                r["payload"] = json.loads(payload)
    return rows


# --------------------------------------------------------------------------- #
# Agregacoes
# --------------------------------------------------------------------------- #


def _summarize_pnl(closed: list[dict]) -> dict:
    """KPIs de P&L realizado a partir das decisoes fechadas."""
    pnls = [d["realized_pnl"] for d in closed if d["realized_pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n_closed": n,
        "total_pnl": round(sum(pnls), 2) if pnls else 0.0,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / n, 4) if n else None,
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "best": round(max(pnls), 2) if pnls else 0.0,
        "worst": round(min(pnls), 2) if pnls else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }


def _group_breakdown(closed: list[dict], key: str) -> list[dict]:
    """Agrupa P&L realizado por `key` (ex: 'strategy' ou 'regime')."""
    buckets: dict[str, dict] = {}
    for d in closed:
        k = d.get(key) or "—"
        b = buckets.setdefault(k, {"key": k, "n": 0, "wins": 0, "pnl": 0.0})
        pnl = d["realized_pnl"]
        if pnl is None:
            continue
        b["n"] += 1
        b["pnl"] += pnl
        if pnl > 0:
            b["wins"] += 1
    out = []
    for b in buckets.values():
        b["pnl"] = round(b["pnl"], 2)
        b["win_rate"] = round(b["wins"] / b["n"], 4) if b["n"] else None
        out.append(b)
    out.sort(key=lambda x: x["pnl"], reverse=True)
    return out


def _pnl_curve(closed: list[dict]) -> list[dict]:
    """Curva de P&L realizado ACUMULADO ao longo do tempo.

    `closed` chega ordenado do mais recente para o mais antigo; aqui invertemos
    para ordem cronologica e acumulamos. Cada ponto: {t, pnl, cum}.
    """
    chrono = [d for d in reversed(closed) if d.get("realized_pnl") is not None]
    cum = 0.0
    out = []
    for d in chrono:
        cum += d["realized_pnl"]
        out.append(
            {
                "t": d.get("closed_at") or d.get("ts"),
                "symbol": d.get("symbol"),
                "pnl": round(d["realized_pnl"], 2),
                "cum": round(cum, 2),
            }
        )
    return out


def _nav_rows(conn: sqlite3.Connection) -> list[dict]:
    """Linhas EOD de nav_history em ordem cronologica (somente-leitura).

    Le do mesmo arquivo SQLite que o resto do dashboard, sem escrever nem migrar
    schema. So o snapshot oficial (`source='eod'`) entra na curva."""
    if not _table_exists(conn, "nav_history"):
        return []
    return _rows(
        conn,
        """
        SELECT date, equity, cash, long_market_value, gross_exposure,
               net_exposure, regime, bench_spy, bench_6040, source
        FROM nav_history
        WHERE source = 'eod'
        ORDER BY date ASC, id ASC
        """,
    )


def _nav_curve(nav_rows: list[dict]) -> list[dict]:
    """Curva de equity vs benchmark p/ o grafico do dashboard.

    Normaliza estrategia, SPY e 60/40 para base 100 no 1o dia de NAV, para a
    comparacao visual ser de RETORNO (e nao de niveis incomparaveis: equity em
    USD vs preco de SPY vs cota do 60/40). Cada ponto: {t, equity, strat, spy, sf}.
    """
    if not nav_rows:
        return []

    def _base(rows: list[dict], key: str) -> float | None:
        for r in rows:
            v = _to_float(r.get(key))
            if v is not None and v > 0:
                return v
        return None

    eq0 = _base(nav_rows, "equity")
    spy0 = _base(nav_rows, "bench_spy")
    sf0 = _base(nav_rows, "bench_6040")

    out = []
    for r in nav_rows:
        eq = _to_float(r.get("equity"))
        spy = _to_float(r.get("bench_spy"))
        sf = _to_float(r.get("bench_6040"))
        out.append(
            {
                "t": r.get("date"),
                "equity": round(eq, 2) if eq is not None else None,
                "strat": round(eq / eq0 * 100, 3) if eq is not None and eq0 else None,
                "spy": round(spy / spy0 * 100, 3) if spy is not None and spy0 else None,
                "sf": round(sf / sf0 * 100, 3) if sf is not None and sf0 else None,
                "regime": r.get("regime"),
            }
        )
    return out


def _track_record_kpis(nav_rows: list[dict]) -> dict:
    """KPIs since-inception sobre nav_history, via reporting.track_record (mesma
    fonte e mesmas formulas do relatorio de texto -> dashboard e relatorio batem).

    Tolerante a serie vazia/curta: devolve dict com chaves nulas. Import tardio
    para nao acoplar o dashboard a numpy no caminho em que nao ha NAV."""
    if len(nav_rows) < 2:
        return {
            "n_days": len(nav_rows), "since": None, "cagr": None, "vol": None,
            "sharpe": None, "sortino": None, "max_drawdown": None, "calmar": None,
            "excess_cagr": None, "beta_vs_spy": None, "pct_positive_months": None,
        }
    import numpy as np  # noqa: PLC0415 - tardio de proposito

    from reporting.track_record import NavSeries, compute

    dates = [r["date"] for r in nav_rows]
    equity = np.array([_to_float(r.get("equity")) or float("nan") for r in nav_rows], dtype=float)
    spy = np.array([_to_float(r.get("bench_spy")) or float("nan") for r in nav_rows], dtype=float)
    sf = np.array([_to_float(r.get("bench_6040")) or float("nan") for r in nav_rows], dtype=float)
    m = compute(NavSeries(dates=dates, equity=equity, bench_spy=spy, bench_6040=sf))
    return {
        "n_days": m.n_days,
        "since": m.start,
        "end": m.end,
        "cagr": m.cagr,
        "vol": m.vol_annual,
        "sharpe": round(m.sharpe, 3),
        "sortino": round(m.sortino, 3),
        "max_drawdown": m.max_drawdown,
        "calmar": round(m.calmar, 3) if m.calmar not in (float("inf"),) else None,
        "excess_cagr": m.excess_cagr,
        "beta_vs_spy": round(m.beta_vs_spy, 3) if m.beta_vs_spy is not None else None,
        "pct_positive_months": m.pct_positive_months,
    }


def _top_trades(closed: list[dict], n: int = 5) -> dict:
    """Maiores ganhos e maiores perdas (principais trades)."""
    ranked = [d for d in closed if d["realized_pnl"] is not None]
    ranked.sort(key=lambda d: d["realized_pnl"], reverse=True)
    cols = ("symbol", "strategy", "regime", "realized_pnl", "return_pct",
            "entry_price", "exit_price", "closed_at")
    def slim(d: dict) -> dict:
        return {c: d.get(c) for c in cols}
    return {
        "winners": [slim(d) for d in ranked[:n] if d["realized_pnl"] > 0],
        "losers": [slim(d) for d in reversed(ranked[-n:]) if d["realized_pnl"] < 0],
    }


# --------------------------------------------------------------------------- #
# Ponto de entrada
# --------------------------------------------------------------------------- #


def build_dashboard_data(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    account: dict | None = None,
    live_positions: list[dict] | None = None,
    broker_status: str = "off",
) -> dict:
    """Monta o payload completo do dashboard (JSON-serializavel).

    `account`/`live_positions` sao enriquecimentos opcionais vindos da corretora
    (ver `dashboard.broker_view`). `broker_status` e um rotulo informativo
    ("live", "off", "error: ...") exibido no cabecalho.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    path = Path(db_path)
    if not path.exists():
        return {
            "generated_at": generated_at,
            "broker_status": broker_status,
            "error": f"Banco nao encontrado: {path}",
            "kpis": {}, "positions": [], "open_orders": [], "recent_trades": [],
            "open_decisions": [], "top_trades": {"winners": [], "losers": []},
            "pnl_curve": [], "by_strategy": [], "by_regime": [], "audit": [],
            "nav_curve": [], "track_record": {},
        }

    conn = _connect_ro(path)
    try:
        closed = _closed_decisions(conn)
        pnl = _summarize_pnl(closed)
        local_positions = _positions_local(conn)
        recent_trades = _recent_trades(conn)
        open_decisions = _open_decisions(conn)
        open_orders = _open_orders(conn)
        audit = _recent_audit(conn)
        nav_rows = _nav_rows(conn)
    finally:
        conn.close()

    nav_curve = _nav_curve(nav_rows)
    track_record = _track_record_kpis(nav_rows)

    # Posicoes: prefere a visao ao vivo da corretora (tem preco/PnL nao realizado);
    # cai para o snapshot local quando o broker esta off.
    positions = live_positions if live_positions is not None else local_positions
    unrealized = sum(
        p["unrealized_pnl"] for p in positions
        if isinstance(p, dict) and p.get("unrealized_pnl") is not None
    )

    kpis = {
        "equity": (account or {}).get("equity"),
        "cash": (account or {}).get("cash"),
        "buying_power": (account or {}).get("buying_power"),
        "open_positions": len(positions),
        "unrealized_pnl": round(unrealized, 2) if positions else 0.0,
        **pnl,
    }

    return {
        "generated_at": generated_at,
        "broker_status": broker_status,
        "kpis": kpis,
        "positions": positions,
        "open_orders": open_orders,
        "recent_trades": recent_trades,
        "open_decisions": open_decisions,
        "top_trades": _top_trades(closed),
        "pnl_curve": _pnl_curve(closed),
        "by_strategy": _group_breakdown(closed, "strategy"),
        "by_regime": _group_breakdown(closed, "regime"),
        "audit": audit,
        # Track record (Fase 1): curva de equity vs benchmark + KPIs since-inception.
        "nav_curve": nav_curve,
        "track_record": track_record,
    }
