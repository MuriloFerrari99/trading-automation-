"""Testes do sistema de TRACK RECORD (Fase 1 do PLAN_BETA_DISCIPLINADO).

Cobre o que torna o track record CREDIVEL para captacao de AUM:
  1. APPEND-ONLY / idempotencia diaria: 2 capturas no mesmo dia => 1 linha
     (UPSERT), nunca 2; e nunca altera dias anteriores.
  2. IMUTABILIDADE: reescrever um dia ja SELADO (existe data posterior) ou fazer
     backfill no passado => ImmutableHistoryError.
  3. CADEIA DE HASH: detecta adulteracao (linha alterada) e remocao de linha.
  4. METRICAS corretas sobre uma serie de NAV CONHECIDA (CAGR, MaxDD, mensais),
     reusando as formulas de simulation/.
  5. BENCHMARK 60/40 incremental reconstroi um 60/40 mensal.
  6. CAPTURA EOD idempotente e broker-agnostica (FakeBroker).
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal

import numpy as np
import pytest

from broker.fake_broker import FakeBroker
from reporting import benchmarks
from reporting.capture import capture_eod
from reporting.nav_repo import (
    GENESIS_PREV_HASH,
    ImmutableHistoryError,
    NavHistoryRepo,
    compute_row_hash,
)
from reporting.track_record import NavSeries, compute, load_series
from reporting.verify_chain import verify


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def repo(tmp_path):
    r = NavHistoryRepo(db_path=tmp_path / "nav.sqlite")
    yield r
    r.close()


def _snap(repo: NavHistoryRepo, day: str, equity, **kw):
    """Atalho: grava um snapshot minimo (exposicao zero salvo override)."""
    return repo.record_snapshot(
        day=day,
        equity=equity,
        cash=kw.get("cash", equity),
        long_market_value=kw.get("long_market_value", 0),
        gross_exposure=kw.get("gross_exposure", 0),
        net_exposure=kw.get("net_exposure", 0),
        regime=kw.get("regime", "unknown"),
        bench_spy=kw.get("bench_spy"),
        bench_6040=kw.get("bench_6040"),
        source=kw.get("source", "eod"),
    )


# --------------------------------------------------------------------------- #
# 1. Append-only / idempotencia diaria
# --------------------------------------------------------------------------- #
def test_idempotente_no_mesmo_dia_nao_duplica(repo):
    _snap(repo, "2026-01-02", 100000)
    _snap(repo, "2026-01-02", 100500)  # mesma data: UPSERT, nao nova linha
    rows = repo.all_rows()
    assert len(rows) == 1
    assert float(rows[0]["equity"]) == 100500.0  # ficou com o ultimo valor do dia


def test_date_e_unico(repo):
    _snap(repo, "2026-01-02", 100000)
    with pytest.raises(sqlite3.IntegrityError):
        # insercao crua de uma 2a linha com a MESMA date deve violar o UNIQUE.
        repo._conn.execute(
            "INSERT INTO nav_history (ts,date,equity,cash,long_market_value,"
            "gross_exposure,net_exposure,regime,source,prev_hash,row_hash) "
            "VALUES ('t','2026-01-02','1','1','0','0','0','x','eod','','h')"
        )


def test_upsert_do_dia_nao_altera_dias_anteriores(repo):
    a = _snap(repo, "2026-01-02", 100000)
    _snap(repo, "2026-01-03", 101000)
    # re-captura do dia 03 (ainda o ultimo): permitido, e nao toca o dia 02.
    _snap(repo, "2026-01-03", 101234)
    again_a = repo.get_by_date("2026-01-02")
    assert again_a["equity"] == a["equity"]
    assert again_a["row_hash"] == a["row_hash"]  # dia 02 intacto
    assert float(repo.get_by_date("2026-01-03")["equity"]) == 101234.0


# --------------------------------------------------------------------------- #
# 2. Imutabilidade (anti-reescrita de historico)
# --------------------------------------------------------------------------- #
def test_reescrever_dia_selado_falha(repo):
    _snap(repo, "2026-01-02", 100000)
    _snap(repo, "2026-01-03", 101000)  # sela o dia 02
    with pytest.raises(ImmutableHistoryError):
        _snap(repo, "2026-01-02", 999999)  # tentar reescrever o dia selado


def test_backfill_no_passado_falha(repo):
    _snap(repo, "2026-01-10", 100000)
    with pytest.raises(ImmutableHistoryError):
        _snap(repo, "2026-01-05", 50000)  # data anterior nunca vista => backfill


# --------------------------------------------------------------------------- #
# 3. Cadeia de hash
# --------------------------------------------------------------------------- #
def test_genesis_e_encadeamento(repo):
    a = _snap(repo, "2026-01-02", 100000)
    b = _snap(repo, "2026-01-03", 101000)
    assert a["prev_hash"] == GENESIS_PREV_HASH
    assert b["prev_hash"] == a["row_hash"]  # elo: prev de b == hash de a


def test_cadeia_integra_passa_verificacao(repo):
    for i, eq in enumerate([100000, 101000, 99000, 102000, 103500]):
        _snap(repo, f"2026-02-{i+2:02d}", eq, bench_spy=400 + i, bench_6040=1 + i * 0.01)
    res = verify(repo=repo)
    assert res.ok
    assert res.n_rows == 5


def test_adulteracao_de_linha_e_detectada(repo):
    _snap(repo, "2026-03-02", 100000)
    _snap(repo, "2026-03-03", 101000)
    _snap(repo, "2026-03-04", 102000)
    # adultera a equity de uma linha antiga DIRETO no banco (sem recomputar hash).
    repo._conn.execute(
        "UPDATE nav_history SET equity = '500000' WHERE date = '2026-03-03'"
    )
    repo._conn.commit()
    res = verify(repo=repo)
    assert not res.ok
    first = res.breaks[0]
    assert first.date == "2026-03-03"
    assert first.reason == "row_hash"  # o hash da propria linha nao bate mais


def test_remocao_de_linha_quebra_o_elo(repo):
    _snap(repo, "2026-04-02", 100000)
    _snap(repo, "2026-04-03", 101000)
    _snap(repo, "2026-04-04", 102000)
    # remove a linha do meio: o prev_hash da seguinte deixa de bater.
    repo._conn.execute("DELETE FROM nav_history WHERE date = '2026-04-03'")
    repo._conn.commit()
    res = verify(repo=repo)
    assert not res.ok
    assert res.breaks[0].date == "2026-04-04"
    assert res.breaks[0].reason == "prev_hash"


def test_hash_e_reproduzivel_por_terceiro(repo):
    """Um auditor recomputa o hash so com os campos crus e o prev_hash."""
    row = _snap(repo, "2026-05-02", 100000, bench_spy=410, bench_6040=1.02)
    fields = {
        "date": row["date"], "equity": row["equity"], "cash": row["cash"],
        "long_market_value": row["long_market_value"],
        "gross_exposure": row["gross_exposure"], "net_exposure": row["net_exposure"],
        "regime": row["regime"], "bench_spy": row["bench_spy"],
        "bench_6040": row["bench_6040"], "source": row["source"],
    }
    assert compute_row_hash(fields, row["prev_hash"]) == row["row_hash"]


# --------------------------------------------------------------------------- #
# 4. Metricas sobre serie conhecida
# --------------------------------------------------------------------------- #
def _business_dates(n: int, start="2024-01-02") -> list[str]:
    """n datas de pregao (dias uteis) consecutivas, como 'YYYY-MM-DD'."""
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


def test_cagr_sobre_serie_conhecida():
    """1 ano de pregao (252 dias), dobrando: NAV 100k -> 200k => CAGR ~ +100%."""
    n = 253  # 252 intervalos
    equity = np.geomspace(100000, 200000, n)
    dates = _business_dates(n)
    series = NavSeries(
        dates=dates, equity=equity,
        bench_spy=np.full(n, np.nan), bench_6040=np.full(n, np.nan),
    )
    m = compute(series)
    assert m.total_return == pytest.approx(1.0, abs=1e-6)
    assert m.cagr == pytest.approx(1.0, abs=0.02)  # ~100% a.a.
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)  # monotonica: sem DD


def test_maxdd_e_datas_sobre_serie_conhecida():
    """Serie com queda controlada: 100 -> 120 -> 90 -> 110. MaxDD = 90/120-1 = -25%."""
    dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    series = NavSeries(dates=dates, equity=equity,
                       bench_spy=np.full(4, np.nan), bench_6040=np.full(4, np.nan))
    m = compute(series)
    assert m.max_drawdown == pytest.approx(-0.25, abs=1e-9)
    assert m.max_dd_peak_date == "2026-01-03"   # pico em 120
    assert m.max_dd_trough_date == "2026-01-04"  # vale em 90
    assert m.drawdowns[0].recovery is None or m.drawdowns[0].depth == pytest.approx(-0.25)


def test_retornos_mensais_conhecidos():
    """2 meses: jan fecha em 110 (de 100 = +10%), fev fecha em 99 (de 110 = -10%)."""
    dates = ["2026-01-15", "2026-01-31", "2026-02-15", "2026-02-28"]
    equity = np.array([100.0, 110.0, 105.0, 99.0])
    series = NavSeries(dates=dates, equity=equity,
                       bench_spy=np.full(4, np.nan), bench_6040=np.full(4, np.nan))
    m = compute(series)
    assert m.monthly_returns[2026][1] == pytest.approx(0.10, abs=1e-9)   # jan
    assert m.monthly_returns[2026][2] == pytest.approx(99 / 110 - 1, abs=1e-9)  # fev
    assert m.worst_month == pytest.approx(99 / 110 - 1, abs=1e-9)
    assert m.pct_positive_months == pytest.approx(0.5, abs=1e-9)


def test_excess_e_beta_vs_spy():
    """Estrategia anda junto com SPY -> beta ~1, correlacao ~1, TE pequeno."""
    n = 60
    rng = np.random.default_rng(7)
    spy_rets = rng.normal(0.0005, 0.01, n - 1)
    spy = 400 * np.cumprod(np.concatenate([[1.0], 1 + spy_rets]))
    equity = 100000 * np.cumprod(np.concatenate([[1.0], 1 + spy_rets]))  # mesmos retornos
    dates = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
    series = NavSeries(dates=dates, equity=equity, bench_spy=spy,
                       bench_6040=np.full(n, np.nan))
    m = compute(series, benchmark="spy")
    assert m.beta_vs_spy == pytest.approx(1.0, abs=0.05)
    assert m.corr_vs_spy == pytest.approx(1.0, abs=0.01)
    assert m.tracking_error == pytest.approx(0.0, abs=1e-6)
    assert m.bench_cagr is not None and m.excess_cagr is not None


def test_serie_vazia_nao_quebra():
    m = compute(NavSeries(dates=[], equity=np.array([]),
                          bench_spy=np.array([]), bench_6040=np.array([])))
    assert m.n_days == 0 and m.cagr == 0.0


# --------------------------------------------------------------------------- #
# 5. Benchmark 60/40 incremental
# --------------------------------------------------------------------------- #
def test_6040_marca_a_mercado_e_pondera():
    """No dia 1, 60/40 a base 1.0; subir SPY +10% (peso 0.6) e IEF flat sobe o NAV
    ~ +6%."""
    p0 = {benchmarks.SPY: Decimal("400"), benchmarks.IEF: Decimal("100")}
    s0 = benchmarks.step(None, "2026-01-02", p0)
    assert s0.nav == pytest.approx(Decimal("1"))
    # SPY +10%, IEF inalterado, MESMO mes (sem rebalance) -> marca a mercado.
    p1 = {benchmarks.SPY: Decimal("440"), benchmarks.IEF: Decimal("100")}
    s1 = benchmarks.step(s0, "2026-01-03", p1)
    assert float(s1.nav) == pytest.approx(1.06, abs=1e-6)  # 0.6*1.10 + 0.4*1.0


def test_6040_rebalanceia_na_virada_de_mes():
    p = {benchmarks.SPY: Decimal("400"), benchmarks.IEF: Decimal("100")}
    s = benchmarks.step(None, "2026-01-02", p)
    last_month = s.last_month
    s2 = benchmarks.step(s, "2026-02-02", p)  # virou o mes
    assert s2.last_month == "2026-02" != last_month
    # apos rebalance, os pesos voltam a 60/40 (units recompostas ao NAV atual).
    val_spy = s2.units[benchmarks.SPY] * p[benchmarks.SPY]
    assert float(val_spy / s2.nav) == pytest.approx(0.60, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. Captura EOD broker-agnostica + idempotente
# --------------------------------------------------------------------------- #
def test_capture_eod_grava_equity_do_broker(tmp_path):
    broker = FakeBroker(cash=Decimal("100000"), prices={
        "AAPL": Decimal("200"), "SPY": Decimal("400"), "IEF": Decimal("100"),
    })
    broker.seed_position("AAPL", Decimal("100"), Decimal("150"))  # 100*200 = 20.000
    repo = NavHistoryRepo(db_path=tmp_path / "n.sqlite")
    try:
        row = capture_eod(broker, repo=repo, day="2026-06-15")
        # equity do broker = cash 100k + 100*200 = 120k.
        assert float(row["equity"]) == pytest.approx(120000.0)
        assert float(row["long_market_value"]) == pytest.approx(20000.0)
        # gross = 20.000 / 120.000.
        assert float(row["gross_exposure"]) == pytest.approx(20000 / 120000, abs=1e-9)
        assert row["bench_spy"] == "400"
        assert row["bench_6040"] is not None  # 60/40 capturado no mesmo dia
    finally:
        repo.close()


def test_capture_eod_idempotente_no_dia(tmp_path):
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("400"), "IEF": Decimal("100")})
    repo = NavHistoryRepo(db_path=tmp_path / "n.sqlite")
    try:
        capture_eod(broker, repo=repo, day="2026-06-15")
        capture_eod(broker, repo=repo, day="2026-06-15")  # 2a vez no mesmo dia
        assert repo.count() == 1  # nao duplicou
    finally:
        repo.close()


def test_capture_eod_sobrevive_a_restart_com_state(tmp_path):
    """Com StateRepository, o NAV do 60/40 avanca dia a dia mesmo recriando o repo
    (estado persistido) — e a captura de cada dia gera 1 linha (sem duplicar)."""
    from data.db import Database
    from data.state_repo import StateRepository

    db = Database(tmp_path / "trading.sqlite")
    state = StateRepository(db)
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("400"), "IEF": Decimal("100")})
    try:
        # dia 1
        repo = NavHistoryRepo(connection=db.conn)
        capture_eod(broker, repo=repo, state=state, day="2026-06-15")
        # "restart": SPY sobe; novo repo (mesma conexao), proximo dia.
        broker.set_price("SPY", Decimal("440"))
        capture_eod(broker, repo=NavHistoryRepo(connection=db.conn), state=state, day="2026-06-16")
        rows = NavHistoryRepo(connection=db.conn).all_rows()
        assert len(rows) == 2
        nav1 = float(rows[0]["bench_6040"])
        nav2 = float(rows[1]["bench_6040"])
        assert nav2 > nav1  # SPY subiu => NAV do 60/40 subiu (estado persistido)
        # integridade preservada apos multiplos dias/conexoes.
        assert verify(repo=NavHistoryRepo(connection=db.conn)).ok
    finally:
        db.close()


def test_load_series_so_pega_eod(tmp_path):
    repo = NavHistoryRepo(db_path=tmp_path / "n.sqlite")
    try:
        _snap(repo, "2026-06-15", 100000, source="eod")
        # snapshot intraday do MESMO dia nao pode existir como 2a linha (UNIQUE date),
        # entao testamos com um dia so intraday separado.
        _snap(repo, "2026-06-16", 100500, source="intraday")
        s = load_series(repo=repo, source="eod")
        assert len(s) == 1 and s.dates == ["2026-06-15"]
    finally:
        repo.close()


def test_report_inclui_since_inception(tmp_path):
    from reporting.track_record import generate_report

    repo = NavHistoryRepo(db_path=tmp_path / "n.sqlite")
    try:
        _snap(repo, "2026-06-15", 100000, bench_spy=400, bench_6040=1.0)
        _snap(repo, "2026-06-16", 101000, bench_spy=404, bench_6040=1.01)
        report = generate_report(repo=repo)
        assert "SINCE-INCEPTION" in report
        assert "2026-06-15" in report
        assert "Rentabilidade passada" in report  # disclaimer presente
    finally:
        repo.close()
