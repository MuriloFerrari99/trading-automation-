"""Metricas since-inception sobre o NAV REAL + relatorio de track record.

O que um alocador de AUM quer ver, calculado sobre a serie GRAVADA em
nav_history (nao um backtest): CAGR, vol anualizada, Sharpe, Sortino, MaxDD
(pico-a-vale, com datas), Calmar, retornos MENSAIS (tabela mes x ano), melhor/
pior mes, % de meses positivos, beta/correlacao vs SPY, excess return e tracking
error vs benchmark, e a tabela dos maiores drawdowns (inicio, vale, recuperacao,
duracao).

ANTI-CHERRY-PICKING: a janela SINCE-INCEPTION (do 1o snapshot ao ultimo) esta
SEMPRE no output. Janelas customizadas, se existirem, sao adendo — nunca
substituem a since-inception.

REUSO (nao reescrever): as formulas estatisticas vem de simulation/:
  - simulation.metrics.sharpe / sortino / max_drawdown (testadas).
A unica coisa que este modulo adiciona e a leitura do nav_history, a agregacao
mensal e a comparacao com benchmark — tudo determinista (G-TR1: reprodutivel por
um terceiro rodando o mesmo script).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# REUSO das formulas ja testadas (doc 06 §3). NAO reimplementar.
from simulation.metrics import max_drawdown, sharpe, sortino
from reporting.nav_repo import DEFAULT_DB_PATH, NavHistoryRepo

TRADING_DAYS = 252

DISCLAIMER = (
    "Rentabilidade passada nao representa garantia de resultados futuros. "
    "Track record de simulacao/paper trading — nao houve gestao de recursos de "
    "terceiros."
)


# --------------------------------------------------------------------------- #
# Estruturas de resultado
# --------------------------------------------------------------------------- #
@dataclass
class DrawdownEpisode:
    start: str          # data do pico anterior (inicio da queda)
    trough: str         # data do vale (fundo)
    recovery: str | None  # data de recuperacao do pico; None se ainda submerso
    depth: float        # profundidade (<=0)
    length_days: int    # do pico ao vale
    recovery_days: int | None  # do vale a recuperacao (None se nao recuperou)


@dataclass
class TrackRecordMetrics:
    n_days: int
    start: str
    end: str
    start_equity: float
    end_equity: float
    total_return: float
    cagr: float
    vol_annual: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_dd_peak_date: str | None
    max_dd_trough_date: str | None
    calmar: float
    best_month: float
    worst_month: float
    pct_positive_months: float
    # vs benchmark (podem ser None se o benchmark nao foi capturado):
    bench_cagr: float | None = None
    bench_max_drawdown: float | None = None
    excess_cagr: float | None = None
    beta_vs_spy: float | None = None
    corr_vs_spy: float | None = None
    tracking_error: float | None = None
    monthly_returns: dict = field(default_factory=dict)  # {year: {month: ret}}
    drawdowns: list = field(default_factory=list)        # list[DrawdownEpisode]

    def as_dict(self) -> dict:
        d = {
            "n_days": self.n_days,
            "start": self.start,
            "end": self.end,
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "vol_annual": self.vol_annual,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "max_dd_peak_date": self.max_dd_peak_date,
            "max_dd_trough_date": self.max_dd_trough_date,
            "calmar": self.calmar,
            "best_month": self.best_month,
            "worst_month": self.worst_month,
            "pct_positive_months": self.pct_positive_months,
            "bench_cagr": self.bench_cagr,
            "bench_max_drawdown": self.bench_max_drawdown,
            "excess_cagr": self.excess_cagr,
            "beta_vs_spy": self.beta_vs_spy,
            "corr_vs_spy": self.corr_vs_spy,
            "tracking_error": self.tracking_error,
        }
        return d


# --------------------------------------------------------------------------- #
# Carregamento da serie
# --------------------------------------------------------------------------- #
@dataclass
class NavSeries:
    dates: list[str]
    equity: np.ndarray
    bench_spy: np.ndarray        # pode conter nan
    bench_6040: np.ndarray       # pode conter nan

    def __len__(self) -> int:
        return len(self.dates)


def load_series(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    source: str = "eod",
    repo: NavHistoryRepo | None = None,
) -> NavSeries:
    """Le nav_history em ordem cronologica. So conta o snapshot oficial (`eod`).

    `repo` permite injetar um repo ja aberto (testes/dashboard); senao abre um
    proprio no db_path.
    """
    own = repo is None
    r = repo or NavHistoryRepo(db_path=db_path)
    try:
        rows = [row for row in r.all_rows() if row.get("source", "eod") == source]
    finally:
        if own:
            r.close()
    dates = [row["date"] for row in rows]
    equity = np.array([_f(row["equity"]) for row in rows], dtype=float)
    bench_spy = np.array([_f(row.get("bench_spy")) for row in rows], dtype=float)
    bench_6040 = np.array([_f(row.get("bench_6040")) for row in rows], dtype=float)
    return NavSeries(dates=dates, equity=equity, bench_spy=bench_spy, bench_6040=bench_6040)


def _f(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Calculo das metricas
# --------------------------------------------------------------------------- #
def _returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=float)
    prev = equity[:-1]
    out = np.where(prev != 0, np.diff(equity) / prev, 0.0)
    return out


def _cagr(equity: np.ndarray, n_days: int) -> float:
    if equity.size < 2 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    years = max(n_days / TRADING_DAYS, 1e-9)
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def _drawdown_episodes(dates: list[str], equity: np.ndarray, top: int = 5) -> list[DrawdownEpisode]:
    """Extrai os episodios de drawdown (pico->vale->recuperacao), ordenados pela
    profundidade. Usa a mesma definicao pico-a-vale do simulation.metrics."""
    if equity.size < 2:
        return []
    episodes: list[DrawdownEpisode] = []
    peak = equity[0]
    peak_i = 0
    in_dd = False
    trough = equity[0]
    trough_i = 0
    for i in range(1, equity.size):
        if equity[i] >= peak:
            # recuperou (ou novo pico): fecha episodio aberto, se houver.
            if in_dd:
                episodes.append(
                    DrawdownEpisode(
                        start=dates[peak_i],
                        trough=dates[trough_i],
                        recovery=dates[i],
                        depth=float(trough / peak - 1.0),
                        length_days=trough_i - peak_i,
                        recovery_days=i - trough_i,
                    )
                )
                in_dd = False
            peak = equity[i]
            peak_i = i
        else:
            if not in_dd:
                in_dd = True
                trough = equity[i]
                trough_i = i
            elif equity[i] < trough:
                trough = equity[i]
                trough_i = i
    # episodio ainda submerso no fim da serie (sem recuperacao).
    if in_dd:
        episodes.append(
            DrawdownEpisode(
                start=dates[peak_i],
                trough=dates[trough_i],
                recovery=None,
                depth=float(trough / peak - 1.0),
                length_days=trough_i - peak_i,
                recovery_days=None,
            )
        )
    episodes.sort(key=lambda e: e.depth)  # mais profundo (mais negativo) primeiro
    return episodes[:top]


def _year_month(d: str) -> tuple[int, int] | None:
    """Extrai (ano, mes) de 'YYYY-MM-DD'. None se nao parsear (defensivo)."""
    try:
        return int(d[:4]), int(d[5:7])
    except (ValueError, TypeError, IndexError):
        return None


def _monthly_returns(dates: list[str], equity: np.ndarray) -> dict:
    """Retorno por mes-calendario: ultimo NAV do mes vs ultimo NAV do mes anterior.

    Estrutura: {ano(int): {mes(int): retorno(float)}}. O 1o mes usa o 1o NAV como
    base (retorno parcial do mes inicial)."""
    if equity.size < 1:
        return {}
    # ultimo equity de cada mes (ordem cronologica garantida pelo load_series).
    last_of_month: dict[tuple[int, int], float] = {}
    order: list[tuple[int, int]] = []
    for d, e in zip(dates, equity):
        key = _year_month(d)
        if key is None:
            continue  # data fora do formato YYYY-MM-DD (defensivo) -> ignora
        if key not in last_of_month:
            order.append(key)
        last_of_month[key] = float(e)

    out: dict[int, dict[int, float]] = {}
    prev_close = float(equity[0])
    for key in order:
        y, m = key
        close = last_of_month[key]
        ret = (close / prev_close - 1.0) if prev_close > 0 else 0.0
        out.setdefault(y, {})[m] = ret
        prev_close = close
    return out


def _beta_corr_te(strat_ret: np.ndarray, bench_ret: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Beta, correlacao e tracking error (anualizado) da estrategia vs benchmark.

    Alinha pelos pontos em que AMBOS tem retorno valido (benchmark pode ter nan).
    """
    if strat_ret.size == 0 or bench_ret.size == 0 or strat_ret.size != bench_ret.size:
        return None, None, None
    mask = np.isfinite(strat_ret) & np.isfinite(bench_ret)
    s = strat_ret[mask]
    b = bench_ret[mask]
    if s.size < 2:
        return None, None, None
    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    sd_s, sd_b = s.std(ddof=1), b.std(ddof=1)
    corr = float(np.corrcoef(s, b)[0, 1]) if sd_s > 0 and sd_b > 0 else None
    te = float((s - b).std(ddof=1) * np.sqrt(TRADING_DAYS))
    return beta, corr, te


def _spy_returns_from_prices(bench_spy: np.ndarray) -> np.ndarray:
    """Retornos diarios de SPY a partir dos precos gravados (nan-safe)."""
    if bench_spy.size < 2:
        return np.array([], dtype=float)
    prev = bench_spy[:-1]
    cur = bench_spy[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where((prev > 0) & np.isfinite(prev) & np.isfinite(cur), cur / prev - 1.0, np.nan)
    return out


def compute(series: NavSeries, *, benchmark: str = "spy") -> TrackRecordMetrics:
    """Calcula TODAS as metricas since-inception sobre a serie de NAV.

    `benchmark`: 'spy' (preco total-return) ou '6040' (NAV sintetico) para o
    comparativo de CAGR/MaxDD/excess. Beta/corr/TE sao sempre vs SPY (o mercado).
    """
    n = len(series)
    if n == 0:
        return TrackRecordMetrics(
            n_days=0, start="-", end="-", start_equity=0.0, end_equity=0.0,
            total_return=0.0, cagr=0.0, vol_annual=0.0, sharpe=0.0, sortino=0.0,
            max_drawdown=0.0, max_dd_peak_date=None, max_dd_trough_date=None,
            calmar=0.0, best_month=0.0, worst_month=0.0, pct_positive_months=0.0,
        )

    equity = series.equity
    rets = _returns(equity)
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
    cagr = _cagr(equity, n)
    vol_annual = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if rets.size >= 2 else 0.0
    shp = sharpe(rets, periods=TRADING_DAYS)
    srt = sortino(rets, periods=TRADING_DAYS)
    mdd = max_drawdown(equity)

    # datas do MaxDD (pico-a-vale).
    episodes = _drawdown_episodes(series.dates, equity, top=5)
    deepest = episodes[0] if episodes else None
    calmar = float(cagr / abs(mdd)) if mdd < 0 else (float("inf") if cagr > 0 else 0.0)

    # mensais.
    monthly = _monthly_returns(series.dates, equity)
    all_months = [m for year in monthly.values() for m in year.values()]
    best_month = max(all_months) if all_months else 0.0
    worst_month = min(all_months) if all_months else 0.0
    pct_pos = (sum(1 for m in all_months if m > 0) / len(all_months)) if all_months else 0.0

    # vs benchmark.
    bench_series = series.bench_spy if benchmark == "spy" else series.bench_6040
    bench_cagr = bench_mdd = excess = None
    if bench_series.size and np.isfinite(bench_series).sum() >= 2:
        valid = bench_series[np.isfinite(bench_series)]
        bench_cagr = _cagr(valid, valid.size)
        bench_mdd = max_drawdown(valid)
        excess = cagr - bench_cagr

    # beta/corr/TE sempre vs SPY (retornos derivados dos precos gravados).
    spy_ret = _spy_returns_from_prices(series.bench_spy)
    beta = corr = te = None
    if spy_ret.size == rets.size and rets.size > 0:
        beta, corr, te = _beta_corr_te(rets, spy_ret)

    return TrackRecordMetrics(
        n_days=n,
        start=series.dates[0],
        end=series.dates[-1],
        start_equity=float(equity[0]),
        end_equity=float(equity[-1]),
        total_return=total_return,
        cagr=cagr,
        vol_annual=vol_annual,
        sharpe=shp,
        sortino=srt,
        max_drawdown=mdd,
        max_dd_peak_date=deepest.start if deepest else None,
        max_dd_trough_date=deepest.trough if deepest else None,
        calmar=calmar,
        best_month=best_month,
        worst_month=worst_month,
        pct_positive_months=pct_pos,
        bench_cagr=bench_cagr,
        bench_max_drawdown=bench_mdd,
        excess_cagr=excess,
        beta_vs_spy=beta,
        corr_vs_spy=corr,
        tracking_error=te,
        monthly_returns=monthly,
        drawdowns=episodes,
    )


# --------------------------------------------------------------------------- #
# Relatorio de texto (estilo *_verdict.txt)
# --------------------------------------------------------------------------- #
_MONTHS = ["", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
           "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _pct(x: float | None) -> str:
    return "   -  " if x is None else f"{x * 100:+.1f}%"


def _num(x: float | None, fmt: str = "{:.2f}") -> str:
    return "  -  " if x is None else fmt.format(x)


def _monthly_table(monthly: dict) -> str:
    if not monthly:
        return "  (sem meses fechados ainda)"
    years = sorted(monthly.keys())
    head = f"  {'Ano':<6}" + "".join(f"{_MONTHS[m]:>7}" for m in range(1, 13)) + f"{'Ano*':>9}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for y in years:
        row = monthly[y]
        cells = "".join(_pct(row.get(m)).rjust(7) for m in range(1, 13))
        # retorno do ano = produto dos meses presentes.
        yr = 1.0
        for m in range(1, 13):
            if m in row:
                yr *= (1.0 + row[m])
        lines.append(f"  {y:<6}{cells}{_pct(yr - 1.0):>9}")
    lines.append("  * Ano = composicao dos meses gravados (o 1o mes pode ser parcial).")
    return "\n".join(lines)


def _drawdown_table(eps: list[DrawdownEpisode]) -> str:
    if not eps:
        return "  (sem drawdowns registrados)"
    head = (
        f"  {'#':>2} | {'profund.':>9} | {'inicio (pico)':>13} | {'vale':>10} | "
        f"{'recuperacao':>11} | {'pico->vale':>10} | {'vale->rec':>9}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for i, e in enumerate(eps, 1):
        rec = e.recovery if e.recovery else "submerso"
        rec_days = "-" if e.recovery_days is None else str(e.recovery_days)
        lines.append(
            f"  {i:>2} | {_pct(e.depth):>9} | {e.start:>13} | {e.trough:>10} | "
            f"{rec:>11} | {e.length_days:>7}d | {rec_days:>7}d"
        )
    return "\n".join(lines)


def build_report(m: TrackRecordMetrics, *, benchmark: str = "spy") -> str:
    bench_label = "SPY (buy&hold)" if benchmark == "spy" else "60/40 (SPY/IEF)"
    L: list[str] = []
    L.append("=" * 92)
    L.append("TRACK RECORD — curva de equity (NAV) mark-to-market, SINCE-INCEPTION, AUDITAVEL")
    L.append("=" * 92)
    L.append(DISCLAIMER)
    L.append("")
    if m.n_days == 0:
        L.append("  (nav_history vazio — nenhum snapshot EOD gravado ainda.)")
        L.append("  Plugue a captura diaria (reporting.capture.capture_eod) no fim do pregao.")
        return "\n".join(L)

    L.append(f"  Janela SINCE-INCEPTION: {m.start} -> {m.end}  ({m.n_days} dias de NAV)")
    L.append(f"  NAV inicial: {m.start_equity:,.2f}   NAV final: {m.end_equity:,.2f}")
    L.append("")
    L.append("  --- METRICAS SINCE-INCEPTION ---------------------------------------------")
    L.append(f"    Retorno total .......... {_pct(m.total_return)}")
    L.append(f"    CAGR ................... {_pct(m.cagr)}")
    L.append(f"    Vol anualizada ........ {_pct(m.vol_annual)}")
    L.append(f"    Sharpe ................ {_num(m.sharpe)}")
    L.append(f"    Sortino ............... {_num(m.sortino)}")
    L.append(f"    MaxDD ................. {_pct(m.max_drawdown)}"
             + (f"  (pico {m.max_dd_peak_date} -> vale {m.max_dd_trough_date})"
                if m.max_dd_peak_date else ""))
    L.append(f"    Calmar (CAGR/|MaxDD|) . {_num(m.calmar)}")
    L.append(f"    Melhor mes ............ {_pct(m.best_month)}")
    L.append(f"    Pior mes .............. {_pct(m.worst_month)}")
    L.append(f"    % meses positivos ..... {_pct(m.pct_positive_months)}")
    L.append("")
    L.append(f"  --- vs BENCHMARK: {bench_label} -----------------------------------")
    L.append(f"    CAGR estrategia ....... {_pct(m.cagr)}   |  CAGR benchmark ... {_pct(m.bench_cagr)}")
    L.append(f"    MaxDD estrategia ...... {_pct(m.max_drawdown)}   |  MaxDD benchmark .. {_pct(m.bench_max_drawdown)}")
    L.append(f"    Excess CAGR (estrat - bench) .... {_pct(m.excess_cagr)}")
    L.append(f"    Beta vs SPY ........... {_num(m.beta_vs_spy)}   |  Correlacao vs SPY . {_num(m.corr_vs_spy)}")
    L.append(f"    Tracking error (a.a.) . {_pct(m.tracking_error)}")
    L.append("")
    L.append("  --- RETORNOS MENSAIS (mes x ano) -----------------------------------------")
    L.append(_monthly_table(m.monthly_returns))
    L.append("")
    L.append("  --- MAIORES DRAWDOWNS (top 5) --------------------------------------------")
    L.append(_drawdown_table(m.drawdowns))
    L.append("")
    L.append("  INTEGRIDADE: serie append-only com cadeia de hash (reporting/nav_repo.py).")
    L.append("  Verifique com:  uv run python -m reporting.verify_chain")
    L.append("")
    return "\n".join(L)


def generate_report(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    benchmark: str = "spy",
    repo: NavHistoryRepo | None = None,
) -> str:
    series = load_series(db_path, repo=repo)
    metrics = compute(series, benchmark=benchmark)
    return build_report(metrics, benchmark=benchmark)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relatorio de track record since-inception sobre nav_history."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="caminho do SQLite")
    parser.add_argument("--benchmark", choices=["spy", "6040"], default="spy")
    parser.add_argument("--report-file", default="data/track_record.txt")
    parser.add_argument("--report", action="store_true", help="(compat) gera o relatorio")
    args = parser.parse_args(argv)

    report = generate_report(args.db, benchmark=args.benchmark)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
