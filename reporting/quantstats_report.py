"""Tearsheet QuantStats sobre o NAV REAL gravado em nav_history.

QuantStats gera num passo o relatorio que um alocador de AUM espera (Sharpe,
Sortino, Calmar, drawdowns, retornos mensais, comparacao com benchmark) em HTML
pronto para enviar. Aqui ele le a MESMA serie auditavel que o track_record usa
(reporting.load_series) — nao um backtest, nao uma serie paralela.

Determinista e sem efeito colateral de mercado: so LE o nav_history e ESCREVE um
HTML/CSV. Nada de ordem, nada externo.

CLI:
    uv run python -m reporting.quantstats_report                  # HTML padrao
    uv run python -m reporting.quantstats_report --no-benchmark   # sem SPY
    uv run python -m reporting.quantstats_report -o reports/x.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.nav_repo import DEFAULT_DB_PATH, NavHistoryRepo
from reporting.track_record import NavSeries, load_series

DEFAULT_OUTPUT = Path("reports/quantstats_tearsheet.html")
DISCLAIMER = (
    "Track record de simulacao/paper trading — sem gestao de recursos de terceiros. "
    "Rentabilidade passada nao garante resultado futuro."
)


def _to_returns(dates: list[str], equity: np.ndarray) -> pd.Series:
    """Curva de equity -> retornos diarios, indexados por data (QuantStats-ready)."""
    idx = pd.to_datetime(pd.Index(dates))
    nav = pd.Series(np.asarray(equity, dtype=float), index=idx).sort_index()
    nav = nav[nav > 0]  # equity invalida (<=0) nao gera retorno significativo
    return nav.pct_change().dropna()


def nav_to_returns(series: NavSeries) -> pd.Series:
    return _to_returns(series.dates, series.equity)


def _benchmark_returns(series: NavSeries) -> pd.Series | None:
    bench = np.asarray(series.bench_spy, dtype=float)
    if bench.size == 0 or np.isnan(bench).all() or np.nansum(bench) == 0:
        return None
    rets = _to_returns(series.dates, np.nan_to_num(bench, nan=0.0))
    return rets if not rets.empty else None


def metrics_table(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Tabela de metricas QuantStats (Sharpe, Sortino, Calmar, MaxDD, ...)."""
    import quantstats as qs

    return qs.reports.metrics(
        returns, benchmark=benchmark, mode="full", display=False
    )


def generate_report(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    repo: NavHistoryRepo | None = None,
    benchmark: bool = True,
    title: str = "Trading Automation — Track Record",
) -> Path | None:
    """Le o nav_history e escreve um tearsheet HTML. Retorna o caminho (ou None)."""
    import quantstats as qs

    series = load_series(db_path=db_path, repo=repo)
    returns = nav_to_returns(series)
    if returns.shape[0] < 2:
        print(f"[quantstats] nav_history com poucos pontos ({returns.shape[0]}); pulei.")
        return None

    bench = _benchmark_returns(series) if benchmark else None
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        returns,
        benchmark=bench,
        output=str(out),
        title=title,
        download_filename=str(out),
    )
    print(f"[quantstats] tearsheet -> {out}  ({returns.shape[0]} dias, benchmark={'SPY' if bench is not None else 'nenhum'})")
    print(DISCLAIMER)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tearsheet QuantStats do NAV real.")
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--no-benchmark", action="store_true")
    args = ap.parse_args()
    generate_report(args.output, db_path=args.db, benchmark=not args.no_benchmark)


if __name__ == "__main__":
    main()
