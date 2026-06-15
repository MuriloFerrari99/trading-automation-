"""Sweep Monte Carlo sobre dados reais: simbolos x janelas rolantes.

Cada (simbolo, janela, estrategia) e um backtest real, rotulado por regime.
Milhares de janelas dao a distribuicao de desempenho e o breakdown por
estrategia@regime — que e onde se ve a acuracia do modelo e os maiores
potenciais de melhora (ex.: ladder sangra em downtrend; trailing sofre
whipsaw em range).
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from simulation.engine import BacktestResult, run_backtest

logger = logging.getLogger("simulation.montecarlo")

STRATEGIES = ("trailing_stop", "ladder_buys")


def make_windows(n: int, window_len: int, step: int) -> list[tuple[int, int]]:
    """Janelas rolantes [start, end) de tamanho window_len, passo step."""
    out = []
    start = 0
    while start + window_len <= n:
        out.append((start, start + window_len))
        start += step
    return out


def run_sweep(
    data: dict[str, dict[str, list[float]]],
    *,
    window_len: int = 160,
    step: int = 10,
    strategies: tuple[str, ...] = STRATEGIES,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    max_runs: int | None = None,
) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    for symbol, ohlc in data.items():
        n = len(ohlc["close"])
        for (a, b) in make_windows(n, window_len, step):
            window = {k: v[a:b] for k, v in ohlc.items()}
            for strat in strategies:
                r = run_backtest(
                    symbol, window, strat,
                    commission_bps=commission_bps, slippage_bps=slippage_bps,
                )
                if r is not None:
                    results.append(r)
                    if max_runs and len(results) >= max_runs:
                        return results
    return results


@dataclass
class GroupSummary:
    label: str
    n: int = 0
    pct_profitable: float = 0.0
    mean_return: float = 0.0
    median_return: float = 0.0
    mean_sharpe: float = 0.0
    mean_sortino: float = 0.0
    mean_max_dd: float = 0.0
    worst_max_dd: float = 0.0
    mean_win_rate: float = 0.0
    _returns: list[float] = field(default_factory=list, repr=False)


def _summarize_group(label: str, items: list[BacktestResult]) -> GroupSummary:
    rets = [r.metrics.total_return for r in items]
    dds = [r.metrics.max_drawdown for r in items]
    return GroupSummary(
        label=label,
        n=len(items),
        pct_profitable=sum(1 for x in rets if x > 0) / len(rets) if rets else 0.0,
        mean_return=statistics.fmean(rets) if rets else 0.0,
        median_return=statistics.median(rets) if rets else 0.0,
        mean_sharpe=statistics.fmean([r.metrics.sharpe for r in items]) if items else 0.0,
        mean_sortino=statistics.fmean([r.metrics.sortino for r in items]) if items else 0.0,
        mean_max_dd=statistics.fmean(dds) if dds else 0.0,
        worst_max_dd=min(dds) if dds else 0.0,
        mean_win_rate=statistics.fmean([r.metrics.win_rate for r in items]) if items else 0.0,
    )


@dataclass
class SweepReport:
    total_runs: int
    overall: GroupSummary
    by_strategy: dict[str, GroupSummary]
    by_regime: dict[str, GroupSummary]
    by_combo: dict[tuple[str, str], GroupSummary]


def summarize(results: list[BacktestResult]) -> SweepReport:
    by_strategy: dict[str, list[BacktestResult]] = defaultdict(list)
    by_regime: dict[str, list[BacktestResult]] = defaultdict(list)
    by_combo: dict[tuple[str, str], list[BacktestResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)
        by_regime[r.regime].append(r)
        by_combo[(r.strategy, r.regime)].append(r)

    return SweepReport(
        total_runs=len(results),
        overall=_summarize_group("TOTAL", results),
        by_strategy={k: _summarize_group(k, v) for k, v in by_strategy.items()},
        by_regime={k: _summarize_group(k, v) for k, v in by_regime.items()},
        by_combo={k: _summarize_group(f"{k[0]}@{k[1]}", v) for k, v in by_combo.items()},
    )


def _row(s: GroupSummary) -> str:
    return (
        f"{s.label:<26} n={s.n:<6} prof%={s.pct_profitable * 100:5.1f} "
        f"retMed={s.median_return * 100:6.2f}% retMed_avg={s.mean_return * 100:6.2f}% "
        f"shpe={s.mean_sharpe:5.2f} sortino={s.mean_sortino:5.2f} "
        f"mddMed={s.mean_max_dd * 100:6.1f}% mddPior={s.worst_max_dd * 100:6.1f}% "
        f"win%={s.mean_win_rate * 100:4.0f}"
    )


def format_report(report: SweepReport) -> str:
    lines = ["=" * 120, f"SWEEP DE SIMULACAO — {report.total_runs} backtests (dados reais Alpaca)", "=" * 120]
    lines.append(_row(report.overall))
    lines.append("-" * 120)
    lines.append("Por estrategia:")
    for s in sorted(report.by_strategy.values(), key=lambda g: g.mean_return, reverse=True):
        lines.append("  " + _row(s))
    lines.append("-" * 120)
    lines.append("Por regime de mercado:")
    for s in sorted(report.by_regime.values(), key=lambda g: g.mean_return, reverse=True):
        lines.append("  " + _row(s))
    lines.append("-" * 120)
    lines.append("Por estrategia@regime (onde ganha / onde sangra):")
    for s in sorted(report.by_combo.values(), key=lambda g: g.mean_return):
        lines.append("  " + _row(s))
    lines.append("=" * 120)
    lines.append(
        "Legenda: prof%=janelas lucrativas retMed=retorno mediano shpe/sortino=media "
        "mddMed=drawdown medio mddPior=pior drawdown win%=acerto medio por trade"
    )
    return "\n".join(lines)
