"""Testes offline do harness de simulacao (sem tocar a Alpaca — doc 06 §6)."""

from __future__ import annotations

import numpy as np

from simulation.engine import run_backtest
from simulation.metrics import compute_metrics, max_drawdown, sharpe
from simulation.montecarlo import make_windows, run_sweep, summarize


def _series(closes: list[float]) -> dict[str, list[float]]:
    # OHLC simplificado: open=close anterior, high/low = close +/- 0.1%.
    opens = [closes[0]] + closes[:-1]
    highs = [c * 1.001 for c in closes]
    lows = [c * 0.999 for c in closes]
    return {"open": opens, "high": highs, "low": lows, "close": closes}


def _trend_up(n=120, start=100.0, drift=0.004):
    return [start * (1 + drift) ** i for i in range(n)]


def _downtrend(n=120, start=100.0, drift=-0.004):
    return [start * (1 + drift) ** i for i in range(n)]


# --- metrics ----------------------------------------------------------------
def test_max_drawdown_simple():
    eq = np.array([100, 120, 90, 110], dtype=float)
    assert abs(max_drawdown(eq) - (90 / 120 - 1)) < 1e-9


def test_sharpe_zero_for_flat():
    assert sharpe(np.zeros(10)) == 0.0


def test_compute_metrics_basic():
    eq = np.array([100, 101, 102, 103], dtype=float)
    m = compute_metrics(eq, [1.0, -0.5, 2.0], n_bars=3, bars_in_market=3)
    assert m.total_return > 0
    assert m.n_trades == 3
    assert 0 <= m.win_rate <= 1


# --- engine -----------------------------------------------------------------
def test_run_backtest_trailing_uptrend_profits():
    ohlc = _series(_trend_up())
    r = run_backtest("AAPL", ohlc, "trailing_stop", warmup=35)
    assert r is not None
    assert r.strategy == "trailing_stop"
    assert r.regime in {"trend_up", "range", "high_vol", "trend_down", "unknown"}
    # buy-and-hold protegido por trailing em alta deve terminar positivo
    assert r.metrics.total_return > 0


def test_run_backtest_ladder_runs():
    ohlc = _series(_downtrend())
    r = run_backtest("AAPL", ohlc, "ladder_buys", warmup=35)
    assert r is not None
    assert r.strategy == "ladder_buys"


def test_run_backtest_too_short_returns_none():
    ohlc = _series([100.0] * 10)
    assert run_backtest("AAPL", ohlc, "trailing_stop", warmup=35) is None


# --- montecarlo -------------------------------------------------------------
def test_make_windows():
    w = make_windows(100, window_len=40, step=20)
    assert w == [(0, 40), (20, 60), (40, 80), (60, 100)]


def test_sweep_and_summarize():
    data = {
        "UP": _series(_trend_up(300)),
        "DOWN": _series(_downtrend(300)),
    }
    results = run_sweep(data, window_len=120, step=30)
    assert len(results) > 0
    report = summarize(results)
    assert report.total_runs == len(results)
    assert "trailing_stop" in report.by_strategy
    assert "ladder_buys" in report.by_strategy
    # cada combo tem contagem coerente
    assert sum(g.n for g in report.by_strategy.values()) == report.total_runs
