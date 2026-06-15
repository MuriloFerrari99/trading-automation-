"""Testes do tribunal estatistico (simulation/statistics.py).

Casos conhecidos: ruido puro NAO deve passar; edge genuino e longo deve passar; o
obstaculo do DSR deve crescer com o numero de tentativas (data-snooping); PBO deve ser
alto para um conjunto de configs puramente aleatorias e baixo quando ha uma config com
edge real e persistente.
"""

from __future__ import annotations

import numpy as np

from simulation.statistics import (
    CRYPTO_PERIODS,
    EQUITY_PERIODS,
    evaluate_edge,
    expected_max_sharpe,
    kurtosis,
    observed_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    skew,
)


def make_returns(sharpe_per_period: float, n: int, seed: int = 0) -> np.ndarray:
    """Serie com Sharpe POR PERIODO exato = sharpe_per_period (std=1, mean=sharpe)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)  # mean 0, std(ddof=1) 1 exatos
    return z + sharpe_per_period


def test_observed_sharpe_exact_and_annualization():
    r = make_returns(0.1, 500, seed=1)
    assert abs(observed_sharpe(r) - 0.1) < 1e-9
    ann = observed_sharpe(r, periods_per_year=CRYPTO_PERIODS)
    assert abs(ann - 0.1 * np.sqrt(365)) < 1e-6


def test_psr_high_for_strong_long_series():
    # Sharpe anual ~2.5 em ~3 anos de dados diarios
    sr = 2.5 / np.sqrt(EQUITY_PERIODS)
    r = make_returns(sr, 756, seed=2)
    psr = probabilistic_sharpe_ratio(sr, r.size, skew(r), kurtosis(r), sr_benchmark=0.0)
    assert psr > 0.99


def test_psr_near_half_for_zero_edge():
    r = make_returns(0.0, 400, seed=3)
    psr = probabilistic_sharpe_ratio(0.0, r.size, skew(r), kurtosis(r))
    assert abs(psr - 0.5) < 0.05


def test_expected_max_sharpe_grows_with_trials():
    var = 0.004
    e10 = expected_max_sharpe(var, 10)
    e100 = expected_max_sharpe(var, 100)
    e1000 = expected_max_sharpe(var, 1000)
    assert 0 < e10 < e100 < e1000
    assert expected_max_sharpe(var, 1) == 0.0  # 1 trial = sem obstaculo


def test_evaluate_edge_passes_genuine_edge_few_trials():
    sr = 2.5 / np.sqrt(CRYPTO_PERIODS)
    r = make_returns(sr, 1095, seed=4)  # ~3 anos de barras diarias cripto
    v = evaluate_edge(r, n_trials=1, periods_per_year=CRYPTO_PERIODS)
    assert v.passes
    assert v.dsr > 0.95
    assert v.sharpe_annual > 0.8


def test_evaluate_edge_fails_same_edge_under_heavy_snooping():
    sr = 2.5 / np.sqrt(CRYPTO_PERIODS)
    r = make_returns(sr, 1095, seed=4)
    few = evaluate_edge(r, n_trials=1, periods_per_year=CRYPTO_PERIODS)
    many = evaluate_edge(r, n_trials=100_000, periods_per_year=CRYPTO_PERIODS)
    # mesmo Sharpe, mas o obstaculo do data-snooping sobe e o DSR despenca
    assert many.sr_benchmark_annual > few.sr_benchmark_annual
    assert many.dsr < few.dsr
    assert not many.passes_dsr


def test_evaluate_edge_fails_weak_sharpe():
    sr = 0.3 / np.sqrt(CRYPTO_PERIODS)  # Sharpe anual 0.3 — abaixo da barra
    r = make_returns(sr, 2000, seed=5)
    v = evaluate_edge(r, n_trials=1, periods_per_year=CRYPTO_PERIODS)
    assert not v.passes_sharpe
    assert not v.passes


def test_pbo_high_for_pure_noise():
    rng = np.random.default_rng(10)
    M = rng.standard_normal((120, 8))  # 8 configs, todas ruido
    pbo = probability_of_backtest_overfitting(M, n_splits=10)
    assert pbo > 0.3  # selecao da melhor IS nao se sustenta OOS


def test_pbo_low_with_one_real_edge():
    rng = np.random.default_rng(11)
    M = rng.standard_normal((120, 8))
    # coluna 0 com edge real e persistente (mean positivo em todo o periodo)
    M[:, 0] += 0.6
    pbo = probability_of_backtest_overfitting(M, n_splits=10)
    assert pbo < 0.15


def test_skew_kurtosis_basic():
    r = make_returns(0.0, 1000, seed=7)
    assert abs(skew(r)) < 0.3
    assert 2.0 < kurtosis(r) < 4.0  # ~3 para normal
