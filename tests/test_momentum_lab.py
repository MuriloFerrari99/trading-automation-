"""Testes do sinal de momentum (corretude e ausencia de look-ahead)."""

from __future__ import annotations

import numpy as np

from simulation.momentum_lab import Params, momentum_weights


def test_weights_zero_before_warmup():
    close = np.linspace(100, 200, 300)
    p = Params(lookback=60, vol_window=30, target_vol=0.4)
    w = momentum_weights(close, p)
    start = max(p.lookback, p.vol_window) + 1
    assert np.all(w[:start] == 0.0)


def test_flat_in_downtrend_long_only():
    # serie estritamente decrescente -> momentum sempre <=0 -> sempre flat
    close = np.linspace(200, 50, 300)
    p = Params(lookback=30, vol_window=20, target_vol=0.4)
    w = momentum_weights(close, p)
    assert np.all(w == 0.0)


def test_long_in_uptrend():
    close = np.linspace(50, 200, 300)
    p = Params(lookback=30, vol_window=20, target_vol=None)  # binario
    w = momentum_weights(close, p)
    start = max(p.lookback, p.vol_window) + 1
    assert np.all((w[start:] == 1.0))  # binario long em uptrend


def test_weights_bounded_by_cap():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.002, 0.03, 500))
    p = Params(lookback=30, vol_window=20, target_vol=0.4)
    w = momentum_weights(close, p, cap=1.0)
    assert w.min() >= 0.0
    assert w.max() <= 1.0


def test_no_lookahead_weight_uses_only_past():
    # mudar um preco FUTURO nao pode alterar o peso de um instante anterior
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400))
    p = Params(lookback=30, vol_window=20, target_vol=0.4)
    w1 = momentum_weights(close, p)
    altered = close.copy()
    altered[300:] *= 1.5  # mexe so no futuro
    w2 = momentum_weights(altered, p)
    assert np.allclose(w1[:300], w2[:300])
