"""Testes do modo de swings por PIVOS (causal, sem scipy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fimathe.engine import FimatheEngine


def _zigzag(n: int = 120, period: int = 30, amp: float = 0.01) -> pd.DataFrame:
    """Onda triangular com picos/vales de 1 barra (pivos limpos e estritos)."""
    half = period // 2
    rows = []
    for i in range(n):
        phase = i % period
        v = 1.10 + amp * (phase / half if phase < half else (period - phase) / half)
        rows.append((v, v + 0.0002, v - 0.0002, v, 1000))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_swing_method_invalido_levanta():
    with pytest.raises(ValueError):
        FimatheEngine(swing_method="nope")


def test_pivots_roda_o_pipeline_completo():
    eng = FimatheEngine(swing_period=10, swing_method="pivots")
    df = eng.process(_zigzag())
    assert {"upper_channel", "lower_channel", "signal", "pcm_score"}.issubset(df.columns)
    # ha pelo menos algum canal definido apos o warmup
    assert df["upper_channel"].notna().any()


def test_pivots_sao_CAUSAIS_sem_look_ahead():
    """O canal no indice i NAO pode mudar quando velas futuras sao removidas.
    Este e o bug da versao original (argrelextrema + ffill) — aqui deve passar."""
    eng = FimatheEngine(swing_period=10, swing_method="pivots")
    df = _zigzag(120)
    full = eng.detect_channels(df.copy())
    for i in (60, 80, 100):
        part = eng.detect_channels(df.iloc[: i + 1].copy())
        a = full["upper_channel"].iloc[i]
        b = part["upper_channel"].iloc[i]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b)
        a2 = full["lower_channel"].iloc[i]
        b2 = part["lower_channel"].iloc[i]
        assert (pd.isna(a2) and pd.isna(b2)) or a2 == pytest.approx(b2)


def test_local_extrema_encontra_pico_e_vale():
    vals = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1], dtype=float)
    highs = FimatheEngine._local_extrema(vals, order=2, greater=True)
    lows = FimatheEngine._local_extrema(vals, order=2, greater=False)
    assert 6 in highs  # o 5 e um pico local (order=2)
    assert 4 in lows   # o 1 no indice 4 e um vale local


def test_rolling_continua_sendo_o_padrao():
    eng = FimatheEngine()
    assert eng.swing_method == "rolling"
