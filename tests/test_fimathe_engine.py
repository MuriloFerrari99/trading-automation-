"""Testes da FimatheEngine (Camada 1 — features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fimathe.engine import ML_FEATURES, FimatheEngine


def _ohlc_uptrend_breakout(n: int = 60) -> pd.DataFrame:
    """Serie lateral e, no fim, um rompimento de alta com corpo forte."""
    base = 1.1000
    rows = []
    for i in range(n):
        # lateral apertada nas primeiras n-1 velas
        mid = base + 0.0001 * (1 if i % 2 == 0 else -1)
        o, c = mid, mid + 0.00005
        h, lo = max(o, c) + 0.00003, min(o, c) - 0.00003
        rows.append((o, h, lo, c, 1000))
    # vela final: rompimento de alta, corpo grande e volume alto
    o = base
    c = base + 0.0030
    h = c + 0.0002
    lo = o - 0.0001
    rows.append((o, h, lo, c, 5000))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


@pytest.fixture()
def engine() -> FimatheEngine:
    return FimatheEngine(swing_period=20)


def test_process_adiciona_todas_as_colunas(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    esperadas = {
        "upper_channel", "lower_channel", "zone_neutra", "channel_width",
        "price_position", "price_position_code", "dist_to_zn",
        "fib_382", "fib_618", "fib_1272", "fib_1618",
        "pcm_score", "body_size", "breakout_strength",
        "rsi", "atr", "adx", "trend_regime", "adx_trend", "cycle_phase",
        "signal", "signal_strength", "setup_valid",
        "stop_loss", "take_profit_1", "take_profit_2", "position_size",
    }
    assert esperadas.issubset(df.columns)
    assert len(df) == 61


def test_canal_usa_velas_anteriores(engine):
    df = engine.detect_channels(_ohlc_uptrend_breakout())
    # primeiras swing_period velas sem canal (warmup) -> NaN
    assert df["upper_channel"].iloc[:20].isna().all()
    assert df["upper_channel"].iloc[25] == pytest.approx(
        df["high"].iloc[5:25].max(), abs=1e-9
    )


def test_rompimento_gera_sinal_de_compra(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    last = df.iloc[-1]
    assert last["signal"] == 1
    assert last["price_position"] == "acima"
    assert 0.0 < last["signal_strength"] <= 1.0
    assert bool(last["setup_valid"]) is True


def test_stops_coerentes_para_compra(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    last = df.iloc[-1]
    assert last["stop_loss"] < last["close"] < last["take_profit_1"]
    assert last["take_profit_2"] > last["take_profit_1"]
    assert last["position_size"] > 0


def test_sem_sinal_sem_stops(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    # numa vela lateral do meio nao ha sinal -> stops NaN, size 0
    mid = df.iloc[30]
    assert mid["signal"] == 0
    assert np.isnan(mid["stop_loss"])
    assert mid["position_size"] == 0.0


def test_indicadores_em_faixa(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    rsi = df["rsi"].dropna()
    adx = df["adx"].dropna()
    assert (rsi.between(0, 100)).all()
    assert (adx.between(0, 100)).all()
    assert (df["pcm_score"].between(0, 1)).all()


def test_features_para_ml(engine):
    df = engine.process(_ohlc_uptrend_breakout())
    feats = engine.get_features_for_ml(df)
    assert list(feats.columns) == list(ML_FEATURES)
    assert not feats.isna().any().any()  # dropna aplicado


def test_falta_coluna_obrigatoria_levanta(engine):
    with pytest.raises(ValueError):
        engine.process(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0]}))


def test_pcm_confirmation_desligavel():
    df = _ohlc_uptrend_breakout()
    eng = FimatheEngine(swing_period=20, pcm_confirmation=False)
    out = eng.process(df)
    # sem exigir corpo forte, o rompimento ainda gera compra
    assert out.iloc[-1]["signal"] == 1
