"""Testes do DecisionEnricher (integracao das camadas)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from core.models import OrderSide, Signal
from feedback.models import MarketRegime
from integration.enricher import DecisionEnricher
from ml.dataset import REGIMES, _CONTEXT_KEYS
from ml.setup_classifier import SetupClassifier


def _ohlc_breakout(n: int = 61) -> pd.DataFrame:
    base, rows = 1.1000, []
    for i in range(n - 1):
        mid = base + 0.0001 * (1 if i % 2 == 0 else -1)
        o, c = mid, mid + 0.00005
        rows.append((o, max(o, c) + 3e-5, min(o, c) - 3e-5, c, 1000))
    o, c = base, base + 0.0030
    rows.append((o, c + 0.0002, o - 0.0001, c, 5000))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def _enrich(enricher, side=OrderSide.BUY, signals=None, ohlc=None,
            regime=MarketRegime.RANGE):
    return enricher.enrich(
        symbol="AAPL", side=side, strategy="trailing_stop", regime=regime,
        equity=Decimal("10000"), entry_price=Decimal("100"), stop_price=Decimal("98"),
        signals=signals, ohlc=ohlc,
    )


def test_neutro_sem_inputs_retorna_decisao():
    res = _enrich(DecisionEnricher())
    assert 0.0 <= res.confidence <= 1.0
    assert "synthesis_direction" in res.context
    assert res.p_win is None


def test_sinal_alinhado_eleva_confianca_e_qty():
    eng = DecisionEnricher()
    neutro = _enrich(eng)
    sig = [Signal(symbol="AAPL", side=OrderSide.BUY, source="congress", confidence=0.9)]
    alinhado = _enrich(eng, signals=sig)
    assert alinhado.confidence > neutro.confidence
    assert alinhado.qty >= neutro.qty


def test_sinal_oposto_derruba_confianca_e_zera_qty():
    sig = [Signal(symbol="AAPL", side=OrderSide.SELL, source="x", confidence=0.9)]
    res = _enrich(DecisionEnricher(), side=OrderSide.BUY, signals=sig)
    assert res.confidence < 0.5
    assert res.qty == 0  # abaixo do floor -> nao opera


def test_fimathe_features_no_contexto():
    res = _enrich(DecisionEnricher(), ohlc=_ohlc_breakout())
    assert "pcm_score" in res.context
    assert res.fimathe_signal in (-1, 0, 1)


def test_ml_promovido_gera_p_win():
    # classifier "promovido": treinado com o numero certo de features
    d = len(_CONTEXT_KEYS) + len(REGIMES)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, d))
    y = (X[:, 0] > 0).astype(int)
    clf = SetupClassifier().fit(X, y)
    res = _enrich(DecisionEnricher(classifier=clf), signals=[
        Signal(symbol="AAPL", side=OrderSide.BUY, source="congress", confidence=0.7)
    ])
    assert res.p_win is not None
    assert 0.0 <= res.p_win <= 1.0
    assert res.context["p_win"] is not None


def test_conflito_detectado_e_no_contexto():
    sigs = [
        Signal(symbol="AAPL", side=OrderSide.BUY, source="a", confidence=0.7),
        Signal(symbol="AAPL", side=OrderSide.SELL, source="b", confidence=0.6),
    ]
    res = _enrich(DecisionEnricher(), signals=sigs)
    assert "conflict" in res.context
