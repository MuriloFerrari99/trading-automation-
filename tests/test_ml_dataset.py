"""Testes do extrator de dataset (decisao->exemplo rotulado)."""

from __future__ import annotations

import json

from ml.dataset import build_training_set


def _rec(status, regime="trend_up", score=0.7, strength=0.6, pnl="100"):
    return {
        "outcome_status": status,
        "regime": regime,
        "realized_pnl": pnl,
        "context": json.dumps({"score": score, "signal_strength": strength}),
    }


def test_extrai_apenas_decisoes_fechadas():
    records = [
        _rec("win"),
        _rec("loss", pnl="-50"),
        _rec("open"),       # ignorado
        _rec("skipped"),    # ignorado
        _rec("breakeven"),  # ignorado
    ]
    ts = build_training_set(records)
    assert len(ts) == 2
    assert ts.X.shape == (2, len(ts.feature_names))  # ctx + one-hot de regime
    assert list(ts.y) == [1, 0]
    assert list(ts.pnl) == [100.0, -50.0]


def test_one_hot_de_regime():
    ts = build_training_set([_rec("win", regime="high_vol")])
    names = ts.feature_names
    col = names.index("regime_high_vol")
    assert ts.X[0, col] == 1.0
    # so um regime ativo
    regime_cols = [i for i, nm in enumerate(names) if nm.startswith("regime_")]
    assert ts.X[0, regime_cols].sum() == 1.0


def test_contexto_invalido_nao_quebra():
    rec = _rec("win")
    rec["context"] = "{nao-e-json"
    ts = build_training_set([rec])
    assert len(ts) == 1  # features de contexto viram 0.0, sem excecao


def test_vazio_retorna_shape_coerente():
    ts = build_training_set([])
    assert len(ts) == 0
    assert ts.X.shape == (0, len(ts.feature_names))
