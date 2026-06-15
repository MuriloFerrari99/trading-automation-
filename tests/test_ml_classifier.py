"""Testes do SetupClassifier (logistica em numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.setup_classifier import SetupClassifier


def test_cold_start_devolve_neutro():
    m = SetupClassifier()
    proba = m.predict_proba(np.zeros((3, 4)))
    assert np.allclose(proba, 0.5)
    assert m.is_trained is False


def test_aprende_padrao_linearmente_separavel():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-3 * x))
    y = (rng.random(n) < p).astype(int)
    noise = rng.normal(0, 1, n)
    X = np.column_stack([x, noise])
    m = SetupClassifier().fit(X, y)
    acc = ((m.predict_proba(X) >= 0.5).astype(int) == y).mean()
    assert acc > 0.8
    # o feature preditivo deve pesar mais que o ruido
    assert abs(m.coef_[0]) > abs(m.coef_[1])


def test_persistencia_roundtrip():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (50, 3))
    y = (X[:, 0] > 0).astype(int)
    m = SetupClassifier().fit(X, y, feature_names=["a", "b", "c"])
    d = m.to_dict()
    m2 = SetupClassifier.from_dict(d)
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
    assert m2.feature_names == ["a", "b", "c"]


def test_fit_vazio_levanta():
    with pytest.raises(ValueError):
        SetupClassifier().fit(np.empty((0, 3)), np.array([]))
