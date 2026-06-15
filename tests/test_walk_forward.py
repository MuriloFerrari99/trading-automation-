"""Testes do walk-forward: promove so com skill consistente entre janelas."""

from __future__ import annotations

import numpy as np

from ml.dataset import REGIMES, TrainingSet
from ml.walk_forward import walk_forward


def _names() -> list[str]:
    return ["ctx_score", "ctx_signal_strength"] + [f"regime_{g}" for g in REGIMES]


def _set(n: int, *, predictive: bool, seed: int = 0) -> TrainingSet:
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    if predictive:
        p = 1 / (1 + np.exp(-2.5 * score))
        y = (rng.random(n) < p).astype(int)
    else:
        y = rng.integers(0, 2, n)
    strength = rng.normal(0, 1, n)
    regime = np.eye(len(REGIMES))[rng.integers(0, len(REGIMES), n)]
    X = np.column_stack([score, strength, regime])
    pnl = np.where(y == 1, 100.0, -50.0)
    return TrainingSet(X=X, y=y, pnl=pnl, feature_names=_names())


def test_promove_quando_skill_e_consistente():
    rep = walk_forward(_set(400, predictive=True))
    assert rep.n_folds >= 2
    assert rep.mean_auc > 0.5
    assert rep.mean_lower_ci > 0.5
    assert rep.recommend_promote is True
    assert "PROMOVER" in rep.reason


def test_nao_promove_com_ruido():
    rep = walk_forward(_set(400, predictive=False, seed=3))
    assert rep.recommend_promote is False


def test_dados_insuficientes_fica_em_shadow():
    rep = walk_forward(_set(20, predictive=True))
    assert rep.n_folds == 0
    assert rep.recommend_promote is False
    assert "insuficientes" in rep.reason


def test_acumula_predicoes_oos_para_calibracao():
    rep = walk_forward(_set(400, predictive=True))
    # As predicoes OOS cobrem todas as dobras (sem repetir) -> base p/ calibracao.
    assert len(rep.oos_y) == len(rep.oos_proba) > 0
    assert set(np.unique(rep.oos_y)).issubset({0, 1})
    assert rep.oos_proba.min() >= 0.0 and rep.oos_proba.max() <= 1.0
