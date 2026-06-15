"""Testes do champion/challenger (promocao so com skill comprovado)."""

from __future__ import annotations

import numpy as np

from ml.champion_challenger import evaluate
from ml.dataset import REGIMES, TrainingSet


def _names() -> list[str]:
    return ["ctx_score", "ctx_signal_strength"] + [f"regime_{g}" for g in REGIMES]


def _set(n: int, *, predictive: bool, seed: int = 0) -> TrainingSet:
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    if predictive:
        p = 1 / (1 + np.exp(-2.5 * score))
        y = (rng.random(n) < p).astype(int)
    else:
        y = rng.integers(0, 2, n)  # rotulo independente das features
    strength = rng.normal(0, 1, n)
    regime = np.eye(len(REGIMES))[rng.integers(0, len(REGIMES), n)]
    X = np.column_stack([score, strength, regime])
    pnl = np.where(y == 1, 100.0, -50.0)
    return TrainingSet(X=X, y=y, pnl=pnl, feature_names=_names())


def test_promove_quando_ha_skill():
    rep = evaluate(_set(300, predictive=True))
    assert rep.challenger_auc > 0.5
    assert rep.recommend_promote is True
    assert "PROMOVER" in rep.reason


def test_nao_promove_com_ruido():
    # Mesmo com um lift de ponto espurio (amostra finita), o IC99 deve barrar.
    rep = evaluate(_set(300, predictive=False, seed=7))
    assert rep.recommend_promote is False
    assert rep.auc_lower_ci <= 0.5  # nao significativo -> fica em shadow


def test_dados_insuficientes_fica_em_shadow():
    rep = evaluate(_set(12, predictive=True))
    assert rep.recommend_promote is False
    assert "insuficientes" in rep.reason


def test_expectancy_do_challenger_supera_baseline_quando_preditivo():
    rep = evaluate(_set(400, predictive=True))
    assert rep.challenger_expectancy > rep.baseline_expectancy
