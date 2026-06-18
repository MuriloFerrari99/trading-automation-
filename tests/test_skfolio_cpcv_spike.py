"""Teste do spike skfolio (Combinatorial Purged CV sobre o dataset de ML real).

Opt-in: skfolio/sklearn sao extras (uv sync --extra quant-ml). Sem eles, ou sem o
sqlite do bootstrap, o teste e PULADO — nao quebra a suite do core.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("skfolio")
pytest.importorskip("sklearn")

from simulation.skfolio_cpcv_spike import DEFAULT_DB, load_dataset, run_spike


@pytest.mark.skipif(not Path(DEFAULT_DB).exists(), reason="ml_bootstrap.sqlite ausente")
def test_dataset_carrega_features_e_labels():
    X, y = load_dataset()
    assert X.shape[0] == y.shape[0] > 0
    assert X.shape[1] == 11  # 6 features de contexto + 5 regimes one-hot
    assert set(int(v) for v in y) <= {0, 1}


@pytest.mark.skipif(not Path(DEFAULT_DB).exists(), reason="ml_bootstrap.sqlite ausente")
def test_cpcv_roda_e_produz_aucs_validos():
    r = run_spike(n_folds=10, n_test_folds=2)
    assert r["purged_paths"] == 45  # C(10,2)
    assert 0.0 <= r["naive_auc"] <= 1.0
    assert 0.0 <= r["purged_auc"] <= 1.0
    assert r["verdict"]  # alguma classificacao foi emitida
