"""Testes do ModelStore: round-trip, atomicidade e ausencia segura."""

from __future__ import annotations

import numpy as np
import pytest

from ml.model_store import ModelStore, load_promoted_classifier
from ml.setup_classifier import SetupClassifier


def _trained() -> SetupClassifier:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] > 0).astype(int)
    return SetupClassifier(epochs=200).fit(X, y, ["a", "b", "c"])


def test_save_load_round_trip(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    clf = _trained()
    store.save(clf, metadata={"mean_auc": 0.7})

    loaded = store.load()
    assert loaded is not None and loaded.is_trained
    assert np.allclose(loaded.coef_, clf.coef_)
    assert loaded.feature_names == clf.feature_names

    rec = store.load_record()
    assert rec.metadata["mean_auc"] == 0.7
    assert rec.promoted_at  # carimbo de tempo presente


def test_load_ausente_devolve_none(tmp_path):
    assert ModelStore(tmp_path / "nada.json").load() is None
    assert load_promoted_classifier(tmp_path / "nada.json") is None


def test_clear_remove_modelo(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    store.save(_trained())
    assert store.exists()
    store.clear()
    assert not store.exists()
    assert store.load() is None
    store.clear()  # idempotente: nao quebra se ja nao existe


def test_recusa_modelo_nao_treinado(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    with pytest.raises(ValueError):
        store.save(SetupClassifier())
    assert not store.exists()


def test_save_e_atomico_sem_deixar_tmp(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    store.save(_trained())
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_arquivo_corrompido_trata_como_ausente(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{nao eh json valido", encoding="utf-8")
    assert ModelStore(p).load() is None
