"""Testes do job de retreino: promove e PERSISTE so com skill consistente."""

from __future__ import annotations

import json

import numpy as np

from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, Outcome, OutcomeStatus
from ml.model_store import ModelStore
from ml.retraining import run_retraining, run_retraining_from_log

_REGIMES = ["trend_up", "trend_down", "range", "high_vol", "unknown"]


def _records(n: int, *, predictive: bool, seed: int = 0) -> list[dict]:
    """Linhas no formato da tabela `decisions` (so o que build_training_set le)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n):
        score = float(rng.normal(0, 1))
        if predictive:
            p = 1 / (1 + np.exp(-2.5 * score))
            win = rng.random() < p
        else:
            win = rng.random() < 0.5
        rows.append({
            "outcome_status": "win" if win else "loss",
            "regime": _REGIMES[rng.integers(0, len(_REGIMES))],
            "context": json.dumps({"pcm_score": score, "signal_strength": float(rng.random())}),
            "realized_pnl": 100.0 if win else -50.0,
        })
    return rows


def test_promove_e_persiste_quando_ha_skill(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    res = run_retraining(_records(400, predictive=True), store)
    assert res.promoted is True
    assert res.calibration is not None
    assert res.model_path is not None
    # O modelo promovido fica DISPONIVEL para o runtime carregar.
    loaded = store.load()
    assert loaded is not None and loaded.is_trained


def test_nao_promove_com_ruido_e_nao_grava(tmp_path):
    store = ModelStore(tmp_path / "m.json")
    res = run_retraining(_records(400, predictive=False, seed=9), store)
    assert res.promoted is False
    assert res.model_path is None
    assert store.load() is None  # nada foi persistido -> segue em shadow


def test_calibracao_ruim_rebaixa_promocao(tmp_path):
    # max_ece impossivelmente baixo: mesmo com skill, recusa promover.
    store = ModelStore(tmp_path / "m.json")
    res = run_retraining(_records(400, predictive=True), store, max_ece=0.0)
    assert res.promoted is False
    assert "calibracao" in res.reason
    assert store.load() is None


def test_caminho_via_decision_log(tmp_path):
    """Integracao: grava decisoes reais no DecisionLog e retreina a partir dele."""
    log = DecisionLog(db_path=":memory:")
    rng = np.random.default_rng(0)
    for _ in range(400):
        score = float(rng.normal(0, 1))
        win = rng.random() < 1 / (1 + np.exp(-2.5 * score))
        dec = Decision(
            strategy="ladder_buys",
            symbol="AAPL",
            action=DecisionAction.BUY,
            regime=MarketRegime.TREND_UP,
            context={"pcm_score": score, "signal_strength": float(rng.random())},
        )
        did = log.record(dec)
        log.attach_outcome(did, Outcome(
            status=OutcomeStatus.WIN if win else OutcomeStatus.LOSS,
            realized_pnl=(100 if win else -50),
        ))

    store = ModelStore(tmp_path / "m.json")
    # run_retraining_from_log abre seu proprio DecisionLog por path; aqui usamos a
    # API de records direto para reaproveitar a conexao :memory: deste teste.
    res = run_retraining(log.all_records(), store)
    log.close()
    assert res.n_samples == 400
    assert res.promoted is True
    assert store.load() is not None


def test_from_log_le_arquivo(tmp_path):
    """run_retraining_from_log le um sqlite em disco e roda o job."""
    db = tmp_path / "trading.sqlite"
    log = DecisionLog(db_path=db)
    rng = np.random.default_rng(1)
    for _ in range(400):
        score = float(rng.normal(0, 1))
        win = rng.random() < 1 / (1 + np.exp(-2.5 * score))
        did = log.record(Decision(
            strategy="s", symbol="X", action=DecisionAction.BUY,
            regime=MarketRegime.RANGE,
            context={"pcm_score": score, "signal_strength": float(rng.random())},
        ))
        log.attach_outcome(did, Outcome(
            status=OutcomeStatus.WIN if win else OutcomeStatus.LOSS,
            realized_pnl=(100 if win else -50),
        ))
    log.close()

    res = run_retraining_from_log(db, tmp_path / "m.json")
    assert res.promoted is True
    assert ModelStore(tmp_path / "m.json").load() is not None
