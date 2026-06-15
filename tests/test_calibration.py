"""Testes de calibracao: ECE/Brier distinguem proba honesta de proba enganosa."""

from __future__ import annotations

import numpy as np

from ml.calibration import calibration_report


def test_modelo_calibrado_tem_ece_e_brier_baixos():
    rng = np.random.default_rng(0)
    p = rng.random(20000)          # probas espalhadas em [0,1]
    y = (rng.random(20000) < p).astype(int)  # rotulo honesto: P(win)=p
    rep = calibration_report(y, p)
    assert rep.ece < 0.05
    assert rep.brier < 0.2


def test_modelo_descalibrado_e_pior_que_o_calibrado():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y_ok = (rng.random(20000) < p).astype(int)
    y_bad = (rng.random(20000) < 0.5).astype(int)  # rotulo ignora a proba
    cal_ok = calibration_report(y_ok, p)
    cal_bad = calibration_report(y_bad, p)
    assert cal_bad.ece > cal_ok.ece
    assert cal_bad.brier > cal_ok.brier


def test_tabela_de_confiabilidade_cobre_amostras():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    rep = calibration_report(y, p, n_bins=10)
    assert len(rep.bins) == 10
    assert sum(b.n for b in rep.bins) == 1000  # todo ponto cai em exatamente 1 faixa


def test_vazio_nao_quebra():
    rep = calibration_report([], [])
    assert rep.n == 0 and rep.ece == 0.0 and rep.brier == 0.0
