"""Calibracao do classificador: as probabilidades previstas batem com a realidade?

AUC mede se o modelo ORDENA bem (ranking), mas nao se P(win)=0.7 de fato ganha
~70% das vezes. Como o sizing dimensiona a posicao PELA confianca, uma proba
descalibrada vira risco mal dosado mesmo com AUC boa. Aqui medimos:

  - Brier score: erro quadratico medio de probabilidade (0=perfeito, menor=melhor).
  - ECE (Expected Calibration Error): |acuracia - confianca| medio por faixa,
    ponderado pela ocupacao da faixa (0=perfeitamente calibrado).
  - Tabela de confiabilidade: por faixa de proba, quantos casos e o win-rate real.

Roda sobre predicoes OOS (do walk-forward) — nunca in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float   # confianca media prevista na faixa
    frac_pos: float    # win-rate real observado na faixa


@dataclass
class CalibrationReport:
    n: int
    brier: float
    ece: float
    bins: list[Bin]

    def summary(self) -> str:
        return f"n={self.n} brier={self.brier:.4f} ECE={self.ece:.4f}"


def calibration_report(y, proba, *, n_bins: int = 10) -> CalibrationReport:
    """Brier + ECE + tabela de confiabilidade a partir de rotulos 0/1 e probas."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    n = len(y)
    if n == 0:
        return CalibrationReport(n=0, brier=0.0, ece=0.0, bins=[])

    brier = float(np.mean((p - y) ** 2))

    # Faixas [0,1] uniformes; a borda direita do ultimo bin inclui 1.0.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[Bin] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(Bin(lo=float(lo), hi=float(hi), n=0, mean_pred=0.0, frac_pos=0.0))
            continue
        mean_pred = float(p[mask].mean())
        frac_pos = float(y[mask].mean())
        ece += (cnt / n) * abs(frac_pos - mean_pred)
        bins.append(Bin(lo=float(lo), hi=float(hi), n=cnt, mean_pred=mean_pred, frac_pos=frac_pos))

    return CalibrationReport(n=n, brier=brier, ece=float(ece), bins=bins)
