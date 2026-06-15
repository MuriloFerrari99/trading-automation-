"""Metricas de discriminacao para o classificador (numpy puro, sem sklearn).

Casa unica do calculo de AUC e do seu erro-padrao (Hanley-McNeil) — usado tanto
pelo champion/challenger (split unico) quanto pelo walk-forward (varias janelas).
Manter aqui evita duas implementacoes divergentes da mesma estatistica.
"""

from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks 1-based com media em empates (suficiente para AUC de Mann-Whitney)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks 1-based
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC via estatistica de Mann-Whitney. 0.5 = sem skill / indefinido."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5  # indefinido -> sem skill
    ranks = rankdata(scores)
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_se(a: float, n_pos: int, n_neg: int) -> float:
    """Erro-padrao da AUC (Hanley-McNeil) — base do IC anti-overfitting."""
    if n_pos == 0 or n_neg == 0:
        return 0.5
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    num = a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    return float(np.sqrt(max(num, 0.0) / (n_pos * n_neg)))


def auc_lower_ci(a: float, n_pos: int, n_neg: int, *, z: float = 2.33) -> float:
    """Limite inferior do IC unilateral da AUC (default z=2.33 -> 99%).

    Barra conservadora: o modelo vai influenciar dinheiro real, entao exigimos
    skill significativamente acima do acaso, nao so o ponto > 0.5.
    """
    return a - z * auc_se(a, n_pos, n_neg)
