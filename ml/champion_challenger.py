"""Champion/Challenger — avalia o modelo de ML (challenger) contra o baseline
deterministico (champion), em dados que o modelo NAO viu.

Disciplina anti-overfitting: o ML so e recomendado para "promocao" (passar a
influenciar decisoes) quando, num split TEMPORAL (sem look-ahead), ele:
  1. tem skill acima do acaso (AUC >= 0.5 + min_lift), E
  2. melhora a expectancy vs operar tudo (take-all).
Ate la, roda em SHADOW (so observa/loga), nunca veta de verdade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.dataset import TrainingSet
from ml.setup_classifier import SetupClassifier


@dataclass
class ShadowReport:
    n_train: int
    n_test: int
    champion_acc: float       # baseline: prever a classe majoritaria
    challenger_acc: float     # ML: proba >= 0.5
    challenger_auc: float     # 0.5 = sem skill
    auc_lower_ci: float       # limite inferior 99% (Hanley-McNeil) — anti-overfit
    baseline_expectancy: float    # P&L medio operando TUDO no teste
    challenger_expectancy: float  # P&L medio dos setups que o ML aprovaria
    lift_auc: float           # challenger_auc - 0.5
    recommend_promote: bool
    reason: str


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks com media em empates (suficiente para AUC)."""
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


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5  # indefinido -> sem skill
    ranks = _rankdata(scores)
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Erro-padrao da AUC (Hanley-McNeil) — usado para o IC anti-overfitting."""
    if n_pos == 0 or n_neg == 0:
        return 0.5
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    num = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    )
    return float(np.sqrt(max(num, 0.0) / (n_pos * n_neg)))


def evaluate(
    ts: TrainingSet,
    *,
    test_frac: float = 0.3,
    threshold: float = 0.5,
    min_test: int = 10,
    min_lift: float = 0.03,
    classifier: SetupClassifier | None = None,
) -> ShadowReport:
    """Treina no inicio da serie e avalia no FINAL (split temporal)."""
    n = len(ts)
    n_test = int(round(n * test_frac))
    if n < (min_test * 2) or n_test < min_test:
        return ShadowReport(
            n_train=max(0, n - n_test), n_test=n_test,
            champion_acc=0.0, challenger_acc=0.0, challenger_auc=0.5,
            auc_lower_ci=0.5, baseline_expectancy=0.0, challenger_expectancy=0.0,
            lift_auc=0.0, recommend_promote=False,
            reason=f"dados insuficientes (n={n}); rodando em shadow",
        )

    split = n - n_test
    Xtr, ytr = ts.X[:split], ts.y[:split]
    Xte, yte, pnlte = ts.X[split:], ts.y[split:], ts.pnl[split:]

    model = (classifier or SetupClassifier()).fit(Xtr, ytr, ts.feature_names)
    proba = model.predict_proba(Xte)

    # Champion = baseline sem skill: classe majoritaria do treino.
    maj = 1 if ytr.mean() >= 0.5 else 0
    champion_acc = float((yte == maj).mean())
    challenger_acc = float(((proba >= 0.5).astype(int) == yte).mean())
    auc = _auc(yte, proba)

    n_pos = int((yte == 1).sum())
    n_neg = int((yte == 0).sum())
    se = _auc_se(auc, n_pos, n_neg)
    # IC 99% unilateral (z=2.33): barra conservadora — o modelo vai influenciar
    # decisoes com dinheiro real, entao exigimos skill bem acima do acaso.
    lower_ci = auc - 2.33 * se

    baseline_exp = float(pnlte.mean()) if len(pnlte) else 0.0
    taken = proba >= threshold
    challenger_exp = float(pnlte[taken].mean()) if taken.any() else 0.0

    lift = auc - 0.5
    # PROMOVER exige: skill SIGNIFICATIVO (IC inferior > 0.5), lift minimo no
    # ponto, E melhora de expectancy. O IC barra o "lift" espurio de amostra
    # pequena — o erro classico de overfitting.
    promote = bool((lower_ci > 0.5) and (lift >= min_lift) and (challenger_exp > baseline_exp))
    reason = (
        f"PROMOVER: AUC {auc:.3f} (IC99 inf {lower_ci:.3f} > 0.5) e expectancy "
        f"{challenger_exp:.2f} > baseline {baseline_exp:.2f}"
        if promote
        else f"manter em shadow: AUC {auc:.3f} (IC99 inf {lower_ci:.3f}), "
        f"expectancy {challenger_exp:.2f} vs baseline {baseline_exp:.2f}"
    )
    return ShadowReport(
        n_train=split, n_test=n_test,
        champion_acc=champion_acc, challenger_acc=challenger_acc,
        challenger_auc=auc, auc_lower_ci=lower_ci, baseline_expectancy=baseline_exp,
        challenger_expectancy=challenger_exp, lift_auc=lift,
        recommend_promote=promote, reason=reason,
    )


def evaluate_from_log(db_path: str = "data/trading.sqlite") -> ShadowReport:
    """Le o DecisionLog, monta o dataset e avalia. Atalho para CLI/cron."""
    from feedback.decision_log import DecisionLog
    from ml.dataset import build_training_set

    log = DecisionLog(db_path=db_path)
    try:
        return evaluate(build_training_set(log.all_records()))
    finally:
        log.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/trading.sqlite"
    rep = evaluate_from_log(path)
    print(
        f"[champion/challenger] n_train={rep.n_train} n_test={rep.n_test} "
        f"AUC={rep.challenger_auc:.3f} IC99inf={rep.auc_lower_ci:.3f} "
        f"acc_ml={rep.challenger_acc:.3f} acc_base={rep.champion_acc:.3f}\n"
        f"  {rep.reason}\n"
        f"  recomenda promover: {rep.recommend_promote}"
    )
