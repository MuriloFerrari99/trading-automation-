"""Validacao WALK-FORWARD do classificador de setup.

O champion/challenger (`evaluate`) usa UM split temporal (treina no inicio,
testa no fim). Isso e honesto contra look-ahead, mas fragil: o veredito depende
de um unico periodo de teste, que pode ser bom ou ruim por sorte.

O walk-forward corrige isso: divide a serie cronologica em VARIAS janelas
IS->OOS (expanding por padrao). Em cada dobra treina no passado e mede AUC,
lift e expectancy no futuro imediato. So recomenda PROMOVER quando o skill e
consistente entre as dobras — nao um lance de sorte de um periodo so. As
predicoes OOS de todas as dobras sao acumuladas (`oos_y`/`oos_proba`) para
alimentar a calibracao (ml/calibration.py) sem vazamento.

Sem dependencia externa de ML (reusa SetupClassifier + ml/metrics).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ml.dataset import TrainingSet
from ml.metrics import auc as _auc
from ml.metrics import auc_lower_ci as _auc_lower_ci
from ml.setup_classifier import SetupClassifier


@dataclass
class Fold:
    """Resultado de uma dobra do walk-forward (treina [0:test_start), testa o resto)."""

    index: int
    n_train: int
    n_test: int
    auc: float
    auc_lower_ci: float
    lift: float
    baseline_expectancy: float
    challenger_expectancy: float
    positive: bool  # passou nos criterios de skill NESTA dobra


@dataclass
class WalkForwardReport:
    n_samples: int
    n_folds: int
    folds: list[Fold]
    mean_auc: float
    mean_lift: float
    mean_lower_ci: float
    frac_folds_positive: float
    mean_baseline_expectancy: float
    mean_challenger_expectancy: float
    recommend_promote: bool
    reason: str
    # predicoes OOS acumuladas (para calibracao honesta) — nao entram no repr.
    oos_y: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    oos_proba: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


def _empty_report(n: int, reason: str) -> WalkForwardReport:
    return WalkForwardReport(
        n_samples=n, n_folds=0, folds=[], mean_auc=0.5, mean_lift=0.0,
        mean_lower_ci=0.5, frac_folds_positive=0.0, mean_baseline_expectancy=0.0,
        mean_challenger_expectancy=0.0, recommend_promote=False, reason=reason,
    )


def walk_forward(
    ts: TrainingSet,
    *,
    n_folds: int = 5,
    min_train: int = 40,
    min_test: int = 10,
    threshold: float = 0.5,
    min_lift: float = 0.03,
    min_frac_folds: float = 0.6,
    classifier_factory: Callable[[], SetupClassifier] | None = None,
) -> WalkForwardReport:
    """Avalia o classificador em janelas expanding IS->OOS.

    PROMOVE se, AGREGANDO as dobras: o IC99 inferior medio da AUC fica acima do
    acaso (> 0.5), o lift medio atinge `min_lift`, a maioria das dobras
    (`min_frac_folds`) tem skill, E a expectancy media do challenger supera o
    baseline (operar tudo). Conservador de proposito.
    """
    n = len(ts)
    usable = n - min_train
    if n < min_train + min_test or usable < min_test:
        return _empty_report(n, f"dados insuficientes p/ walk-forward (n={n}); shadow")

    make = classifier_factory or SetupClassifier
    max_folds = max(1, usable // min_test)
    k = max(1, min(n_folds, max_folds))
    test_size = usable // k

    folds: list[Fold] = []
    oos_y: list[np.ndarray] = []
    oos_p: list[np.ndarray] = []

    for i in range(k):
        test_start = min_train + i * test_size
        test_end = n if i == k - 1 else test_start + test_size
        Xtr, ytr = ts.X[:test_start], ts.y[:test_start]
        Xte, yte, pnlte = ts.X[test_start:test_end], ts.y[test_start:test_end], ts.pnl[test_start:test_end]
        # Sem as duas classes no treino a logistica nao aprende ranking util:
        # pula a dobra (nao a penaliza nem a credita).
        if len(np.unique(ytr)) < 2 or len(yte) == 0:
            continue

        model = make().fit(Xtr, ytr, ts.feature_names)
        proba = model.predict_proba(Xte)
        a = _auc(yte, proba)
        n_pos = int((yte == 1).sum())
        n_neg = int((yte == 0).sum())
        lower = _auc_lower_ci(a, n_pos, n_neg)
        lift = a - 0.5
        base_exp = float(pnlte.mean()) if len(pnlte) else 0.0
        taken = proba >= threshold
        chal_exp = float(pnlte[taken].mean()) if taken.any() else 0.0
        positive = bool(lower > 0.5 and lift >= min_lift and chal_exp > base_exp)

        folds.append(Fold(
            index=i, n_train=int(test_start), n_test=int(test_end - test_start),
            auc=a, auc_lower_ci=lower, lift=lift, baseline_expectancy=base_exp,
            challenger_expectancy=chal_exp, positive=positive,
        ))
        oos_y.append(np.asarray(yte))
        oos_p.append(np.asarray(proba))

    if not folds:
        return _empty_report(n, "nenhuma dobra avaliavel (classe unica no treino); shadow")

    mean_auc = float(np.mean([f.auc for f in folds]))
    mean_lift = float(np.mean([f.lift for f in folds]))
    mean_lower = float(np.mean([f.auc_lower_ci for f in folds]))
    mean_base = float(np.mean([f.baseline_expectancy for f in folds]))
    mean_chal = float(np.mean([f.challenger_expectancy for f in folds]))
    frac_pos = sum(f.positive for f in folds) / len(folds)

    promote = bool(
        mean_lower > 0.5
        and mean_lift >= min_lift
        and frac_pos >= min_frac_folds
        and mean_chal > mean_base
    )
    reason = (
        f"PROMOVER: {len(folds)} dobras, AUC medio {mean_auc:.3f} "
        f"(IC99 inf {mean_lower:.3f} > 0.5), {frac_pos * 100:.0f}% das dobras com "
        f"skill, expectancy {mean_chal:.2f} > baseline {mean_base:.2f}"
        if promote
        else f"manter em shadow: AUC medio {mean_auc:.3f} (IC99 inf {mean_lower:.3f}), "
        f"{frac_pos * 100:.0f}% dobras com skill, exp {mean_chal:.2f} vs base {mean_base:.2f}"
    )
    return WalkForwardReport(
        n_samples=n, n_folds=len(folds), folds=folds, mean_auc=mean_auc,
        mean_lift=mean_lift, mean_lower_ci=mean_lower, frac_folds_positive=frac_pos,
        mean_baseline_expectancy=mean_base, mean_challenger_expectancy=mean_chal,
        recommend_promote=promote, reason=reason,
        oos_y=np.concatenate(oos_y), oos_proba=np.concatenate(oos_p),
    )
