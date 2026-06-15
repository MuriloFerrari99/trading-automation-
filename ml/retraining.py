"""Job de retreino + promocao AUTOMATICA do classificador de setup.

Fecha o ciclo que estava aberto: ate aqui, o ML era treinado, avaliado e... o
veredito de promocao morria num booleano de relatorio (promocao manual). Este
job roda periodicamente (cron/APScheduler) ou via CLI e:

  1. monta o dataset a partir do DecisionLog (decisoes fechadas win/loss);
  2. valida em WALK-FORWARD (varias janelas IS->OOS, nao um split so);
  3. mede CALIBRACAO nas predicoes OOS (Brier/ECE);
  4. se o gate passa, treina o modelo final em TODOS os dados e PROMOVE
     (grava no ModelStore, de onde o runtime carrega no boot).

Se o gate nao passa, NAO sobrescreve o modelo vigente (degrada com graca: segue
o que estava promovido, ou shadow se nunca houve um). Conservador de proposito —
o modelo influencia dinheiro real.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.calibration import CalibrationReport, calibration_report
from ml.dataset import build_training_set
from ml.model_store import DEFAULT_MODEL_PATH, ModelStore
from ml.setup_classifier import SetupClassifier
from ml.walk_forward import WalkForwardReport, walk_forward


@dataclass
class RetrainingResult:
    promoted: bool
    reason: str
    n_samples: int
    walk_forward: WalkForwardReport
    calibration: CalibrationReport | None
    model_path: str | None  # caminho do modelo promovido (None se nao promoveu)

    def summary(self) -> str:
        cal = f" | {self.calibration.summary()}" if self.calibration else ""
        head = "PROMOVIDO" if self.promoted else "MANTIDO EM SHADOW"
        return f"[retraining] {head}: {self.reason}{cal}"


def run_retraining(
    records: list[dict],
    store: ModelStore | None = None,
    *,
    n_folds: int = 5,
    min_lift: float = 0.03,
    min_frac_folds: float = 0.6,
    max_ece: float | None = 0.15,
) -> RetrainingResult:
    """Avalia em walk-forward e promove se houver skill consistente.

    `max_ece`: se definido, calibracao pior que esse ECE rebaixa a promocao
    (mesmo com AUC boa) — proba mal calibrada estraga o sizing por confianca.
    None desliga o gate de calibracao.
    """
    store = store or ModelStore()
    ts = build_training_set(records)

    wf = walk_forward(ts, n_folds=n_folds, min_lift=min_lift, min_frac_folds=min_frac_folds)
    cal: CalibrationReport | None = None
    if len(wf.oos_y):
        cal = calibration_report(wf.oos_y, wf.oos_proba)

    promote = wf.recommend_promote
    reason = wf.reason
    if promote and max_ece is not None and cal is not None and cal.ece > max_ece:
        promote = False
        reason = f"skill OK mas calibracao ruim (ECE {cal.ece:.3f} > {max_ece}); shadow"

    if not promote:
        return RetrainingResult(
            promoted=False, reason=reason, n_samples=len(ts),
            walk_forward=wf, calibration=cal, model_path=None,
        )

    # Gate passou: treina o modelo final em TODO o historico e promove.
    final = SetupClassifier().fit(ts.X, ts.y, ts.feature_names)
    metadata = {
        "n_samples": len(ts),
        "n_folds": wf.n_folds,
        "mean_auc": round(wf.mean_auc, 4),
        "mean_auc_lower_ci": round(wf.mean_lower_ci, 4),
        "frac_folds_positive": round(wf.frac_folds_positive, 4),
        "mean_challenger_expectancy": round(wf.mean_challenger_expectancy, 4),
        "mean_baseline_expectancy": round(wf.mean_baseline_expectancy, 4),
        "brier": None if cal is None else round(cal.brier, 4),
        "ece": None if cal is None else round(cal.ece, 4),
        "reason": reason,
    }
    rec = store.save(final, metadata=metadata)
    return RetrainingResult(
        promoted=True, reason=f"{reason} | promovido em {rec.promoted_at}",
        n_samples=len(ts), walk_forward=wf, calibration=cal, model_path=str(store.path),
    )


def run_retraining_from_log(
    db_path: Path | str = "data/trading.sqlite",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    **kwargs,
) -> RetrainingResult:
    """Atalho para CLI/cron: le o DecisionLog e roda o retreino."""
    from feedback.decision_log import DecisionLog

    log = DecisionLog(db_path=db_path)
    try:
        return run_retraining(log.all_records(), ModelStore(model_path), **kwargs)
    finally:
        log.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "data/trading.sqlite"
    res = run_retraining_from_log(db)
    print(res.summary())
    for f in res.walk_forward.folds:
        print(
            f"  dobra {f.index}: treino={f.n_train} teste={f.n_test} "
            f"AUC={f.auc:.3f} (IC99inf {f.auc_lower_ci:.3f}) "
            f"exp_ml={f.challenger_expectancy:.2f} exp_base={f.baseline_expectancy:.2f} "
            f"{'OK' if f.positive else '-'}"
        )
