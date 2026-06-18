"""Spike skfolio — Combinatorial Purged CV sobre o dataset de ML REAL.

Radar de stack (Fase 2): o veredito do ml/bootstrap foi "AUC ~0.48, ML sem edge".
Esse numero veio de UM split temporal (ml.walk_forward). A duvida que sobra e:
o AUC baixo e *falta de edge* ou *overfit/leakage* mal medido? Este spike responde
comparando, sobre o MESMO dataset rotulado (data/ml_bootstrap.sqlite, 1479 trades):

  1. KFold embaralhado  -> AUC OTIMISTA (vaza ordem temporal; baseline "ingenuo").
  2. CombinatorialPurgedCV (skfolio) -> AUC HONESTO (purga+embargo, multiplos
     caminhos de teste, a la Lopez de Prado). E o lever que o mlfinlab (pago)
     prometia e o skfolio (BSD, vivo) entrega de graca.

Leitura do resultado:
  - naive >> purged  -> havia leakage; o edge "real" e menor do que parecia.
  - ambos ~0.50      -> sem edge mesmo (confirma o veredito; gate de regime fica).
  - purged > 0.55    -> ha sinal que o split unico do walk_forward nao captou.

Honesto por construcao: reusa o MESMO vetorizador do ML vivo (ml.dataset.row_features),
entao nao ha skew treino/predicao. So LE o sqlite — nada de rede, broker ou ordem.
Determinista (seed fixa no shuffle; LogisticRegression e deterministico).

CLI:
    uv run --with skfolio --with scikit-learn python -m simulation.skfolio_cpcv_spike
    # ou, com o extra instalado:  uv sync --extra quant-ml && uv run python -m simulation.skfolio_cpcv_spike
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ml.dataset import feature_names, row_features

DEFAULT_DB = Path("data/ml_bootstrap.sqlite")


def load_dataset(db_path: Path | str = DEFAULT_DB) -> tuple[np.ndarray, np.ndarray]:
    """Le as decisoes FECHADAS em ordem temporal -> (X features, y win/loss).

    Usa o mesmo `row_features` do ML vivo (contexto numerico + one-hot de regime).
    y = 1 se return_pct > 0 (win), 0 caso contrario. Ordem por ts (a CPCV depende
    da ordem temporal para purgar/embargar corretamente).
    """
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT context, regime, return_pct FROM decisions "
            "WHERE return_pct IS NOT NULL ORDER BY ts ASC"
        ).fetchall()
    finally:
        con.close()

    X, y = [], []
    for context, regime, return_pct in rows:
        ctx = json.loads(context) if context else {}
        X.append(row_features(ctx, regime or "unknown"))
        y.append(1 if float(return_pct) > 0 else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _flatten_idx(idx) -> np.ndarray:
    """Achata indices de teste da CPCV (que vem como folds de tamanhos diferentes)."""
    try:
        return np.asarray(idx, dtype=int).ravel()
    except (ValueError, TypeError):  # array irregular (object) -> concatena fold a fold
        return np.concatenate([np.asarray(x, dtype=int).ravel() for x in idx])


def run_spike(
    db_path: Path | str = DEFAULT_DB,
    *,
    n_folds: int = 10,
    n_test_folds: int = 2,
    seed: int = 7,
) -> dict:
    """Compara AUC ingenuo (KFold shuffle) vs honesto (CombinatorialPurgedCV)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from skfolio.model_selection import CombinatorialPurgedCV

    X, y = load_dataset(db_path)
    # Escalas muito distintas (dist_to_zn ~ 1e4 vs rsi ~ 1e2): padroniza antes.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    naive_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    naive = cross_val_score(model, X, y, cv=naive_cv, scoring="roc_auc", error_score=np.nan)

    # CPCV devolve multiplos caminhos de teste por split -> loop manual (o
    # cross_val_score do sklearn nao lida com o shape irregular dos folds).
    purged_cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)
    purged_scores = []
    for train, test in purged_cv.split(X, y):
        tr, te = _flatten_idx(train), _flatten_idx(test)
        if len(np.unique(y[te])) < 2:  # roc_auc indefinido com classe unica
            continue
        est = clone(model).fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        purged_scores.append(roc_auc_score(y[te], proba))
    purged = np.asarray(purged_scores, dtype=float)

    naive_auc = float(np.nanmean(naive))
    purged_auc = float(np.nanmean(purged))
    gap = naive_auc - purged_auc

    if purged_auc >= 0.55:
        verdict = "SINAL: purged AUC>=0.55 — ha edge que o split unico nao captou"
    elif gap >= 0.05:
        verdict = f"LEAKAGE: naive infla {gap:+.3f} sobre o honesto — edge real menor"
    else:
        verdict = "SEM EDGE: ambos ~0.50, confirma o veredito (gate de regime fica)"

    return {
        "n_samples": int(len(y)),
        "win_rate": float(np.mean(y)),
        "n_features": int(X.shape[1]),
        "naive_auc": naive_auc,
        "purged_auc": purged_auc,
        "purged_paths": int(purged_cv.get_n_splits(X, y)),
        "gap_naive_minus_purged": gap,
        "verdict": verdict,
    }


def main() -> None:
    r = run_spike()
    print("=== skfolio CPCV spike (dataset ML real: data/ml_bootstrap.sqlite) ===")
    print(f"amostras           : {r['n_samples']} (win rate {r['win_rate']:.1%})")
    print(f"features           : {r['n_features']} -> {feature_names()}")
    print(f"AUC KFold ingenuo  : {r['naive_auc']:.3f}")
    print(f"AUC CombPurgedCV   : {r['purged_auc']:.3f}  ({r['purged_paths']} caminhos)")
    print(f"gap (ingenuo-honesto): {r['gap_naive_minus_purged']:+.3f}")
    print(f"VEREDITO: {r['verdict']}")


if __name__ == "__main__":
    main()
