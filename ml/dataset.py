"""Extrai o dataset de treino do loop de feedback.

Cada DECISAO fechada com resultado (win/loss) no `DecisionLog` vira um exemplo
rotulado: features (do contexto + colunas) -> label (1=win, 0=loss). E o
dataset proprietario do qual a Camada 2 aprende. Mantem a ordem cronologica
(para split temporal sem look-ahead).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# Regimes na MESMA ordem usada no one-hot (estavel para reproducibilidade).
REGIMES = ("trend_up", "trend_down", "range", "high_vol", "unknown")

# Features numericas para o ML — APENAS o que e conhecido ANTES da decisao
# (nada de score/conviccao, que sao saidas: usa-las seria leaky/circular).
# signal_strength + features da FimatheEngine que o enricher grava no contexto.
# Chaves ausentes em linhas antigas viram 0.0 (graceful). O MESMO vetorizador
# serve treino e predicao (sem skew) — ver `context_to_features`.
_CONTEXT_KEYS = (
    "signal_strength",
    "pcm_score",
    "dist_to_zn",
    "breakout_strength",
    "rsi",
    "adx",
)


@dataclass
class TrainingSet:
    X: np.ndarray  # (n, d) features
    y: np.ndarray  # (n,) rotulos 0/1
    pnl: np.ndarray  # (n,) P&L realizado — para avaliacao por expectancy
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def _feature_names(context_keys: tuple[str, ...]) -> list[str]:
    return [f"ctx_{k}" for k in context_keys] + [f"regime_{g}" for g in REGIMES]


def feature_names(context_keys: tuple[str, ...] = _CONTEXT_KEYS) -> list[str]:
    """Nomes das features na ordem do vetor (publico, para inspecao)."""
    return _feature_names(context_keys)


def row_features(
    ctx: dict, regime: str, *, context_keys: tuple[str, ...] = _CONTEXT_KEYS
) -> list[float]:
    """Vetor de features de UMA decisao: contexto numerico + one-hot de regime.
    Usado por `build_training_set` (treino) E `context_to_features` (predicao)
    — garante que treino e producao usem EXATAMENTE o mesmo encoding."""
    regime = (regime or "unknown").lower()
    feats = [_as_float(ctx.get(k)) for k in context_keys]
    feats += [1.0 if regime == g else 0.0 for g in REGIMES]
    return feats


def context_to_features(
    ctx: dict, regime: str, *, context_keys: tuple[str, ...] = _CONTEXT_KEYS
) -> np.ndarray:
    """(1, d) pronto para `SetupClassifier.predict_proba` — mesmo encoding do treino."""
    return np.array([row_features(ctx, regime, context_keys=context_keys)], dtype=float)


def build_training_set(
    records: list[dict],
    *,
    context_keys: tuple[str, ...] = _CONTEXT_KEYS,
) -> TrainingSet:
    """Constroi (X, y, pnl) a partir das linhas do `DecisionLog` (ordem cron.).

    Considera apenas decisoes FECHADAS com win/loss (breakeven/open/skip ficam
    de fora — nao sao exemplos de qualidade de setup).
    """
    names = _feature_names(context_keys)
    rows_X: list[list[float]] = []
    rows_y: list[int] = []
    rows_pnl: list[float] = []

    for r in records:
        status = r.get("outcome_status")
        if status not in ("win", "loss"):
            continue
        ctx = {}
        if r.get("context"):
            try:
                ctx = json.loads(r["context"])
            except (ValueError, TypeError):
                ctx = {}
        feats = row_features(ctx, r.get("regime", "unknown"), context_keys=context_keys)
        rows_X.append(feats)
        rows_y.append(1 if status == "win" else 0)
        rows_pnl.append(_as_float(r.get("realized_pnl")))

    d = len(names)
    X = np.array(rows_X, dtype=float).reshape(-1, d)
    y = np.array(rows_y, dtype=int)
    pnl = np.array(rows_pnl, dtype=float)
    return TrainingSet(X=X, y=y, pnl=pnl, feature_names=names)


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
