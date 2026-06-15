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

# Features numericas extraidas do contexto da decisao (alem do one-hot de regime).
# Crescem automaticamente quando o contexto for enriquecido (ex.: features da
# FimatheEngine) — basta adicionar a chave aqui.
_CONTEXT_KEYS = ("score", "signal_strength")


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
        feats = [_as_float(ctx.get(k)) for k in context_keys]
        regime = (r.get("regime") or "unknown").lower()
        feats += [1.0 if regime == g else 0.0 for g in REGIMES]
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
