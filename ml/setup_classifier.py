"""SetupClassifier — regressao logistica (numpy puro) para P(win) de um setup.

Sem sklearn de proposito: o projeto e enxuto e uma logistica com padronizacao +
L2 e suficiente como primeiro modelo (e serve de baseline honesto para modelos
mais fortes depois — GBM/floresta seriam o proximo passo). Cold-start seguro:
enquanto NAO treinado, `predict_proba` devolve 0.5 (neutro) — nunca opina sem
dados.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # estavel para z muito negativo/positivo
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class SetupClassifier:
    def __init__(self, l2: float = 1.0, lr: float = 0.1, epochs: int = 800) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.is_trained: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> "SetupClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X vazio ou com shape invalido para fit")
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # coluna constante -> nao escala (intercept cuida)
        self.std_ = std
        xz = (X - self.mean_) / self.std_

        w = np.zeros(d)
        b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(xz @ w + b)
            err = p - y
            grad_w = xz.T @ err / n + self.l2 * w / n
            grad_b = float(err.mean())
            w -= self.lr * grad_w
            b -= self.lr * grad_b

        self.coef_ = w
        self.intercept_ = b
        self.feature_names = feature_names
        self.is_trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, self._dim())
        if not self.is_trained:
            return np.full(len(X), 0.5)  # cold-start: neutro
        xz = (X - self.mean_) / self.std_
        return _sigmoid(xz @ self.coef_ + self.intercept_)

    def _dim(self) -> int:
        if self.coef_ is not None:
            return len(self.coef_)
        if self.mean_ is not None:
            return len(self.mean_)
        return 1

    # --------------------------- persistencia --------------------------- #
    def to_dict(self) -> dict:
        return {
            "l2": self.l2,
            "lr": self.lr,
            "epochs": self.epochs,
            "coef_": None if self.coef_ is None else self.coef_.tolist(),
            "intercept_": self.intercept_,
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
            "std_": None if self.std_ is None else self.std_.tolist(),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SetupClassifier":
        m = cls(l2=d.get("l2", 1.0), lr=d.get("lr", 0.1), epochs=d.get("epochs", 800))
        m.coef_ = None if d.get("coef_") is None else np.array(d["coef_"], dtype=float)
        m.intercept_ = float(d.get("intercept_", 0.0))
        m.mean_ = None if d.get("mean_") is None else np.array(d["mean_"], dtype=float)
        m.std_ = None if d.get("std_") is None else np.array(d["std_"], dtype=float)
        m.feature_names = d.get("feature_names")
        m.is_trained = bool(d.get("is_trained", False))
        return m
