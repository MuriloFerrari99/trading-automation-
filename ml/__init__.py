"""Camada 2 — ML (classificador de qualidade de setup).

Aprende P(win) a partir das features da decisao (Camada 1) e dos resultados do
loop de feedback (Camada 0), e roda em CHAMPION/CHALLENGER: o modelo so e
considerado apto a influenciar decisoes quando PROVA, em dados que nao viu, que
bate o baseline deterministico (disciplina anti-overfitting).

Sem dependencia externa de ML (regressao logistica em numpy puro) — sklearn nao
esta instalado e o projeto e enxuto. Ver docs/research/11-camada-ml.md.
"""

from ml.champion_challenger import ShadowReport, evaluate
from ml.dataset import REGIMES, TrainingSet, build_training_set
from ml.setup_classifier import SetupClassifier

__all__ = [
    "SetupClassifier",
    "TrainingSet",
    "build_training_set",
    "REGIMES",
    "evaluate",
    "ShadowReport",
]
