"""Camada 0 — Loop de feedback.

O motor que torna o sistema "cada vez mais inteligente": registra toda DECISAO
(inclusive a de NAO operar), o CONTEXTO de mercado (regime, preco, sinais) e,
quando conhecido, o RESULTADO (P&L realizado). Em cima desse historico,
`evaluation` calcula metricas por estrategia e por regime.

Independente do resto do sistema de proposito: cria a propria tabela
`decisions` no SQLite e nao depende dos modelos de ordem nem do broker, para
poder evoluir (e ser testado) isoladamente. Ver docs/research/09-loop-de-feedback.md.
"""

from feedback.decision_log import DecisionLog
from feedback.evaluation import GroupStats, Report, evaluate, format_report
from feedback.models import (
    Decision,
    DecisionAction,
    MarketRegime,
    Outcome,
    OutcomeStatus,
)
from feedback.regime import classify_regime

__all__ = [
    "DecisionLog",
    "Decision",
    "DecisionAction",
    "MarketRegime",
    "Outcome",
    "OutcomeStatus",
    "classify_regime",
    "evaluate",
    "format_report",
    "GroupStats",
    "Report",
]
