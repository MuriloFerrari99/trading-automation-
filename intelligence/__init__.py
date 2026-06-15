"""Camada de decisao inteligente.

Usa o historico do loop de feedback (pacote `feedback/`) para tornar as
decisoes mais inteligentes ANTES de executar:
- classifica o regime de mercado (feedback.regime);
- veta combinacoes estrategia x regime com edge negativo comprovado
  (DecisionPolicy);
- modula a confianca com os sinais de smart money (sem nunca deixar um sinal
  CRIAR um trade — ele so pondera intents que uma estrategia ja produziu);
- registra TODA decisao (executada ou vetada) no DecisionLog, fechando o ciclo.

Ver docs/research/10-camada-decisao.md.
"""

from intelligence.decision_policy import DecisionPolicy, PolicyVerdict
from intelligence.engine import DecisionIntelligence, ProcessResult

__all__ = [
    "DecisionPolicy",
    "PolicyVerdict",
    "DecisionIntelligence",
    "ProcessResult",
]
