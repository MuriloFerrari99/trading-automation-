"""Integracao das camadas de inteligencia (o "cerebro" que liga tudo).

Compoe, num unico ponto, as pecas construidas em camadas:
  - FimatheEngine (doc 08)  -> features ricas sobre as barras
  - Synthesis  (doc 13)     -> visoes ortogonais -> conviccao + conflito
  - SetupClassifier (doc 11)-> P(win), se promovido (champion/challenger)
  - DynamicSizer (doc 12)   -> quantidade pela confianca

Degrada com graca: cada peca e opcional/condicional. Sem barras OHLC, pula as
features da FimatheEngine; sem modelo promovido, ignora o ML; etc. Produz um
contexto enriquecido (para o DecisionLog), uma confianca e uma quantidade —
que o agente de decisao consome. NUNCA cria trades: so enriquece/score/sizing.

Ver docs/research/14-integracao.md.
"""

from integration.enricher import DecisionEnricher, EnrichedDecision

__all__ = ["DecisionEnricher", "EnrichedDecision"]
