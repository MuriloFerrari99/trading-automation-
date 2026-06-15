"""Camada 3 — sintese de visoes (ensemble + racional).

Combina VISOES ortogonais sobre um ativo (tecnico, smart money, ML, regime,
noticias/sentimento no futuro) numa CONVICCAO unificada, com medida de consenso
e um racional legivel. Pluga um `Reasoner` opcional (LLM) para narrativa
qualitativa — mas o nucleo e deterministico e testavel, sem dependencia de LLM.

Respeita a regra do projeto: a sintese produz uma VISAO/score que MODULA
decisoes — nunca cria trades por conta propria. Ver docs/research/13-camada-sintese.md.
"""

from synthesis.reasoner import LLMReasoner, Reasoner, TemplateReasoner
from synthesis.synthesizer import SynthesisResult, Synthesizer, synthesize
from synthesis.views import Direction, View, view_from_proba, view_from_signal

__all__ = [
    "Direction",
    "View",
    "view_from_signal",
    "view_from_proba",
    "Synthesizer",
    "SynthesisResult",
    "synthesize",
    "Reasoner",
    "TemplateReasoner",
    "LLMReasoner",
]
