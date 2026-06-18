"""Testes da saida tipada (Pydantic) do reasoner via LLM no trading.

Sem rede: o LLM e um call_fn fake. Cobre structured_call (valido/reparo/fallback)
e o StructuredLLMReasoner (narrativa tipada + fallback no template).
"""

from __future__ import annotations

import json

from synthesis.reasoner import NarrativaSintese, StructuredLLMReasoner
from synthesis.structured import extract_json, structured_call
from synthesis.views import Direction, View


def _views():
    return [
        View(source="momentum", direction=Direction.LONG, confidence=0.8,
             weight=1.0, rationale="rompimento de resistencia"),
        View(source="macro", direction=Direction.SHORT, confidence=0.6,
             weight=0.8, rationale="juro subindo"),
    ]


def test_extract_json_tolerante():
    assert extract_json('lixo {"a": 1} fim') == {"a": 1}
    assert extract_json("```json\n{\"a\": 2}\n```") == {"a": 2}
    assert extract_json("sem json") is None


def test_structured_call_valido():
    payload = json.dumps({"narrativa": "alta tatica", "drivers": ["momentum"], "conflitos": ["macro"]})
    out = structured_call(lambda p: payload, "prompt", NarrativaSintese,
                          default=NarrativaSintese(narrativa="fallback"))
    assert out.narrativa == "alta tatica"
    assert out.drivers == ["momentum"] and out.conflitos == ["macro"]


def test_structured_call_fallback():
    out = structured_call(lambda p: "nao e json", "prompt", NarrativaSintese,
                          default=NarrativaSintese(narrativa="FALLBACK"), repair_attempts=1)
    assert out.narrativa == "FALLBACK"


def test_structured_call_repara_no_segundo_try():
    respostas = iter(["{quebrado", '{"narrativa": "ok no reparo"}'])
    out = structured_call(lambda p: next(respostas), "prompt", NarrativaSintese,
                          default=NarrativaSintese(narrativa="fb"), repair_attempts=1)
    assert out.narrativa == "ok no reparo"


def test_reasoner_estruturado_explain():
    payload = json.dumps({"narrativa": "comprar com convicc moderada", "drivers": ["mom"]})
    r = StructuredLLMReasoner(call_fn=lambda p: payload)
    txt = r.explain("PETR4", _views(), direction=Direction.LONG, conviction=0.7, agreement=0.6)
    assert txt == "comprar com convicc moderada"
    obj = r.explain_structured("PETR4", _views(), direction=Direction.LONG, conviction=0.7, agreement=0.6)
    assert isinstance(obj, NarrativaSintese) and obj.drivers == ["mom"]


def test_reasoner_estruturado_fallback_template():
    # call_fn nao retorna JSON -> cai no default (narrativa do TemplateReasoner).
    r = StructuredLLMReasoner(call_fn=lambda p: "blá blá sem json")
    txt = r.explain("PETR4", _views(), direction=Direction.LONG, conviction=0.7, agreement=0.6)
    assert isinstance(txt, str) and txt  # narrativa de template, nao quebrou
