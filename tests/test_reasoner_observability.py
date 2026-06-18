"""Garante que a instrumentacao Langfuse do LLMReasoner e transparente.

Sem chaves Langfuse no ambiente (caso padrao dos testes), observe_generation e
no-op: o LLMReasoner chama o call_fn injetado normalmente e cai no template em
caso de erro. Nao requer langfuse instalado.
"""

from __future__ import annotations

from synthesis.observability import langfuse_enabled, observe_generation
from synthesis.reasoner import LLMReasoner
from synthesis.views import Direction, View


def _views():
    return [
        View(source="momentum", direction=Direction.LONG, confidence=0.8,
             weight=1.0, rationale="tendencia de alta"),
    ]


def test_observe_generation_noop_sem_chaves(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_enabled() is False
    calls = []

    @observe_generation("teste")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(21) == 42
    assert calls == [21]  # passou direto, sem wrapper


def test_llm_reasoner_usa_call_fn(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    r = LLMReasoner(call_fn=lambda prompt: "NARRATIVA LLM")
    out = r.explain("PETR4", _views(), direction=Direction.LONG,
                    conviction=0.7, agreement=0.9)
    assert out == "NARRATIVA LLM"


def test_llm_reasoner_fallback_no_erro(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    def boom(_prompt):
        raise RuntimeError("LLM caiu")

    r = LLMReasoner(call_fn=boom)
    out = r.explain("PETR4", _views(), direction=Direction.LONG,
                    conviction=0.7, agreement=0.9)
    assert isinstance(out, str) and out  # caiu no template, nao quebrou
