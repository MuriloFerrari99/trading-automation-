"""Testes da Camada 3 — sintese de visoes."""

from __future__ import annotations

from core.models import OrderSide, Signal
from synthesis.reasoner import LLMReasoner, TemplateReasoner
from synthesis.synthesizer import Synthesizer, synthesize
from synthesis.views import Direction, View, view_from_proba, view_from_signal


def _v(source, direction, conf=0.6, weight=1.0):
    return View(source=source, direction=direction, confidence=conf, weight=weight)


def test_sem_visoes_e_neutro():
    r = synthesize("AAPL", [])
    assert r.direction == Direction.NEUTRAL
    assert r.conviction == 0.0
    assert r.n_views == 0


def test_consenso_total_long():
    views = [_v("fimathe", Direction.LONG, 0.8), _v("congress", Direction.LONG, 0.6)]
    r = synthesize("AAPL", views)
    assert r.direction == Direction.LONG
    assert r.conviction > 0
    assert r.agreement == 1.0
    assert r.conflict is False


def test_conflito_sinalizado():
    # LONG vence no liquido, mas ha oposicao relevante -> conflito
    views = [
        _v("a", Direction.LONG, 0.7),
        _v("b", Direction.LONG, 0.1),
        _v("c", Direction.SHORT, 0.6),
    ]
    r = synthesize("AAPL", views, conflict_threshold=0.6)
    assert r.direction == Direction.LONG
    assert r.conflict is True
    assert r.agreement < 0.6


def test_forcas_equilibradas_dao_neutro():
    views = [_v("a", Direction.LONG, 0.6), _v("b", Direction.SHORT, 0.6)]
    r = synthesize("AAPL", views)
    assert r.direction == Direction.NEUTRAL


def test_peso_por_fonte_pode_inverter_direcao():
    synth = Synthesizer(source_weights={"strong": 5.0})
    views = [_v("weak", Direction.LONG, 0.6), _v("strong", Direction.SHORT, 0.6)]
    r = synth.synthesize("AAPL", views)
    assert r.direction == Direction.SHORT


def test_adaptadores():
    sig = Signal(symbol="MSFT", side=OrderSide.BUY, source="congress", confidence=0.7)
    v = view_from_signal(sig, weight=2.0)
    assert v.direction == Direction.LONG and v.confidence == 0.7 and v.weight == 2.0

    v2 = view_from_proba("ml", 0.66, OrderSide.SELL)
    assert v2.direction == Direction.SHORT and v2.confidence == 0.66


def test_template_reasoner_no_racional():
    views = [_v("fimathe", Direction.LONG, 0.8)]
    r = synthesize("AAPL", views, reasoner=TemplateReasoner())
    assert "AAPL" in r.rationale and "LONG" in r.rationale


def test_llm_reasoner_usa_call_fn():
    views = [_v("fimathe", Direction.LONG, 0.8)]
    r = synthesize("AAPL", views, reasoner=LLMReasoner(lambda prompt: "narrativa custom"))
    assert r.rationale == "narrativa custom"


def test_llm_reasoner_fallback_em_erro():
    def boom(_prompt):
        raise RuntimeError("LLM offline")

    views = [_v("fimathe", Direction.LONG, 0.8)]
    r = synthesize("AAPL", views, reasoner=LLMReasoner(boom))
    assert "AAPL" in r.rationale  # caiu no template, nao quebrou
