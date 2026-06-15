"""Testes do sizing dinamico (Kelly fracionario por confianca)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from sizing.dynamic import DynamicSizer, kelly_fraction


def test_kelly_fraction_valores_conhecidos():
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)   # (0.6-0.4)/1
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)   # sem edge
    assert kelly_fraction(0.3, 2.0) == 0.0                  # negativo -> 0
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)   # (1.2-0.4)/2
    assert kelly_fraction(0.9, -1.0) == 0.0                 # b invalido


def test_risk_pct_zero_abaixo_do_floor():
    s = DynamicSizer(confidence_floor=0.5)
    assert s.risk_pct(0.49) == 0.0
    assert s.risk_pct(0.55) > 0.0


def test_risk_pct_zero_sem_edge():
    # confianca acima do floor mas abaixo do breakeven do R:R -> sem edge
    s = DynamicSizer(confidence_floor=0.3, default_win_loss_ratio=2.0)
    assert s.risk_pct(0.33) == 0.0  # breakeven b=2 e ~0.333


def test_risk_pct_monotonico_e_capeado():
    s = DynamicSizer(
        kelly_cap=0.5, max_risk_pct=0.5, confidence_floor=0.4,
        min_risk_pct=0.0, default_win_loss_ratio=2.0,
    )
    vals = [s.risk_pct(c) for c in (0.45, 0.55, 0.7, 0.85, 1.0)]
    assert vals == sorted(vals)        # nao-decrescente com a confianca
    assert vals[2] > vals[0]           # graded na regiao nao saturada
    # cap respeitado
    s2 = DynamicSizer(kelly_cap=0.5, max_risk_pct=0.02)
    assert s2.risk_pct(1.0) == pytest.approx(0.02)


def test_qty_cresce_com_confianca():
    s = DynamicSizer(
        kelly_cap=0.5, max_risk_pct=0.5, confidence_floor=0.4,
        min_risk_pct=0.0, default_win_loss_ratio=2.0,
    )
    eq, entry, stop = Decimal("10000"), Decimal("100"), Decimal("98")
    q_low = s.qty(confidence=0.55, equity=eq, entry_price=entry, stop_price=stop)
    q_high = s.qty(confidence=0.85, equity=eq, entry_price=entry, stop_price=stop)
    assert q_high > q_low > 0


def test_qty_zero_abaixo_do_floor():
    s = DynamicSizer(confidence_floor=0.5)
    q = s.qty(confidence=0.4, equity=Decimal("10000"),
              entry_price=Decimal("100"), stop_price=Decimal("98"))
    assert q == 0


def test_qty_stop_invalido_zero():
    s = DynamicSizer()
    q = s.qty(confidence=0.9, equity=Decimal("10000"),
              entry_price=Decimal("100"), stop_price=Decimal("100"))
    assert q == 0


def test_qty_respeita_cap_de_exposicao():
    s = DynamicSizer(kelly_cap=0.5, max_risk_pct=0.5, confidence_floor=0.4)
    eq, entry, stop = Decimal("10000"), Decimal("100"), Decimal("98")
    # sem cap, alta confianca dimensiona grande; cap de 1% do equity limita
    q_sem = s.qty(confidence=0.95, equity=eq, entry_price=entry, stop_price=stop)
    q_com = s.qty(confidence=0.95, equity=eq, entry_price=entry, stop_price=stop,
                  max_per_symbol_pct=Decimal("0.01"))
    assert q_com < q_sem
    assert q_com == 1  # 1% de 10000 = 100 / preco 100 = 1
