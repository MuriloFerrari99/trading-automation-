"""Testes de position sizing (doc 02 §4)."""

from __future__ import annotations

from decimal import Decimal

from risk.sizing import atr_qty, fixed_fractional_qty, max_qty_for_exposure


def test_fixed_fractional_basic():
    # equity 100k, risco 1% => $1000; stop a $5/share => 200 shares
    qty = fixed_fractional_qty(Decimal("100000"), Decimal("0.01"), Decimal("100"), Decimal("95"))
    assert qty == Decimal("200")


def test_fixed_fractional_invalid_stop_returns_zero():
    assert fixed_fractional_qty(Decimal("100000"), Decimal("0.01"), Decimal("100"), Decimal("100")) == 0


def test_atr_sizing():
    # equity 100k, risco 1% => $1000; ATR 2.50, mult 2 => risk/share 5 => 200
    qty = atr_qty(Decimal("100000"), Decimal("0.01"), Decimal("2.50"), Decimal("2"))
    assert qty == Decimal("200")


def test_max_qty_for_exposure_caps():
    # equity 100k, max 20%/simbolo => $20k; preco 100 => 200 shares no total
    # ja tenho 50 => posso adicionar 150
    add = max_qty_for_exposure(Decimal("100000"), Decimal("0.20"), Decimal("100"), Decimal("50"))
    assert add == Decimal("150")


def test_max_qty_for_exposure_already_full():
    add = max_qty_for_exposure(Decimal("100000"), Decimal("0.20"), Decimal("100"), Decimal("200"))
    assert add == Decimal("0")
