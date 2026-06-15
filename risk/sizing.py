"""Position sizing (doc 02 §4).

Default: fixed fractional (arrisca uma fracao fixa do equity por trade, com o
tamanho derivado da distancia ate o stop). ATR-based disponivel como opcao.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN


def fixed_fractional_qty(
    equity: Decimal,
    risk_pct: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
) -> Decimal:
    """Quantidade que arrisca `risk_pct` do equity, dado o stop.

    risk_$ = equity * risk_pct ; risk_per_share = |entry - stop| ;
    qty = floor(risk_$ / risk_per_share). Retorna 0 se o stop for invalido.
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0 or equity <= 0 or risk_pct <= 0:
        return Decimal(0)
    risk_dollars = equity * risk_pct
    qty = (risk_dollars / risk_per_share).to_integral_value(rounding=ROUND_DOWN)
    return max(qty, Decimal(0))


def atr_qty(
    equity: Decimal,
    risk_pct: Decimal,
    atr: Decimal,
    atr_mult: Decimal = Decimal("2.0"),
) -> Decimal:
    """Dimensiona com stop a `atr_mult * ATR` (risco em $ constante)."""
    risk_per_share = atr_mult * atr
    if risk_per_share <= 0 or equity <= 0 or risk_pct <= 0:
        return Decimal(0)
    qty = (equity * risk_pct / risk_per_share).to_integral_value(rounding=ROUND_DOWN)
    return max(qty, Decimal(0))


def max_qty_for_exposure(
    equity: Decimal,
    max_per_symbol_pct: Decimal,
    price: Decimal,
    current_qty: Decimal = Decimal(0),
) -> Decimal:
    """Quantidade ADICIONAL maxima sem ultrapassar a exposicao por simbolo."""
    if price <= 0 or equity <= 0:
        return Decimal(0)
    max_value = equity * max_per_symbol_pct
    max_total_qty = (max_value / price).to_integral_value(rounding=ROUND_DOWN)
    remaining = max_total_qty - current_qty
    return max(remaining, Decimal(0))
