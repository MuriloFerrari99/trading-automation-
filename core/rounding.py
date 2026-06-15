"""Arredondamento de quantidade e preco ao passo do ativo (tick/lote).

Generalizacao multi-mercado: cada ativo pode ter um tick size (passo minimo de
preco) e um lote minimo (passo de quantidade). Antes de enviar uma ordem,
alinhamos qty e limit_price a esses passos para a corretora nao rejeitar.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


def round_qty(qty: Decimal, lot_size: Decimal | None, *, fractional: bool = False) -> Decimal:
    """Arredonda a quantidade ao lote (sempre PARA BAIXO — nunca excede o alvo).

    Sem lote: inteiro (acoes) ou valor cru (se fractional). Com lote: multiplo do lote."""
    if lot_size is not None and lot_size > 0:
        steps = (qty / lot_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * lot_size
    if fractional:
        return qty
    return qty.to_integral_value(rounding=ROUND_DOWN)


def round_price(price: Decimal | None, tick_size: Decimal | None) -> Decimal | None:
    """Arredonda o preco ao tick mais proximo. None/sem tick => inalterado."""
    if price is None or tick_size is None or tick_size <= 0:
        return price
    steps = (price / tick_size).to_integral_value(rounding=ROUND_HALF_UP)
    return steps * tick_size
