"""Estrategia de Trailing Stop.

Para cada ativo da watchlist em que ha posicao comprada:
- Mantem um high-water mark (a maxima observada) persistido em `state`.
- O stop = high_water * (1 - trailing_pct). Como o high_water nunca diminui,
  o stop nunca desce (sobe junto com o preco, conforme o briefing).
- Se o preco atual <= stop, gera uma OrderIntent de VENDA da posicao inteira
  (ordem a mercado) e reseta o high-water.

O ajuste do stop acontece implicitamente a cada ciclo: o Monitor chama a
estrategia periodicamente, ela atualiza o high-water e recalcula o gatilho.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from core.models import OrderIntent, OrderSide, OrderType
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("strategy.trailing_stop")


def high_water_key(symbol: str) -> str:
    return f"trailing_high:{symbol.upper()}"


class TrailingStopStrategy(Strategy):
    name = "trailing_stop"

    def evaluate(self, ctx: StrategyContext) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        for item in ctx.watchlist.items:
            if item.trailing_stop_pct is None:
                continue  # ativo sem trailing stop configurado
            symbol = item.symbol
            key = high_water_key(symbol)

            position = ctx.broker.get_position(symbol)
            if position is None or position.qty <= 0:
                # Sem posicao: nao ha o que proteger; limpa o high-water.
                ctx.state.delete(key)
                continue

            price = ctx.broker.get_last_price(symbol)

            # Inicializa o high-water com o maior entre preco de entrada e atual.
            stored_high = ctx.state.get_decimal(key)
            baseline = stored_high if stored_high is not None else position.avg_entry_price
            high_water = max(baseline, price)
            if high_water != stored_high:
                ctx.state.set_decimal(key, high_water)

            stop_level = high_water * (Decimal(1) - item.trailing_stop_pct)

            logger.debug(
                "%s price=%s high_water=%s stop=%s (pct=%s)",
                symbol, price, high_water, stop_level, item.trailing_stop_pct,
            )

            if price <= stop_level:
                logger.info(
                    "Trailing stop disparado %s: price=%s <= stop=%s (qty=%s)",
                    symbol, price, stop_level, position.qty,
                )
                intents.append(
                    OrderIntent(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=position.qty,
                        order_type=OrderType.MARKET,
                        strategy=self.name,
                    )
                )
                # Reset: posicao sera fechada; proximo ciclo recomeca limpo.
                ctx.state.delete(key)

        return intents
