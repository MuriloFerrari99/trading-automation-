"""Estrategia de Trailing Stop — NATIVO da Alpaca (server-side).

Conforme decisao de arquitetura (doc 02 §1.3): usar o trailing stop NATIVO da
Alpaca, que sobrevive a um crash do bot. Em vez de o bot calcular o high-water
e vender a mercado, ele coloca UMA ordem `trailing_stop` (com trail_percent) e
o broker rastreia a maxima e dispara sozinho.

Logica por ativo da watchlist com trailing configurado:
- Sem posicao comprada: nada a proteger.
- Com posicao E sem ordem trailing aberta para o ativo: emite UMA OrderIntent
  do tipo TRAILING_STOP (side=SELL, qty = posicao inteira, trail_percent).
- Com posicao E ja existe trailing aberta: nao faz nada (ja protegido).

A verificacao de "ja existe trailing aberta" usa o broker como fonte de verdade
(get_open_orders), entao sobrevive a restart sem duplicar protecao.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from core.models import OrderIntent, OrderSide, OrderType
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("strategy.trailing_stop")


class TrailingStopStrategy(Strategy):
    name = "trailing_stop"

    def evaluate(self, ctx: StrategyContext) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        # Ordens trailing ja abertas no broker, por simbolo (broker = verdade).
        open_trailing = self._open_trailing_symbols(ctx)

        for item in ctx.watchlist.items:
            if item.trailing_stop_pct is None:
                continue
            symbol = item.symbol

            position = ctx.broker.get_position(symbol)
            if position is None or position.qty <= 0:
                continue  # sem posicao: nada a proteger

            if symbol in open_trailing:
                continue  # ja existe trailing stop nativo cobrindo o ativo

            trail_percent = item.trailing_stop_pct * Decimal(100)  # fracao -> pontos %
            logger.info(
                "Colocando trailing stop nativo %s: qty=%s trail=%s%%",
                symbol, position.qty, trail_percent,
            )
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=position.qty,
                    order_type=OrderType.TRAILING_STOP,
                    trail_percent=trail_percent,
                    strategy=self.name,
                )
            )

        return intents

    @staticmethod
    def _open_trailing_symbols(ctx: StrategyContext) -> set[str]:
        try:
            orders = ctx.broker.get_open_orders()
        except Exception:
            logger.warning("Falha ao consultar ordens abertas; assumindo nenhuma.")
            return set()
        return {
            o.symbol.upper()
            for o in orders
            if "trailing" in (o.order_type or "").lower()
        }
