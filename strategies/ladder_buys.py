"""Estrategia de Ladder Buys (compras escalonadas em quedas).

Para cada ativo com config de ladder, define-se uma ancora (preco de
referencia) e degraus: "compre `qty` ao cair `drop_pct` abaixo da ancora".
Conforme o preco cai, degraus mais profundos sao acionados, reduzindo o preco
medio da posicao.

Estado persistido em `state`:
- ladder_anchor:SYMBOL    -> ancora (definida no primeiro ciclo se nao vier da config).
- ladder_rung:SYMBOL:i    -> "filled" quando o degrau i ja foi comprado.

Cada degrau dispara no MAXIMO uma vez (idempotencia entre ciclos). Para
re-armar a escada (ex: apos zerar a posicao), limpe o estado do ativo via
`reset(symbol, state)`.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from config.watchlist import LadderConfig
from core.models import OrderIntent, OrderSide, OrderType
from data.state_repo import StateRepository
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("strategy.ladder_buys")


def anchor_key(symbol: str) -> str:
    return f"ladder_anchor:{symbol.upper()}"


def rung_key(symbol: str, index: int) -> str:
    return f"ladder_rung:{symbol.upper()}:{index}"


class LadderBuysStrategy(Strategy):
    name = "ladder_buys"

    def evaluate(self, ctx: StrategyContext) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        for item in ctx.watchlist.items:
            if item.ladder is None:
                continue
            symbol = item.symbol
            price = ctx.broker.get_last_price(symbol)
            anchor = self._resolve_anchor(symbol, item.ladder, price, ctx.state)

            for index, rung in enumerate(item.ladder.rungs):
                key = rung_key(symbol, index)
                if ctx.state.get(key) is not None:
                    continue  # degrau ja comprado

                trigger = anchor * (Decimal(1) - rung.drop_pct)
                if price <= trigger:
                    logger.info(
                        "Ladder %s degrau %d disparado: price=%s <= trigger=%s "
                        "(anchor=%s, -%s) qty=%s",
                        symbol, index, price, trigger, anchor, rung.drop_pct, rung.qty,
                    )
                    intents.append(
                        OrderIntent(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            qty=rung.qty,
                            order_type=OrderType.MARKET,
                            strategy=self.name,
                        )
                    )
                    ctx.state.set(key, "filled")

        return intents

    def _resolve_anchor(
        self,
        symbol: str,
        ladder: LadderConfig,
        price: Decimal,
        state: StateRepository,
    ) -> Decimal:
        """Retorna a ancora persistida; inicializa-a no primeiro ciclo."""
        stored = state.get_decimal(anchor_key(symbol))
        if stored is not None:
            return stored
        anchor = ladder.anchor_price if ladder.anchor_price is not None else price
        state.set_decimal(anchor_key(symbol), anchor)
        logger.info("Ladder %s ancora definida em %s", symbol, anchor)
        return anchor

    @staticmethod
    def reset(symbol: str, state: StateRepository, max_rungs: int = 64) -> None:
        """Re-arma a escada de um ativo limpando ancora e degraus comprados."""
        state.delete(anchor_key(symbol))
        for i in range(max_rungs):
            state.delete(rung_key(symbol, i))
