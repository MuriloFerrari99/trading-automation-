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

from core.market_clock import asset_tradable_now
from core.models import OrderIntent, OrderSide, OrderType
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("strategy.trailing_stop")


class TrailingStopStrategy(Strategy):
    name = "trailing_stop"

    def __init__(
        self,
        default_trailing_stop_pct: Decimal | None = None,
        *,
        excluded_symbols: set[str] | None = None,
    ) -> None:
        # Trailing padrao para QUALQUER long sem config propria. None mantem o
        # comportamento legado (so protege itens com trailing_stop_pct).
        self._default_pct = default_trailing_stop_pct
        # Simbolos geridos por OUTRO book (ex.: o rebalanceador de beta) que NAO
        # devem receber trailing stop deste pipeline-base. A tese do beta e
        # vol-target/rebalance SEM stop por ativo — um trailing aqui liquidaria
        # o book numa queda de 10% e corromperia o track record (MF-1). Os
        # simbolos vem NA FORMA DO BROKER (ex.: 'BTC/USD'), que e como
        # get_positions os devolve. Set vazio => comportamento legado (flag off).
        self._excluded = {s.upper() for s in (excluded_symbols or set())}

    def evaluate(self, ctx: StrategyContext) -> list[OrderIntent]:
        intents: list[OrderIntent] = []

        # Ordens trailing ja abertas no broker, por simbolo (broker = verdade).
        open_trailing = self._open_trailing_symbols(ctx)

        # Protege TODO long aberto (broker = verdade), nao so os da watchlist:
        # posicoes orfas (ex: herdadas de um reconcile) tambem ganham stop.
        for position in self._open_longs(ctx):
            symbol = position.symbol

            # MF-1: posicoes de OUTRO book (beta) NUNCA recebem trailing stop
            # deste pipeline. Pula ANTES de qualquer logica (inclusive do ramo
            # cripto 24/7, que dispararia ate com o pregao fechado).
            if symbol.upper() in self._excluded:
                continue

            item = ctx.watchlist.get(symbol)

            asset_class = item.asset_class if item is not None else "equity"
            if not asset_tradable_now(asset_class, ctx.equity_market_open):
                continue  # ativo de acao fora do pregao (cripto, 24/7, segue)

            # Ladder gere seu PROPRIO stop de invalidacao (acumula na queda); um
            # trailing aqui venderia cedo e brigaria com a tese. Pulamos.
            if item is not None and item.ladder is not None:
                continue

            if symbol in open_trailing:
                continue  # ja existe trailing stop nativo cobrindo o ativo

            pct = (
                item.trailing_stop_pct
                if item is not None and item.trailing_stop_pct is not None
                else self._default_pct
            )
            if pct is None:
                continue  # sem trailing configurado e sem default: nao protege

            trail_percent = pct * Decimal(100)  # fracao -> pontos %
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
    def _open_longs(ctx: StrategyContext):
        try:
            positions = ctx.broker.get_positions()
        except Exception:
            logger.warning("Falha ao consultar posicoes; assumindo nenhuma.")
            return []
        return [p for p in positions if p.qty > 0]

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
