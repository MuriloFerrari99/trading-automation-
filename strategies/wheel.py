"""Wheel Strategy (Nivel 3 — opcoes).

Fluxo classico da "roda":
1. Sem acoes do ativo: vende CASH-SECURED PUT ~otm_pct abaixo do preco (coleta
   premio; se exercido, entra comprado).
2. Com acoes (ex: apos assignment do put): vende COVERED CALL ~otm_pct acima do
   preco de custo (coleta premio; se exercido, sai vendido com lucro).

GATE EXPLICITO (requisito do briefing): antes de gerar QUALQUER ordem de
opcoes, verifica:
  - nivel de aprovacao de opcoes da conta (options_level >= REQUIRED);
  - liquidez suficiente p/ garantir o put (cash >= strike * 100 * contratos).
Se o gate reprovar, a estrategia apenas loga o motivo e nao gera ordem.

Defesa em profundidade: o Executor reaplica o gate de nivel antes de enviar.

Cada perna e marcada em `state` (wheel_open_put/call:SYM) para nao reenviar a
cada ciclo do Monitor. Use `WheelStrategy.reset(symbol, state)` para re-armar.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from core.models import OptionOrderIntent, OptionType, OrderSide
from strategies.base import Strategy, StrategyContext

logger = logging.getLogger("strategy.wheel")

# Nivel minimo de opcoes p/ vender cash-secured puts e covered calls (Alpaca
# trata isso como Nivel 1). options_level=0 => sem permissao.
REQUIRED_OPTIONS_LEVEL = 1
SHARES_PER_CONTRACT = Decimal(100)


def open_put_key(symbol: str) -> str:
    return f"wheel_open_put:{symbol.upper()}"


def open_call_key(symbol: str) -> str:
    return f"wheel_open_call:{symbol.upper()}"


class WheelStrategy(Strategy):
    name = "wheel"

    def __init__(self, *, required_options_level: int = REQUIRED_OPTIONS_LEVEL) -> None:
        self._required_level = required_options_level

    def evaluate(self, ctx: StrategyContext) -> list[OptionOrderIntent]:
        intents: list[OptionOrderIntent] = []

        items = [i for i in ctx.watchlist.items if i.wheel is not None]
        if not items:
            return intents

        # Opcoes negociam no pregao de acoes; fora dele, nao gera ordem.
        if not ctx.equity_market_open:
            logger.info("Wheel: pregao fechado — nenhuma ordem de opcoes neste ciclo.")
            return intents

        account = ctx.broker.get_account()

        # GATE 1: nivel de opcoes da conta.
        if account.options_level < self._required_level:
            logger.warning(
                "Wheel desabilitada: nivel de opcoes da conta (%d) < requerido (%d).",
                account.options_level, self._required_level,
            )
            return intents

        for item in items:
            symbol = item.symbol
            wheel = item.wheel
            contracts = Decimal(wheel.contracts)
            shares_needed = contracts * SHARES_PER_CONTRACT

            position = ctx.broker.get_position(symbol)
            shares_held = position.qty if position else Decimal(0)

            if shares_held >= shares_needed:
                intent = self._maybe_covered_call(ctx, item, position, contracts)
            else:
                price = ctx.broker.get_last_price(symbol)
                intent = self._maybe_cash_secured_put(ctx, item, account, price, contracts)

            if intent is not None:
                intents.append(intent)

        return intents

    def _maybe_cash_secured_put(self, ctx, item, account, price, contracts) -> OptionOrderIntent | None:
        symbol = item.symbol
        if ctx.state.get(open_put_key(symbol)) is not None:
            return None  # ja ha put aberta neste ativo

        target_strike = price * (Decimal(1) - item.wheel.otm_pct)

        # GATE 2: liquidez p/ garantir o put (cash-secured).
        required_collateral = target_strike * SHARES_PER_CONTRACT * contracts
        if account.cash < required_collateral:
            logger.warning(
                "Wheel %s: liquidez insuficiente p/ cash-secured put "
                "(cash=%s < colateral~%s). Pulando.",
                symbol, account.cash, required_collateral,
            )
            return None

        contract = ctx.broker.select_option_contract(
            symbol, OptionType.PUT, target_strike,
            min_dte=item.wheel.min_dte, max_dte=item.wheel.max_dte,
        )
        if contract is None:
            logger.info("Wheel %s: nenhum contrato de PUT elegivel.", symbol)
            return None

        logger.info(
            "Wheel %s: vendendo %s PUT strike=%s exp=%s (alvo %s)",
            symbol, contracts, contract.strike, contract.expiration, target_strike,
        )
        ctx.state.set(open_put_key(symbol), contract.occ_symbol)
        return OptionOrderIntent(
            contract=contract, side=OrderSide.SELL, qty=contracts, strategy=self.name
        )

    def _maybe_covered_call(self, ctx, item, position, contracts) -> OptionOrderIntent | None:
        symbol = item.symbol
        if ctx.state.get(open_call_key(symbol)) is not None:
            return None  # ja ha call aberta

        cost_basis = position.avg_entry_price
        target_strike = cost_basis * (Decimal(1) + item.wheel.otm_pct)

        contract = ctx.broker.select_option_contract(
            symbol, OptionType.CALL, target_strike,
            min_dte=item.wheel.min_dte, max_dte=item.wheel.max_dte,
        )
        if contract is None:
            logger.info("Wheel %s: nenhum contrato de CALL elegivel.", symbol)
            return None

        logger.info(
            "Wheel %s: vendendo %s COVERED CALL strike=%s exp=%s (custo=%s)",
            symbol, contracts, contract.strike, contract.expiration, cost_basis,
        )
        ctx.state.set(open_call_key(symbol), contract.occ_symbol)
        return OptionOrderIntent(
            contract=contract, side=OrderSide.SELL, qty=contracts, strategy=self.name
        )

    @staticmethod
    def reset(symbol: str, state) -> None:
        """Re-arma a wheel de um ativo (limpa pernas marcadas)."""
        state.delete(open_put_key(symbol))
        state.delete(open_call_key(symbol))
