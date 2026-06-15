"""Implementacao em memoria do BrokerClient para testes.

Permite exercitar agentes e estrategias sem nenhuma chamada de rede. Simula
contas, posicoes, precos e preenchimento imediato de ordens a mercado.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

from broker.base import AccountInfo, BrokerClient
from core.models import OrderIntent, OrderResult, OrderSide, Position


class FakeBroker(BrokerClient):
    """Broker simulado, deterministico, para uso em testes."""

    def __init__(
        self,
        *,
        cash: Decimal = Decimal("100000"),
        prices: dict[str, Decimal] | None = None,
        options_level: int = 0,
        market_open: bool = True,
    ) -> None:
        self._cash = cash
        self._prices: dict[str, Decimal] = dict(prices or {})
        self._positions: dict[str, Position] = {}
        self._options_level = options_level
        self._market_open = market_open
        self._order_ids = itertools.count(1)
        self.submitted: list[OrderIntent] = []  # historico p/ asserts em testes

    # --- helpers de teste ---------------------------------------------------
    def set_price(self, symbol: str, price: Decimal | str | float) -> None:
        self._prices[symbol.upper()] = Decimal(str(price))

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    def seed_position(self, symbol: str, qty: Decimal, avg_entry_price: Decimal) -> None:
        symbol = symbol.upper()
        self._positions[symbol] = Position(
            symbol=symbol, qty=qty, avg_entry_price=avg_entry_price
        )

    # --- BrokerClient -------------------------------------------------------
    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            (p.qty * self._prices.get(p.symbol, p.avg_entry_price) for p in self._positions.values()),
            Decimal(0),
        )
        return AccountInfo(
            cash=self._cash,
            buying_power=self._cash,
            equity=equity,
            options_level=self._options_level,
        )

    def get_positions(self) -> list[Position]:
        result = []
        for p in self._positions.values():
            price = self._prices.get(p.symbol)
            result.append(
                Position(
                    symbol=p.symbol,
                    qty=p.qty,
                    avg_entry_price=p.avg_entry_price,
                    current_price=price,
                )
            )
        return result

    def get_position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        p = self._positions.get(symbol)
        if p is None:
            return None
        return Position(
            symbol=p.symbol,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            current_price=self._prices.get(symbol),
        )

    def get_last_price(self, symbol: str) -> Decimal:
        symbol = symbol.upper()
        if symbol not in self._prices:
            raise KeyError(f"FakeBroker sem preco para {symbol}; use set_price()")
        return self._prices[symbol]

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        self.submitted.append(intent)
        fill_price = self._prices.get(intent.symbol, intent.limit_price or Decimal("1"))
        self._apply_fill(intent, fill_price)
        return OrderResult(
            broker_order_id=f"fake-{next(self._order_ids)}",
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            filled_avg_price=fill_price,
            status="filled",
        )

    def _apply_fill(self, intent: OrderIntent, fill_price: Decimal) -> None:
        symbol = intent.symbol
        signed = intent.qty if intent.side == OrderSide.BUY else -intent.qty
        existing = self._positions.get(symbol)
        if existing is None:
            new_qty = signed
            avg = fill_price
        else:
            new_qty = existing.qty + signed
            if intent.side == OrderSide.BUY:
                total_cost = existing.qty * existing.avg_entry_price + intent.qty * fill_price
                avg = total_cost / new_qty if new_qty != 0 else fill_price
            else:
                avg = existing.avg_entry_price
        cash_delta = -signed * fill_price
        self._cash += cash_delta
        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                symbol=symbol, qty=new_qty, avg_entry_price=avg
            )

    def cancel_all_orders(self) -> int:
        return 0  # FakeBroker preenche ordens imediatamente; nada pendente.

    def is_market_open(self) -> bool:
        return self._market_open
