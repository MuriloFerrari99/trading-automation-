"""Implementacao em memoria do BrokerClient para testes.

Permite exercitar agentes e estrategias sem nenhuma chamada de rede. Simula
contas, posicoes, precos e preenchimento imediato de ordens a mercado.
"""

from __future__ import annotations

import itertools
from datetime import date, timedelta
from decimal import Decimal

from broker.base import AccountInfo, BrokerClient, BrokerOrder
from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
)


class FakeBroker(BrokerClient):
    """Broker simulado, deterministico, para uso em testes."""

    def __init__(
        self,
        *,
        cash: Decimal = Decimal("100000"),
        prices: dict[str, Decimal] | None = None,
        options_level: int = 0,
        market_open: bool = True,
        today: date | None = None,
    ) -> None:
        self._cash = cash
        self._prices: dict[str, Decimal] = dict(prices or {})
        self._positions: dict[str, Position] = {}
        self._options_level = options_level
        self._market_open = market_open
        self._today = today or date(2026, 6, 15)
        self._order_ids = itertools.count(1)
        self.submitted: list[OrderIntent] = []  # historico p/ asserts em testes
        self.submitted_options: list[OptionOrderIntent] = []
        # Idempotencia + reconciliacao: ordens conhecidas por client_order_id.
        self._orders_by_cid: dict[str, BrokerOrder] = {}
        self._open_orders: dict[str, BrokerOrder] = {}  # por broker_order_id

    # --- helpers de teste ---------------------------------------------------
    def set_price(self, symbol: str, price: Decimal | str | float) -> None:
        self._prices[symbol.upper()] = Decimal(str(price))

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    def set_options_level(self, level: int) -> None:
        self._options_level = level

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
        # Idempotencia: mesmo client_order_id => devolve a ordem existente sem
        # reaplicar o fill (simula a rejeicao de duplicado pelo broker).
        cid = intent.client_order_id
        if cid is not None and cid in self._orders_by_cid:
            return self._as_result(self._orders_by_cid[cid])

        self.submitted.append(intent)
        broker_id = f"fake-{next(self._order_ids)}"

        # Trailing stop nativo: fica ABERTO no broker (nao preenche na hora).
        if intent.order_type == OrderType.TRAILING_STOP:
            order = BrokerOrder(
                broker_order_id=broker_id,
                client_order_id=cid,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                filled_qty=Decimal(0),
                status="new",
                order_type="trailing_stop",
            )
            if cid:
                self._orders_by_cid[cid] = order
            self._open_orders[broker_id] = order
            return self._as_result(order)

        # Demais tipos: preenchem imediatamente.
        fill_price = self._prices.get(intent.symbol, intent.limit_price or Decimal("1"))
        self._apply_fill(intent, fill_price)
        order = BrokerOrder(
            broker_order_id=broker_id,
            client_order_id=cid,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=intent.qty,
            filled_qty=intent.qty,
            status="filled",
            order_type=intent.order_type.value,
        )
        if cid:
            self._orders_by_cid[cid] = order
        return OrderResult(
            broker_order_id=broker_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            filled_avg_price=fill_price,
            status="filled",
        )

    @staticmethod
    def _as_result(order: BrokerOrder) -> OrderResult:
        return OrderResult(
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=OrderSide(order.side),
            qty=order.qty,
            filled_qty=order.filled_qty,
            filled_avg_price=None,
            status=order.status,
        )

    def seed_open_order(self, order: BrokerOrder) -> None:
        """Insere uma ordem aberta no broker (p/ testes de reconciliacao)."""
        self._open_orders[order.broker_order_id] = order
        if order.client_order_id:
            self._orders_by_cid[order.client_order_id] = order

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
        n = len(self._open_orders)
        self._open_orders.clear()
        return n

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self._open_orders.values())

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return self._orders_by_cid.get(client_order_id)

    def is_market_open(self) -> bool:
        return self._market_open

    # --- Opcoes -------------------------------------------------------------
    def select_option_contract(
        self,
        underlying: str,
        option_type: OptionType,
        target_strike: Decimal,
        *,
        min_dte: int,
        max_dte: int,
    ) -> OptionContract | None:
        # Sintetiza um contrato: strike arredondado ao inteiro mais proximo e
        # vencimento no meio da janela de DTE pedida. Suficiente para exercitar
        # a logica da Wheel sem uma cadeia de opcoes real.
        underlying = underlying.upper()
        strike = Decimal(round(target_strike))
        dte = (min_dte + max_dte) // 2
        expiration = self._today + timedelta(days=dte)
        occ = (
            f"{underlying}{expiration:%y%m%d}"
            f"{'P' if option_type == OptionType.PUT else 'C'}"
            f"{int(strike) * 1000:08d}"
        )
        return OptionContract(
            occ_symbol=occ,
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
        )

    def submit_option_order(self, intent: OptionOrderIntent) -> OrderResult:
        cid = intent.client_order_id
        if cid is not None and cid in self._orders_by_cid:
            return self._as_result(self._orders_by_cid[cid])  # idempotente

        self.submitted_options.append(intent)
        # Premio simulado: 1% do strike por acao (100 acoes por contrato).
        premium_per_share = intent.contract.strike * Decimal("0.01")
        premium = premium_per_share * Decimal(100) * intent.qty
        if intent.side == OrderSide.SELL:
            self._cash += premium  # vender premio credita caixa
        broker_id = f"fake-opt-{next(self._order_ids)}"
        if cid:
            self._orders_by_cid[cid] = BrokerOrder(
                broker_order_id=broker_id,
                client_order_id=cid,
                symbol=intent.contract.occ_symbol,
                side=intent.side.value,
                qty=intent.qty,
                filled_qty=intent.qty,
                status="filled",
                order_type="option",
            )
        return OrderResult(
            broker_order_id=broker_id,
            symbol=intent.contract.occ_symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            filled_avg_price=premium_per_share,
            status="filled",
        )
