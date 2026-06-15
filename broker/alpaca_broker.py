"""Implementacao real do BrokerClient sobre o SDK oficial `alpaca-py`.

Sempre instanciado em modo paper (paper=True). O guard em config.settings ja
garante que credenciais/endpoint apontem para paper; aqui reforcamos passando
paper=True explicitamente ao TradingClient.
"""

from __future__ import annotations

from decimal import Decimal

from broker.base import AccountInfo, BrokerClient
from config.settings import Settings
from core.models import (
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)


def _to_decimal(value) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal(0)


class AlpacaBroker(BrokerClient):
    """Adaptador da Alpaca para a interface BrokerClient (paper trading)."""

    def __init__(self, settings: Settings) -> None:
        # Imports tardios: mantem o SDK fora do caminho dos testes que usam
        # apenas o FakeBroker e evita custo de import quando nao necessario.
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self._settings = settings
        self._trading = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,  # reforco do guard paper-only
        )
        self._data = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_account(self) -> AccountInfo:
        acct = self._trading.get_account()
        options_level = int(getattr(acct, "options_approved_level", 0) or 0)
        return AccountInfo(
            cash=_to_decimal(acct.cash),
            buying_power=_to_decimal(acct.buying_power),
            equity=_to_decimal(acct.equity),
            currency=getattr(acct, "currency", "USD"),
            options_level=options_level,
        )

    def get_positions(self) -> list[Position]:
        return [self._to_position(p) for p in self._trading.get_all_positions()]

    def get_position(self, symbol: str) -> Position | None:
        from alpaca.common.exceptions import APIError

        try:
            p = self._trading.get_open_position(symbol.upper())
        except APIError:
            return None
        return self._to_position(p)

    @staticmethod
    def _to_position(p) -> Position:
        return Position(
            symbol=p.symbol,
            qty=_to_decimal(p.qty),
            avg_entry_price=_to_decimal(p.avg_entry_price),
            current_price=_to_decimal(getattr(p, "current_price", None))
            if getattr(p, "current_price", None) is not None
            else None,
        )

    def get_last_price(self, symbol: str) -> Decimal:
        from alpaca.data.requests import StockLatestTradeRequest

        req = StockLatestTradeRequest(symbol_or_symbols=symbol.upper())
        latest = self._data.get_stock_latest_trade(req)
        return _to_decimal(latest[symbol.upper()].price)

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        order_request = self._build_order_request(intent)
        order = self._trading.submit_order(order_request)
        return OrderResult(
            broker_order_id=str(order.id),
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=_to_decimal(getattr(order, "filled_qty", 0)),
            filled_avg_price=(
                _to_decimal(order.filled_avg_price)
                if getattr(order, "filled_avg_price", None) is not None
                else None
            ),
            status=str(getattr(order, "status", "accepted")),
        )

    def _build_order_request(self, intent: OrderIntent):
        from alpaca.trading.enums import OrderSide as ASide
        from alpaca.trading.enums import TimeInForce as ATIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = ASide.BUY if intent.side == OrderSide.BUY else ASide.SELL
        tif = ATIF.GTC if intent.time_in_force == TimeInForce.GTC else ATIF.DAY
        qty = float(intent.qty)
        common = dict(symbol=intent.symbol, qty=qty, side=side, time_in_force=tif)

        if intent.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        if intent.order_type == OrderType.LIMIT:
            return LimitOrderRequest(limit_price=float(intent.limit_price), **common)
        if intent.order_type == OrderType.STOP:
            return StopOrderRequest(stop_price=float(intent.stop_price), **common)
        if intent.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=float(intent.stop_price),
                limit_price=float(intent.limit_price),
                **common,
            )
        raise ValueError(f"order_type nao suportado: {intent.order_type}")

    def cancel_all_orders(self) -> int:
        responses = self._trading.cancel_orders()
        return len(responses) if responses else 0

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)
