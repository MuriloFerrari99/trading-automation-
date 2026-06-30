"""Implementacao real do BrokerClient sobre o SDK oficial `alpaca-py`.

Sempre instanciado em modo paper (paper=True). O guard em config.settings ja
garante que credenciais/endpoint apontem para paper; aqui reforcamos passando
paper=True explicitamente ao TradingClient.
"""

from __future__ import annotations

from decimal import Decimal

from broker.base import AccountInfo, BrokerClient, BrokerOrder, MarketClockInfo
from core.models import is_crypto_symbol
from config.settings import Settings
from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)


def _to_decimal(value) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal(0)


def _normalize_crypto_symbol(symbol: str) -> str:
    """Normaliza um simbolo de cripto SEM barra ('BTCUSD') para a forma 'BASE/USD'.

    A Alpaca pode devolver posicoes de cripto como 'BTCUSD' (sem barra), enquanto
    o resto do sistema (e o get_positions/ordem) usa 'BTC/USD'. Sem normalizar, o
    lookup de peso atual no rebalancer ERRA e a posicao e relida como 0 -> compra
    dobrada (MF-3c). So toca simbolos que TERMINEM em uma quote conhecida e NAO
    tenham barra; qualquer outra coisa (acoes) passa intacta."""
    if "/" in symbol:
        return symbol
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[: -len(quote)]}/{quote}"
    return symbol


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
        # Cliente de dados de CRIPTO construido sob demanda (MF-3): mantem o custo
        # fora do caminho quando nao ha cripto no universo.
        self._crypto_data = None

        # Streaming de precos por WS (opt-in): alimenta um cache que get_last_price
        # le antes de bater REST. Best-effort — desligado, tudo segue no REST.
        self._stream_enabled = bool(getattr(settings, "stream_enabled", False))
        self._price_stream = None

    def _crypto_data_client(self):
        """CryptoHistoricalDataClient lazy (MF-3a/3b). Crypto market data na Alpaca
        e publico; passamos as chaves por consistencia (sao aceitas)."""
        if self._crypto_data is None:
            from alpaca.data.historical import CryptoHistoricalDataClient

            self._crypto_data = CryptoHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
        return self._crypto_data

    def get_account(self) -> AccountInfo:
        acct = self._trading.get_account()
        options_level = int(getattr(acct, "options_approved_level", 0) or 0)
        # Pool NAO-MARGINAVEL (cripto cash-only): a Alpaca expoe o campo
        # `non_marginable_buying_power` (caixa liquidado contra o qual a cripto e
        # avaliada). Se o SDK/conta nao trouxer, cai no `cash` (a aproximacao
        # correta — caixa liquidado). E o que o pre-trade usa p/ barrar cripto.
        nmbp_raw = getattr(acct, "non_marginable_buying_power", None)
        nmbp = _to_decimal(nmbp_raw) if nmbp_raw is not None else _to_decimal(acct.cash)
        return AccountInfo(
            cash=_to_decimal(acct.cash),
            buying_power=_to_decimal(acct.buying_power),
            equity=_to_decimal(acct.equity),
            currency=getattr(acct, "currency", "USD"),
            options_level=options_level,
            non_marginable_buying_power=nmbp,
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
        # MF-3c: a Alpaca pode devolver posicoes de cripto como 'BTCUSD' (sem
        # barra). Normalizamos para 'BTC/USD' para que o lookup de peso atual no
        # rebalancer (positions.get('BTC/USD')) ACERTE e nao releia a posicao como
        # 0 (o que dobraria a compra). Acoes/ETFs passam intactos.
        symbol = _normalize_crypto_symbol(p.symbol)
        return Position(
            symbol=symbol,
            qty=_to_decimal(p.qty),
            avg_entry_price=_to_decimal(p.avg_entry_price),
            current_price=_to_decimal(getattr(p, "current_price", None))
            if getattr(p, "current_price", None) is not None
            else None,
        )

    def _price_stream_client(self):
        """PriceStream lazy (opt-in). None se o streaming estiver desligado."""
        if not self._stream_enabled:
            return None
        if self._price_stream is None:
            from broker.price_stream import PriceStream

            self._price_stream = PriceStream(
                self._settings.alpaca_api_key,
                self._settings.alpaca_secret_key,
                feed=getattr(self._settings, "stream_feed", "iex"),
                max_age_seconds=getattr(self._settings, "stream_max_age_seconds", 5.0),
            )
        return self._price_stream

    def get_last_price(self, symbol: str) -> Decimal:
        # WS-first: se o streaming estiver ligado e houver preco fresco no cache,
        # devolve sem bater REST. Senao cai no REST (caminho de seguranca) e inscreve
        # o simbolo para as proximas leituras virem do cache.
        stream = self._price_stream_client()
        if stream is not None:
            cached = stream.get(symbol)
            if cached is not None:
                return cached

        # MF-3a: cripto (par com '/') vai pelo feed de cripto; acoes pelo de acoes.
        # Sem isso, get_stock_latest_trade('BTC/USD') quebra e o ativo de cripto
        # cai em "sem preco" no rebalancer (nunca negociado) e mismarca o NAV.
        if is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoLatestTradeRequest

            req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            latest = self._crypto_data_client().get_crypto_latest_trade(req)
            price = _to_decimal(latest[symbol].price)
        else:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=symbol.upper())
            latest = self._data.get_stock_latest_trade(req)
            price = _to_decimal(latest[symbol.upper()].price)

        if stream is not None:
            stream.subscribe(symbol)  # proxima leitura deste simbolo vem do cache
        return price

    # Barras de mercado aproximadas por dia de pregao, por timeframe — usado p/
    # calcular uma janela `start` ampla o bastante para devolver `limit` barras.
    _BARS_PER_DAY = {"1min": 390, "5min": 78, "15min": 26, "1hour": 7, "1day": 1}

    def get_bars(self, symbol: str, limit: int = 60, *, timeframe: str = "1Day") -> list[Decimal]:
        # MF-3b: cripto (par com '/') vai pelo feed de barras de cripto. Mesmo
        # contrato (closes em ordem cronologica), coerente com a fonte do backtest
        # (closes diarios). Acoes pelo feed de acoes.
        if is_crypto_symbol(symbol):
            from alpaca.data.requests import CryptoBarsRequest

            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._timeframe(timeframe),
                limit=limit,
                start=self._lookback_start(timeframe, limit),
            )
            bars = self._crypto_data_client().get_crypto_bars(req)
            data = getattr(bars, "data", {}).get(symbol, []) if bars else []
            return [_to_decimal(b.close) for b in data]

        from alpaca.data.requests import StockBarsRequest

        symbol = symbol.upper()
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(timeframe),
            limit=limit,
            start=self._lookback_start(timeframe, limit),
        )
        bars = self._data.get_stock_bars(req)
        data = getattr(bars, "data", {}).get(symbol, []) if bars else []
        return [_to_decimal(b.close) for b in data]

    @classmethod
    def _lookback_start(cls, timeframe: str, limit: int):
        """`start` ASSAZ amplo p/ a Alpaca devolver `limit` barras.

        Sem `start`, a API retorna so a janela recente (ex.: 1 barra diaria) — o
        que deixava o classificador de regime sempre em UNKNOWN ao vivo. Estima
        os dias de calendario a partir das barras/dia do timeframe, com folga
        para fins de semana/feriados."""
        from datetime import datetime, timedelta, timezone

        bpd = cls._BARS_PER_DAY.get(timeframe.lower(), 1)
        trading_days = limit / bpd
        calendar_days = int(trading_days * 1.6) + 5  # folga p/ pregao fechado
        return datetime.now(timezone.utc) - timedelta(days=calendar_days)

    @staticmethod
    def _timeframe(tf: str):
        """Traduz uma string portavel ("1Min"/"5Min"/"1Hour"/"1Day") para o
        TimeFrame do alpaca-py. Default diario se desconhecido."""
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        table = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day,
        }
        return table.get(tf.lower(), TimeFrame.Day)

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
            client_order_id=intent.client_order_id,
        )

    def _build_order_request(self, intent: OrderIntent):
        from alpaca.trading.enums import OrderSide as ASide
        from alpaca.trading.enums import TimeInForce as ATIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
            TrailingStopOrderRequest,
        )

        side = ASide.BUY if intent.side == OrderSide.BUY else ASide.SELL
        # MF-3d: a Alpaca REJEITA ordens de cripto com TIF=day — cripto exige GTC.
        # Forcamos GTC quando o simbolo for cripto (par com '/'), independentemente
        # do TIF do intent (que vem DAY por default). Acoes seguem o intent.
        if is_crypto_symbol(intent.symbol):
            tif = ATIF.GTC
        else:
            tif = ATIF.GTC if intent.time_in_force == TimeInForce.GTC else ATIF.DAY
        qty = float(intent.qty)
        # client_order_id garante idempotencia: retry com o mesmo id e rejeitado.
        common = dict(
            symbol=intent.symbol,
            qty=qty,
            side=side,
            time_in_force=tif,
            client_order_id=intent.client_order_id,
        )

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
        if intent.order_type == OrderType.TRAILING_STOP:
            # Trailing stop nativo (server-side): sobrevive a crash do bot.
            return TrailingStopOrderRequest(
                trail_percent=float(intent.trail_percent), **common
            )
        raise ValueError(f"order_type nao suportado: {intent.order_type}")

    def cancel_all_orders(self) -> int:
        responses = self._trading.cancel_orders()
        return len(responses) if responses else 0

    def get_open_orders(self) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._trading.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        return [self._to_broker_order(o) for o in (orders or [])]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        from alpaca.common.exceptions import APIError

        try:
            o = self._trading.get_order_by_client_id(client_order_id)
        except APIError:
            return None
        return self._to_broker_order(o) if o else None

    @staticmethod
    def _to_broker_order(o) -> BrokerOrder:
        fap = getattr(o, "filled_avg_price", None)
        return BrokerOrder(
            broker_order_id=str(o.id),
            client_order_id=getattr(o, "client_order_id", None),
            symbol=o.symbol,
            side=str(getattr(o.side, "value", o.side)),
            qty=_to_decimal(getattr(o, "qty", 0)),
            filled_qty=_to_decimal(getattr(o, "filled_qty", 0)),
            status=str(getattr(o.status, "value", o.status)),
            order_type=str(getattr(o, "order_type", getattr(o, "type", "unknown"))),
            filled_avg_price=_to_decimal(fap) if fap is not None else None,
        )

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)

    def get_clock(self) -> MarketClockInfo:
        clock = self._trading.get_clock()
        return MarketClockInfo(
            is_open=bool(clock.is_open),
            next_open=getattr(clock, "next_open", None),
            next_close=getattr(clock, "next_close", None),
            timestamp=getattr(clock, "timestamp", None),
        )

    # --- Opcoes -------------------------------------------------------------
    # NOTA: caminho de opcoes ainda NAO exercitado contra a API real. A
    # selecao de contrato usa o endpoint de option contracts do alpaca-py.
    def select_option_contract(
        self,
        underlying: str,
        option_type: OptionType,
        target_strike: Decimal,
        *,
        min_dte: int,
        max_dte: int,
    ) -> OptionContract | None:
        from datetime import timedelta

        from alpaca.trading.enums import ContractType
        from alpaca.trading.requests import GetOptionContractsRequest

        today = self._trading.get_clock().timestamp.date()
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying.upper()],
            type=ContractType.PUT if option_type == OptionType.PUT else ContractType.CALL,
            expiration_date_gte=today + timedelta(days=min_dte),
            expiration_date_lte=today + timedelta(days=max_dte),
            limit=200,
        )
        contracts = self._trading.get_option_contracts(req).option_contracts or []
        if not contracts:
            return None
        # Escolhe o strike mais proximo do alvo.
        best = min(contracts, key=lambda c: abs(_to_decimal(c.strike_price) - target_strike))
        return OptionContract(
            occ_symbol=best.symbol,
            underlying=underlying.upper(),
            option_type=option_type,
            strike=_to_decimal(best.strike_price),
            expiration=best.expiration_date,
        )

    def submit_option_order(self, intent: OptionOrderIntent) -> OrderResult:
        from alpaca.trading.enums import OrderSide as ASide
        from alpaca.trading.enums import TimeInForce as ATIF
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = ASide.BUY if intent.side == OrderSide.BUY else ASide.SELL
        common = dict(
            symbol=intent.contract.occ_symbol,
            qty=float(intent.qty),
            side=side,
            time_in_force=ATIF.DAY,
            client_order_id=intent.client_order_id,
        )
        # LIMIT quando houver limit_price (premio-alvo); senao MARKET.
        if intent.order_type == OrderType.LIMIT and intent.limit_price is not None:
            req = LimitOrderRequest(limit_price=float(intent.limit_price), **common)
        else:
            req = MarketOrderRequest(**common)
        order = self._trading.submit_order(req)
        return OrderResult(
            broker_order_id=str(order.id),
            symbol=intent.contract.occ_symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=_to_decimal(getattr(order, "filled_qty", 0)),
            filled_avg_price=(
                _to_decimal(order.filled_avg_price)
                if getattr(order, "filled_avg_price", None) is not None
                else None
            ),
            status=str(getattr(order, "status", "accepted")),
            client_order_id=intent.client_order_id,
        )
