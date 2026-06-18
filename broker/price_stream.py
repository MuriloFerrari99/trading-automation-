"""Cache de ultimos precos alimentado por WebSocket (alpaca-py) — mata o REST polling.

O loop de trading chama `broker.get_last_price(symbol)` repetidamente e cada chamada e
um GET REST. Este modulo mantem um cache `{symbol: (price, ts_monotonic)}` alimentado
em tempo real pelos streams de TRADES e QUOTES da Alpaca (StockDataStream p/ acoes,
CryptoDataStream p/ cripto), rodando em thread(s) daemon. Trade price e o "last price"
canonico; o mid dos quotes serve de preco vivo entre trades (na feed de cripto da
Alpaca os trades sao esparsos, mas os quotes fluem direto). `get(symbol)` devolve o
preco se fresco (< max_age); senao None, e o broker cai no REST — que tambem inscreve
o simbolo para as proximas leituras virem do cache.

Opt-in via ALPACA_STREAM_ENABLED. **Best-effort por design:** qualquer falha de WS NAO
pode quebrar o trading — o fallback REST no broker e sempre o caminho de seguranca. Por
isso toda operacao de rede aqui e protegida e o cache (update/get) e puro e testavel
sem rede.
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal

from core.models import is_crypto_symbol


def stream_key(symbol: str) -> str:
    """Chave de cache canonica: cripto mantem 'BASE/QUOTE'; acao vira UPPER.

    Casa com o `trade.symbol` que a Alpaca emite no stream (ticker em maiusculas
    para acoes; par com barra para cripto)."""
    return symbol if is_crypto_symbol(symbol) else symbol.upper()


class PriceStream:
    """Cache thread-safe de ultimos precos via WS da Alpaca. Best-effort, opt-in."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
        max_age_seconds: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed
        self._max_age = float(max_age_seconds)

        self._prices: dict[str, tuple[Decimal, float]] = {}
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()

        self._stock_stream = None
        self._crypto_stream = None
        self._threads: list[threading.Thread] = []

    # --- cache puro (sem rede, totalmente testavel) -------------------------
    def update(self, symbol: str, price) -> None:
        """Grava um preco no cache (chamado pelo handler do WS e por testes)."""
        with self._lock:
            self._prices[symbol] = (Decimal(str(price)), time.monotonic())

    def get(self, symbol: str) -> Decimal | None:
        """Ultimo preco se ainda fresco (< max_age); senao None (broker cai no REST)."""
        key = stream_key(symbol)
        with self._lock:
            entry = self._prices.get(key)
        if entry is None:
            return None
        price, ts = entry
        if time.monotonic() - ts > self._max_age:
            return None
        return price

    def is_subscribed(self, symbol: str) -> bool:
        with self._lock:
            return stream_key(symbol) in self._subscribed

    # --- ciclo de vida do WS (rede; protegido) ------------------------------
    async def _on_trade(self, trade) -> None:
        # handler async exigido pelo alpaca-py; trade price = "last price" canonico.
        self.update(getattr(trade, "symbol", None), getattr(trade, "price", None))

    async def _on_quote(self, quote) -> None:
        # quotes fluem MUITO mais que trades (a feed de cripto da Alpaca tem trades
        # esparsos); usamos o mid como preco vivo fresco. Latest-wins por timestamp,
        # entao um trade real sobrescreve o mid quando chega.
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        if bid and ask and bid > 0 and ask > 0:
            self.update(getattr(quote, "symbol", None), (bid + ask) / 2.0)

    def _ensure_stock_stream(self):
        if self._stock_stream is None:
            from alpaca.data.enums import DataFeed
            from alpaca.data.live import StockDataStream

            feed = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}.get(self._feed.lower(), DataFeed.IEX)
            self._stock_stream = StockDataStream(self._api_key, self._secret_key, feed=feed)
        return self._stock_stream

    def _ensure_crypto_stream(self):
        if self._crypto_stream is None:
            from alpaca.data.live import CryptoDataStream

            self._crypto_stream = CryptoDataStream(self._api_key, self._secret_key)
        return self._crypto_stream

    def _run_in_thread(self, stream) -> None:
        def _runner() -> None:
            try:
                stream.run()  # bloqueante; roda seu proprio event loop
            except Exception:  # WS caiu: silencioso, o broker segue no REST
                pass

        t = threading.Thread(target=_runner, name="alpaca-price-stream", daemon=True)
        t.start()
        self._threads.append(t)

    def subscribe(self, symbol: str) -> bool:
        """Inscreve o simbolo no stream certo (acao/cripto). True se a inscricao subiu.

        Idempotente. Best-effort: erro de rede vira False (broker continua no REST)."""
        key = stream_key(symbol)
        with self._lock:
            if key in self._subscribed:
                return True
            self._subscribed.add(key)
        try:
            if is_crypto_symbol(symbol):
                stream = self._ensure_crypto_stream()
                stream.subscribe_trades(self._on_trade, key)
                stream.subscribe_quotes(self._on_quote, key)
                if not getattr(self, "_crypto_started", False):
                    self._crypto_started = True
                    self._run_in_thread(stream)
            else:
                stream = self._ensure_stock_stream()
                stream.subscribe_trades(self._on_trade, key)
                stream.subscribe_quotes(self._on_quote, key)
                if not getattr(self, "_stock_started", False):
                    self._stock_started = True
                    self._run_in_thread(stream)
            return True
        except Exception:  # falhou ao inscrever: remove do set p/ tentar de novo depois
            with self._lock:
                self._subscribed.discard(key)
            return False

    def stop(self) -> None:
        for stream in (self._stock_stream, self._crypto_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
