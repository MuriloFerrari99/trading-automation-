"""Testes do PriceStream (cache de precos por WS) e da integracao no AlpacaBroker.

So a logica de cache/fallback — sem rede. O WS real (start/run) nao e exercitado aqui;
e best-effort e protegido. Garante que: cache fresco evita REST; cache velho/ausente
cai no REST e inscreve o simbolo; streaming desligado = 100% REST (comportamento atual).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from broker.price_stream import PriceStream, stream_key


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr("broker.price_stream.time.monotonic", c)
    return c


def test_stream_key_canoniza_acao_e_cripto():
    assert stream_key("aapl") == "AAPL"
    assert stream_key("BTC/USD") == "BTC/USD"  # cripto mantem a barra


def test_get_devolve_preco_fresco(clock):
    s = PriceStream("k", "s", max_age_seconds=5.0)
    s.update("AAPL", 190.25)
    assert s.get("aapl") == Decimal("190.25")  # case-insensitive via stream_key


def test_get_expira_preco_velho(clock):
    s = PriceStream("k", "s", max_age_seconds=5.0)
    s.update("AAPL", 190.25)
    clock.t += 6.0  # passou do max_age
    assert s.get("AAPL") is None


def test_get_ausente_e_none():
    s = PriceStream("k", "s")
    assert s.get("MSFT") is None


def test_update_cripto_keyed_por_par(clock):
    s = PriceStream("k", "s", max_age_seconds=5.0)
    s.update("BTC/USD", 65000)
    assert s.get("BTC/USD") == Decimal("65000")


class _Quote:
    def __init__(self, symbol, bid, ask):
        self.symbol, self.bid_price, self.ask_price = symbol, bid, ask


def test_quote_handler_grava_mid(clock):
    import asyncio

    s = PriceStream("k", "s", max_age_seconds=5.0)
    asyncio.run(s._on_quote(_Quote("BTC/USD", 63850.0, 63950.0)))
    assert s.get("BTC/USD") == Decimal("63900.0")  # mid (bid+ask)/2


def test_quote_handler_ignora_bid_ask_invalido(clock):
    import asyncio

    s = PriceStream("k", "s", max_age_seconds=5.0)
    asyncio.run(s._on_quote(_Quote("BTC/USD", 0.0, 63950.0)))  # bid invalido
    assert s.get("BTC/USD") is None


# --- integracao no AlpacaBroker (sem construir o SDK) -----------------------

class _FakeStream:
    def __init__(self, cached=None):
        self._cached = cached
        self.subscribed: list[str] = []

    def get(self, symbol):
        return self._cached

    def subscribe(self, symbol):
        self.subscribed.append(symbol)
        return True


class _Trade:
    def __init__(self, price):
        self.price = price


class _FakeData:
    def __init__(self):
        self.calls = 0

    def get_stock_latest_trade(self, req):
        self.calls += 1
        return {"AAPL": _Trade(191.0)}


def _broker_with(stream, data):
    from broker.alpaca_broker import AlpacaBroker

    b = object.__new__(AlpacaBroker)  # evita __init__/SDK
    b._stream_enabled = True
    b._price_stream = stream
    b._data = data
    b._crypto_data = None
    return b


def test_broker_usa_cache_quando_fresco():
    stream = _FakeStream(cached=Decimal("190.25"))
    data = _FakeData()
    b = _broker_with(stream, data)
    assert b.get_last_price("AAPL") == Decimal("190.25")
    assert data.calls == 0  # NAO bateu REST
    assert stream.subscribed == []  # ja estava no cache, nem precisa inscrever


def test_broker_cai_no_rest_e_inscreve_no_miss():
    stream = _FakeStream(cached=None)  # cache vazio
    data = _FakeData()
    b = _broker_with(stream, data)
    assert b.get_last_price("AAPL") == Decimal("191.0")
    assert data.calls == 1  # bateu REST (fallback)
    assert stream.subscribed == ["AAPL"]  # inscreveu p/ proxima leitura vir do cache
