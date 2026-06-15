"""Dados historicos reais da Alpaca + cache em parquet (doc 06 §4).

Busca barras DIARIAS ajustadas (adjustment=all) para um universo de acoes e
pares de cripto ("paridade entre moedas"), e cacheia por simbolo em
data/cache/*.parquet — assim o sweep de 10k+ janelas roda offline apos um
unico fetch. Usa dados ajustados por split/dividendo para avaliacao de
performance (doc 06 §4).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings

logger = logging.getLogger("simulation.data")

CACHE_DIR = Path("data/cache")


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}_1d.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _fetch_stock(symbols: list[str], start: datetime, settings) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment

    client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=start, adjustment=Adjustment.ALL,
    )
    bars = client.get_stock_bars(req)
    return _split_by_symbol(bars.df if hasattr(bars, "df") else None)


def _fetch_crypto(symbols: list[str], start: datetime, settings) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = CryptoHistoricalDataClient()  # dados de cripto sao publicos
    req = CryptoBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start
    )
    bars = client.get_crypto_bars(req)
    return _split_by_symbol(bars.df if hasattr(bars, "df") else None)


def _split_by_symbol(df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    # alpaca-py retorna MultiIndex (symbol, timestamp).
    for symbol in df.index.get_level_values(0).unique():
        sub = df.xs(symbol, level=0)[["open", "high", "low", "close"]].copy()
        out[str(symbol)] = sub
    return out


def fetch_daily(
    stock_symbols: list[str],
    crypto_symbols: list[str] | None = None,
    *,
    years: int = 4,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Retorna {symbol: DataFrame(open,high,low,close)}, usando cache em disco."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    crypto_symbols = crypto_symbols or []
    start = datetime.now(timezone.utc) - timedelta(days=365 * years)
    settings = get_settings()

    result: dict[str, pd.DataFrame] = {}
    to_fetch_stock: list[str] = []
    to_fetch_crypto: list[str] = []

    for sym in stock_symbols:
        p = _cache_path(sym)
        if p.exists() and not force:
            result[sym] = _read_cache(p)
        else:
            to_fetch_stock.append(sym)
    for sym in crypto_symbols:
        p = _cache_path(sym)
        if p.exists() and not force:
            result[sym] = _read_cache(p)
        else:
            to_fetch_crypto.append(sym)

    if to_fetch_stock:
        logger.info("Buscando %d acoes na Alpaca...", len(to_fetch_stock))
        for sym, df in _fetch_stock(to_fetch_stock, start, settings).items():
            _write_cache(df, _cache_path(sym))
            result[sym] = df
    if to_fetch_crypto:
        logger.info("Buscando %d pares de cripto na Alpaca...", len(to_fetch_crypto))
        for sym, df in _fetch_crypto(to_fetch_crypto, start, settings).items():
            _write_cache(df, _cache_path(sym))
            result[sym] = df

    return {s: d for s, d in result.items() if d is not None and not d.empty}


def to_ohlc_lists(df: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "open": df["open"].astype(float).tolist(),
        "high": df["high"].astype(float).tolist(),
        "low": df["low"].astype(float).tolist(),
        "close": df["close"].astype(float).tolist(),
    }
