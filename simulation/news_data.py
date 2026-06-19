"""Fetcher de noticias historicas da Alpaca (Benzinga) com cache em disco.

Point-in-time por construcao: cada item tem `created_at` (instante da publicacao),
entao o backtest so pode AGIR depois disso (sem look-ahead). Cacheia por simbolo
em data/news_cache/<sym>.csv para o backtest rodar offline apos um fetch unico.

Uso:
    uv run python -m signals.news_data --years 6           # universo de acoes
    uv run python -m signals.news_data --symbols AAPL,MSFT
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("simulation.news_data")

CACHE_DIR = Path("data/news_cache")


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.csv"


def fetch_symbol_news(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Baixa todas as noticias de UM simbolo no intervalo (pagina ate o fim)."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    from config.settings import get_settings

    s = get_settings()
    cli = NewsClient(s.alpaca_api_key, s.alpaca_secret_key)

    # O NewsClient AUTO-PAGINA quando `limit` nao e passado (passar limit vira
    # teto TOTAL e corta a serie). Buscamos em blocos anuais p/ limitar memoria.
    rows: list[dict] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=365), end)
        resp = cli.get_news(NewsRequest(
            symbols=symbol, start=chunk_start, end=chunk_end, include_content=False,
        ))
        items = resp.data.get("news", []) if hasattr(resp, "data") else list(resp)
        for it in items:
            rows.append({
                "created_at": it.created_at,
                "headline": it.headline or "",
                "symbols": "|".join(it.symbols or []),
                "source": getattr(it, "source", ""),
            })
        chunk_start = chunk_end
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.drop_duplicates(subset=["created_at", "headline"]).sort_values("created_at").reset_index(drop=True)
    return df


def load_or_fetch(symbol: str, *, years: int = 6, force: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(symbol)
    if p.exists() and not force:
        df = pd.read_csv(p)
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df
    start = datetime.now(timezone.utc) - timedelta(days=365 * years)
    end = datetime.now(timezone.utc)
    df = fetch_symbol_news(symbol, start, end)
    df.to_csv(p, index=False)
    logger.info("%s: %d noticias cacheadas (%s..%s)", symbol, len(df),
                df["created_at"].min() if not df.empty else "-",
                df["created_at"].max() if not df.empty else "-")
    return df


def fetch_universe(symbols: list[str], *, years: int = 6, force: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_or_fetch(sym, years=years, force=force)
            out[sym] = df
            logger.info("[%d/%d] %s: %d itens", i, len(symbols), sym, len(df))
            time.sleep(0.15)  # gentileza com o rate limit
        except Exception as exc:
            logger.warning("Falha em %s: %s", sym, exc)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch de noticias historicas (Alpaca/Benzinga)")
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--symbols", default="", help="lista separada por virgula; vazio = universo de acoes")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from simulation.run import DEFAULT_STOCKS
        syms = DEFAULT_STOCKS
    res = fetch_universe(syms, years=args.years, force=args.force)
    total = sum(len(d) for d in res.values())
    logger.info("Concluido: %d simbolos, %d noticias no total.", len(res), total)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
