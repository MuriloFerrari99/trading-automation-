"""Loader de precos diarios ajustados para o tribunal de RESIDUAL MOMENTUM.

Universo: equities US liquidas (membros estaveis do S&P 100 / mega-large caps)
baixadas GRATIS via yfinance (auto_adjust=True -> ja inclui dividendos/splits).
Cacheia uma matriz wide de closes ajustados em data/residmom_cache/*.csv para o
tribunal rodar offline apos um unico fetch.

Tambem baixa SPY (proxy de mercado) e, quando disponivel, fatores Fama-French
diarios GRATUITOS (Ken French data library) para o modelo de 3 fatores. Se o
ZIP da French falhar (rede), o tribunal cai p/ market-model (CAPM) so com SPY.

SEM LOOK-AHEAD: o loader so entrega closes ajustados; toda a logica de
sinal(t-1)->retorno(t+1) mora no tribunal.

Uso:
    uv run python -m data.residmom_data            # baixa universo + SPY (+FF se der)
    uv run python -m data.residmom_data --force
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("data.residmom_data")

CACHE_DIR = Path("data/residmom_cache")

# Universo liquido e estavel: large/mega caps com historico longo (>=10y) e
# baixo risco de survivorship grosseiro (todas vivas hoje; assumimos o vies e
# tratado como limitacao reportada, nao removido). ~40 nomes, setores variados.
UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "V",
    "PG", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "INTC", "CSCO",
    "KO", "PEP", "WMT", "MCD", "ABT", "TMO", "COST", "AVGO", "TXN", "QCOM",
    "NKE", "ORCL", "IBM", "GE", "CAT", "MMM", "CVX", "XOM", "UNH", "PFE",
    "MRK", "WFC", "C", "GS", "AMGN", "HON", "LOW", "UPS", "BA", "SBUX",
]
MARKET = "SPY"

PERIOD = "15y"
INTERVAL = "1d"
TRADING_DAYS = 252

FF_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def closes_path() -> Path:
    return CACHE_DIR / "closes_adj.csv"


def market_path() -> Path:
    return CACHE_DIR / "spy_adj.csv"


def ff_path() -> Path:
    return CACHE_DIR / "ff_factors_daily.csv"


def _yf_download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers, period=PERIOD, interval=INTERVAL,
        progress=False, auto_adjust=True, threads=True, group_by="column",
    )


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extrai a matriz wide de 'Close' (ja ajustado por auto_adjust=True)."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in set(lvl0) else lvl0[0]
        close = raw[field].copy()
    else:
        # download de 1 ticker
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]
    close.index = pd.to_datetime(close.index, utc=True, errors="coerce")
    close = close[~close.index.isna()].sort_index()
    return close.apply(pd.to_numeric, errors="coerce")


def load_closes(*, force: bool = False) -> pd.DataFrame:
    """Matriz wide (datas x tickers) de closes ajustados do universo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = closes_path()
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]
    raw = _yf_download(UNIVERSE)
    close = _extract_close(raw, UNIVERSE)
    if not close.empty:
        close.to_csv(p)
        logger.info("Universo: %d tickers x %d datas -> %s",
                    close.shape[1], close.shape[0], p)
    return close


def load_market(*, force: bool = False) -> pd.Series:
    """Serie de closes ajustados do SPY (proxy de mercado)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = market_path()
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        return df.iloc[:, 0]
    raw = _yf_download([MARKET])
    close = _extract_close(raw, [MARKET])
    if close.empty:
        return pd.Series(dtype=float)
    s = close.iloc[:, 0].rename(MARKET)
    s.to_frame().to_csv(p)
    logger.info("SPY: %d datas -> %s", len(s), p)
    return s


def load_ff_factors(*, force: bool = False) -> pd.DataFrame:
    """Fatores Fama-French diarios (Mkt-RF, SMB, HML, RF) em FRACAO (nao %).

    GRATUITO via Ken French data library. Se falhar (rede), retorna DataFrame
    vazio e o tribunal cai p/ CAPM (so SPY).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = ff_path()
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]
    try:
        req = urllib.request.Request(FF_DAILY_URL, headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            blob = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin-1")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FF download falhou (%s) -> CAPM fallback", exc)
        return pd.DataFrame()

    # O CSV tem cabecalho/rodape textual; a tabela diaria sao linhas YYYYMMDD,...
    rows = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
            try:
                rows.append([
                    pd.to_datetime(parts[0], format="%Y%m%d", utc=True),
                    *[float(x) / 100.0 for x in parts[1:]],
                ])
            except ValueError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "MktRF", "SMB", "HML", "RF"])
    df = df.set_index("date").sort_index()
    df.to_csv(p)
    logger.info("FF daily: %d datas (%s..%s) -> %s",
                len(df), df.index[0].date(), df.index[-1].date(), p)
    return df


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download precos US + SPY + FF (yfinance/French)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    closes = load_closes(force=args.force)
    mkt = load_market(force=args.force)
    ff = load_ff_factors(force=args.force)
    if closes.empty or mkt.empty:
        logger.error("DADOS PENDENTES. Rode: uv run python -m data.residmom_data --force")
        return 1
    logger.info("OK closes=%s SPY=%d FF=%s", closes.shape, len(mkt),
                "vazio" if ff.empty else f"{ff.shape}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
