"""Loader de OHLC historico de FOREX (majors) com cache em disco.

Baseline do tribunal FIMATHE em forex. Baixa OHLC GRATUITO via yfinance
(tickers `EURUSD=X` etc.), normaliza para colunas minusculas
(open/high/low/close) e cacheia por par+timeframe em data/forex_cache/*.csv —
assim o tribunal roda offline apos um unico fetch.

TIMEFRAMES
- D1 (diario): `period=5y` -> ~1300 barras/par. Suficiente e robusto p/ baseline.
- H1 (horario): yfinance limita intraday a ~730 dias -> ~17k barras/par.
- H4: NAO e intervalo nativo confiavel no yfinance (gera barras parciais). Aqui
  e DERIVADO por resample causal de H1 (4h, label/closed='left'), preservando
  open/high/low/close corretos.

SEM LOOK-AHEAD: o loader so entrega OHLC bruto; a logica de "sinal no fechamento,
entrada no open seguinte" mora no tribunal (simulation/fimathe_forex.py).

PIP_SIZE por par (para o tribunal): 0.0001 nos majors; 0.01 nos pares JPY.

Uso:
    uv run python -m data.forex_data                 # baixa universo default (D1 + H1)
    uv run python -m data.forex_data --timeframes D1
    uv run python -m data.forex_data --force         # ignora cache

Se a rede bloquear, o fetch lanca e o tribunal reporta "DADOS PENDENTES" com o
comando exato acima — sem inventar numeros.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("data.forex_data")

CACHE_DIR = Path("data/forex_cache")

# Universo: majors líquidos e menos voláteis. yfinance usa o sufixo "=X".
# pip_size: 0.0001 majors; 0.01 nos pares com JPY (cotação ~150, 1 pip = 0.01).
PAIRS: dict[str, dict] = {
    "EURUSD": {"yf": "EURUSD=X", "pip_size": 0.0001},
    "EURGBP": {"yf": "EURGBP=X", "pip_size": 0.0001},  # menor vol
    "USDCHF": {"yf": "USDCHF=X", "pip_size": 0.0001},  # menor vol
    "GBPUSD": {"yf": "GBPUSD=X", "pip_size": 0.0001},
    "USDJPY": {"yf": "USDJPY=X", "pip_size": 0.01},
}
DEFAULT_PAIRS = list(PAIRS.keys())

# yfinance: interval nativo -> (interval, period). H4 e derivado de H1.
_YF_NATIVE = {
    "D1": ("1d", "5y"),
    "H1": ("1h", "730d"),  # teto de intraday do yfinance
}
TIMEFRAMES = ("D1", "H4", "H1")

# Barras por ano (anualizacao do Sharpe). Forex ~5 dias/semana, sessao ~24h.
TRADING_DAYS_FX = 252
PERIODS_PER_YEAR = {
    "D1": 252,
    "H4": 252 * 6,   # ~6 barras de 4h por dia de pregao
    "H1": 252 * 24,  # ~24 barras horarias por dia de pregao
}


def cache_path(pair: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{pair}_{timeframe}.csv"


def pip_size(pair: str) -> float:
    return PAIRS[pair]["pip_size"]


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance -> DataFrame(open,high,low,close[,volume]) com índice datetime UTC."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # yfinance devolve MultiIndex de colunas (campo, ticker) p/ download de 1 ticker.
    if isinstance(out.columns, pd.MultiIndex):
        try:
            out = out.xs(ticker, axis=1, level=1)
        except KeyError:
            out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower() for c in out.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].copy()
    # índice -> UTC tz-aware (intraday vem em Europe/London no yf).
    idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = idx
    out = out[~out.index.isna()]
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    # Forex em fim de semana às vezes traz barras de range zero (mercado fechado).
    flat = (out["high"] == out["low"]) & (out["open"] == out["close"])
    out = out[~flat]
    return out.sort_index()


def _resample_h4(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 -> H4 por resample CAUSAL (4h, alinhado à esquerda). Sem look-ahead:
    a barra rotulada em t cobre [t, t+4h) e só fecha ao fim do bloco."""
    if h1.empty:
        return h1
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in h1.columns:
        agg["volume"] = "sum"
    out = h1.resample("4h", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def _yf_download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=False, threads=False,
    )


def load_pair(
    pair: str, timeframe: str, *, force: bool = False, write: bool = True
) -> pd.DataFrame:
    """OHLC de UM par/timeframe. Usa cache; baixa se ausente ou `force`.

    Levanta a exceção de rede do yfinance se o download falhar (o tribunal trata
    isso como DADOS PENDENTES). Não inventa dados.
    """
    if pair not in PAIRS:
        raise ValueError(f"Par desconhecido: {pair}. Conhecidos: {DEFAULT_PAIRS}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Timeframe desconhecido: {timeframe}. {TIMEFRAMES}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(pair, timeframe)
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]

    ticker = PAIRS[pair]["yf"]
    if timeframe == "H4":
        # Deriva de H1 (nativo). Reusa o cache de H1 se houver.
        h1 = load_pair(pair, "H1", force=force, write=write)
        df = _resample_h4(h1)
    else:
        interval, period = _YF_NATIVE[timeframe]
        raw = _yf_download(ticker, interval, period)
        df = _normalize(raw, ticker)

    if write and not df.empty:
        df.to_csv(p)
        logger.info(
            "%s %s: %d barras (%s..%s) -> %s",
            pair, timeframe, len(df),
            df.index[0].date() if len(df) else "-",
            df.index[-1].date() if len(df) else "-", p,
        )
    return df


def load_universe(
    pairs: list[str] | None = None,
    timeframes: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[tuple[str, str], pd.DataFrame]:
    """{(par, timeframe): DataFrame}. Pula pares que falharem, logando o erro."""
    pairs = pairs or DEFAULT_PAIRS
    timeframes = list(timeframes or TIMEFRAMES)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for tf in timeframes:
        for pair in pairs:
            try:
                df = load_pair(pair, tf, force=force)
                if not df.empty:
                    out[(pair, tf)] = df
            except Exception as exc:  # noqa: BLE001 — rede/yf; reporta e segue
                logger.warning("Falha em %s %s: %s", pair, tf, exc)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download de OHLC forex (yfinance) + cache")
    parser.add_argument("--pairs", default="", help="CSV; vazio = universo default")
    parser.add_argument(
        "--timeframes", default="D1,H1",
        help="CSV de D1,H4,H1 (H4 é derivado de H1). Default: D1,H1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    pairs = (
        [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
        if args.pairs else DEFAULT_PAIRS
    )
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    res = load_universe(pairs, tfs, force=args.force)
    if not res:
        logger.error(
            "NENHUM dado baixado. Verifique a rede. Comando: "
            "uv run python -m data.forex_data --force"
        )
        return 1
    total = sum(len(d) for d in res.values())
    logger.info("Concluido: %d series, %d barras no total.", len(res), total)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
