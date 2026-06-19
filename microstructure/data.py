"""Downloader de dados de microestrutura da Binance (gratis, data.binance.vision).

Dois tipos de dado, com disponibilidade MUITO diferente no arquivo publico:

1) aggTrades (DISPONIVEL, gratis): trades agregados com price/quantity/timestamp
   e `is_buyer_maker`. Daqui sai o FLUXO DE ORDENS AGRESSOR (trade-flow): se
   is_buyer_maker=True, o comprador era o maker -> o AGRESSOR foi o vendedor
   (trade de venda); se False, o agressor foi o comprador (trade de compra).
   Esta e a unica fonte historica gratis de order-flow. Granularidade DIARIA
   (arquivos menores e disponiveis ate ~ontem); tambem ha mensal.

2) bookTicker (TOPO DE LIVRO): melhor bid/ask + tamanhos. Necessario para queue
   imbalance, microprice e OFI. >>> No arquivo publico atual o bookTicker
   historico esta 404 (futures UM e spot, mensal e diario) <<<. Entao aqui o
   downloader fica WIRED, mas marca PENDENTE se 404. A captura viavel e o
   COLETOR L2 AO VIVO de simulation.binance_book (forward capture), reaproveitado.

Tudo cacheia em data/microstructure_cache/. Em falha de rede/404 NAO inventa:
loga o aviso e retorna None (DADOS PENDENTES).

Convencao de aggressor (Binance): aggressor_side = +1 (buy) se is_buyer_maker
e False, -1 (sell) se True. signed_qty = side * quantity.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("microstructure.data")

CACHE_DIR = Path("data/microstructure_cache")

# data.binance.vision. aggTrades existe p/ spot e futures USDS-M (um). bookTicker
# (so futures, e atualmente 404 no arquivo) fica wired para quando/se voltar.
VISION = "https://data.binance.vision/data"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def _aggtrades_url(symbol: str, day: str, *, market: str = "um") -> str:
    """day no formato YYYY-MM-DD (diario). market: 'um' (futures USDS-M) ou 'spot'."""
    if market == "spot":
        base = f"{VISION}/spot/daily/aggTrades/{symbol}"
    else:
        base = f"{VISION}/futures/um/daily/aggTrades/{symbol}"
    return f"{base}/{symbol}-aggTrades-{day}.zip"


def _bookticker_url(symbol: str, day: str) -> str:
    """bookTicker diario de futures USDS-M. (Atualmente 404 no arquivo publico.)"""
    return f"{VISION}/futures/um/daily/bookTicker/{symbol}/{symbol}-bookTicker-{day}.zip"


def _aggtrades_csv_path(symbol: str, day: str, market: str) -> Path:
    return CACHE_DIR / f"{market}_{symbol}-aggTrades-{day}.csv"


def _bookticker_csv_path(symbol: str, day: str) -> Path:
    return CACHE_DIR / f"um_{symbol}-bookTicker-{day}.csv"


# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def _download_zip_csv(url: str, out: Path, *, force: bool, label: str) -> Path | None:
    """Baixa um .zip do vision, extrai o unico CSV interno e grava em `out`.

    Idempotente (cache hit se `out` existe e nao for force). Em 404/erro de rede
    NAO levanta — loga e retorna None.
    """
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        logger.info("cache hit %s -> %s", label, out.name)
        return out
    try:
        resp = requests.get(url, timeout=180)
        if resp.status_code == 404:
            logger.warning("404 (indisponivel no arquivo) %s", url)
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            data = zf.read(name)
        out.write_bytes(data)
        logger.info("baixado %s (%d bytes csv) -> %s", label, len(data), out.name)
        return out
    except Exception as exc:  # rede bloqueada / timeout / zip ruim
        logger.warning("falha ao baixar %s: %s", label, exc)
        return None


def download_aggtrades(
    symbol: str, day: str, *, market: str = "um", force: bool = False
) -> Path | None:
    """Baixa um DIA de aggTrades (YYYY-MM-DD). Retorna o caminho do CSV ou None."""
    url = _aggtrades_url(symbol, day, market=market)
    out = _aggtrades_csv_path(symbol, day, market)
    return _download_zip_csv(url, out, force=force, label=f"aggTrades {market} {symbol} {day}")


def download_bookticker(symbol: str, day: str, *, force: bool = False) -> Path | None:
    """Baixa um DIA de bookTicker (futures UM). Tipicamente 404 hoje -> None (PENDENTE)."""
    url = _bookticker_url(symbol, day)
    out = _bookticker_csv_path(symbol, day)
    return _download_zip_csv(url, out, force=force, label=f"bookTicker {symbol} {day}")


def download_universe_aggtrades(
    symbols: list[str], days: list[str], *, market: str = "um", force: bool = False
) -> dict[str, list[Path]]:
    """Baixa aggTrades p/ varios simbolos x dias. Mapa symbol -> [csv paths obtidos]."""
    out: dict[str, list[Path]] = {}
    for sym in symbols:
        paths: list[Path] = []
        for d in days:
            p = download_aggtrades(sym, d, market=market, force=force)
            if p is not None:
                paths.append(p)
        out[sym] = paths
    return out


def probe_bookticker(symbols: list[str], days: list[str]) -> dict[tuple[str, str], int | str]:
    """HEAD nos bookTicker p/ documentar disponibilidade no arquivo (sem baixar).

    Retorna {(symbol, day): status_code|err}. Usado para o report deixar EXPLICITO
    que bookTicker historico esta indisponivel (404) e os sinais de livro ficam
    pendentes do coletor ao vivo.
    """
    import requests

    out: dict[tuple[str, str], int | str] = {}
    for sym in symbols:
        for d in days:
            url = _bookticker_url(sym, d)
            try:
                r = requests.head(url, timeout=15, allow_redirects=True)
                out[(sym, d)] = int(r.status_code)
            except Exception as exc:  # noqa: BLE001
                out[(sym, d)] = f"ERR {type(exc).__name__}"
    return out


# ---------------------------------------------------------------------------
# PARSE DE aggTrades -> arrays numpy (sem pandas; arquivos grandes)
# ---------------------------------------------------------------------------

@dataclass
class TradeArrays:
    """Trades de um dia, ordenados por tempo. Vetores alinhados por indice.

    ts_ms        : timestamp do trade em ms (int64).
    price        : preco do trade (float64).
    qty          : quantidade (float64, >=0).
    side         : aggressor +1 (buy) / -1 (sell), de is_buyer_maker.
    signed_qty   : side * qty (fluxo assinado).
    """

    symbol: str
    day: str
    market: str
    ts_ms: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    side: np.ndarray
    signed_qty: np.ndarray

    @property
    def n(self) -> int:
        return int(self.ts_ms.size)


def _truthy_maker(v: str) -> bool:
    return v.strip().lower() in ("true", "1", "t", "yes")


def load_aggtrades_csv(path: Path, symbol: str, day: str, market: str) -> TradeArrays | None:
    """Le um CSV de aggTrades para arrays numpy. Tolerante a header ausente.

    Schema esperado (vision):
        agg_trade_id, price, quantity, first_trade_id, last_trade_id,
        transact_time(ms), is_buyer_maker
    """
    if not path.exists():
        return None
    ts: list[int] = []
    px: list[float] = []
    qt: list[float] = []
    sd: list[int] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        first = next(reader, None)
        if first is None:
            return None
        # detecta se a primeira linha e header
        price_i, qty_i, time_i, maker_i = 1, 2, 5, 6
        is_header = False
        try:
            float(first[price_i])
        except (ValueError, IndexError):
            is_header = True
        rows_iter = reader if is_header else _chain_first(first, reader)
        for row in rows_iter:
            try:
                p = float(row[price_i])
                q = float(row[qty_i])
                t = int(float(row[time_i]))
                maker = _truthy_maker(row[maker_i])
            except (ValueError, IndexError):
                continue
            ts.append(t)
            px.append(p)
            qt.append(q)
            # is_buyer_maker=True -> agressor foi o VENDEDOR (sell) -> -1
            sd.append(-1 if maker else 1)
    if not ts:
        return None
    ts_a = np.asarray(ts, dtype=np.int64)
    px_a = np.asarray(px, dtype=np.float64)
    qt_a = np.asarray(qt, dtype=np.float64)
    sd_a = np.asarray(sd, dtype=np.int8)
    # garante ordenacao temporal (vision ja vem ordenado, mas nao confiar cegamente)
    if not np.all(np.diff(ts_a) >= 0):
        order = np.argsort(ts_a, kind="stable")
        ts_a, px_a, qt_a, sd_a = ts_a[order], px_a[order], qt_a[order], sd_a[order]
    signed = sd_a.astype(np.float64) * qt_a
    return TradeArrays(
        symbol=symbol, day=day, market=market,
        ts_ms=ts_a, price=px_a, qty=qt_a, side=sd_a.astype(np.int64), signed_qty=signed,
    )


def _chain_first(first, reader):
    yield first
    yield from reader


def _day_from_path(path: Path) -> str:
    stem = path.stem  # e.g. um_BTCUSDT-aggTrades-2026-06-13
    if "-aggTrades-" in stem:
        return stem.split("-aggTrades-")[-1]
    return stem


def _market_symbol_from_path(path: Path) -> tuple[str, str]:
    stem = path.stem  # um_BTCUSDT-aggTrades-2026-06-13
    market = "um"
    rest = stem
    if "_" in stem:
        market, rest = stem.split("_", 1)
    sym = rest.split("-aggTrades-")[0] if "-aggTrades-" in rest else rest
    return market, sym


def discover_aggtrades(symbol: str, *, market: str = "um") -> list[Path]:
    if not CACHE_DIR.exists():
        return []
    return sorted(CACHE_DIR.glob(f"{market}_{symbol}-aggTrades-*.csv"))


def load_all_days(symbol: str, *, market: str = "um") -> list[TradeArrays]:
    """Carrega todos os dias de aggTrades em cache p/ o simbolo, um TradeArrays por dia."""
    out: list[TradeArrays] = []
    for p in discover_aggtrades(symbol, market=market):
        ta = load_aggtrades_csv(p, symbol, _day_from_path(p), market)
        if ta is not None and ta.n > 0:
            out.append(ta)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Downloader de microestrutura Binance (aggTrades real + bookTicker wired)"
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="lista por virgula")
    parser.add_argument("--days", default="", help="YYYY-MM-DD por virgula (diario)")
    parser.add_argument("--market", default="um", choices=["um", "spot"], help="futures UM ou spot")
    parser.add_argument("--bookticker", action="store_true", help="tenta tambem bookTicker (provavel 404)")
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo com cache")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    if not days:
        logger.error("--days exige YYYY-MM-DD[,...]")
        return 2

    got = download_universe_aggtrades(symbols, days, market=args.market, force=args.force)
    for sym, paths in got.items():
        logger.info("%s: %d/%d dias de aggTrades em cache", sym, len(paths), len(days))

    if args.bookticker:
        probe = probe_bookticker(symbols, days)
        n404 = sum(1 for v in probe.values() if v == 404)
        logger.info("bookTicker HEAD: %d/%d retornaram 404 (arquivo indisponivel)", n404, len(probe))
        for sym in symbols:
            for d in days:
                download_bookticker(sym, d, force=args.force)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
