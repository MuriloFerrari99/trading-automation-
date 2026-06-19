"""Microestrutura de mercado da Binance: spread historico + coletor L2 ao vivo.

Base para (a) modelar a versao MAKER do tribunal de momentum com SPREAD real e
(b) avaliar viabilidade de market-making. Duas frentes:

1) HISTORICO DE SPREAD (gratis): a Binance publica `bookTicker` mensal para
   futuros USDS-M em data.binance.vision. Cada linha tem o melhor bid/ask
   (top-of-book) com timestamp. Daqui calculamos spread em bps por simbolo:
   media, mediana, distribuicao e % do tempo com spread <= 1bp / <= 2bps.
   Isso responde se "ganhar o spread" via ordem maker e realista.

       uv run python -m simulation.binance_book --download --symbols BTCUSDT,ETHUSDT,SOLUSDT --months 2026-04,2026-05
       uv run python -m simulation.binance_book --analyze   # le o que estiver em cache e escreve o report

2) COLETOR L2 AO VIVO (forward capture): L2 profundo historico nao e gratis,
   entao abrimos um websocket de `depth` (top-of-book + ~5 niveis) e gravamos em
   disco com timestamp para acumular amostra a partir de agora.

       uv run python -m simulation.binance_book --collect --symbols BTCUSDT --minutes 1

Tudo cacheia em data/binance_book/. Se a rede estiver bloqueada, nada e inventado:
o downloader/coletor ficam prontos e o report marca "DADOS PENDENTES".

NOTA: este modulo so MEXE em data/binance_book/. Nao toca data/binance_1m/ nem
simulation/crypto_intraday.py (editados por outro agente).
"""

from __future__ import annotations

import io
import json
import logging
import statistics as _stats
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("simulation.binance_book")

CACHE_DIR = Path("data/binance_book")
REPORT_PATH = CACHE_DIR / "spread_report.txt"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# data.binance.vision: bookTicker mensal de futuros USDS-M (um = USDT-margined).
VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/bookTicker"

# Websocket de futuros USDS-M (fstream). depth5 = top-5 niveis @ 100ms.
WS_BASE = "wss://fstream.binance.com/stream?streams="


def _vision_url(symbol: str, month: str) -> str:
    """month no formato YYYY-MM."""
    return f"{VISION_BASE}/{symbol}/{symbol}-bookTicker-{month}.zip"


def _csv_cache_path(symbol: str, month: str) -> Path:
    return CACHE_DIR / f"{symbol}-bookTicker-{month}.csv"


# ---------------------------------------------------------------------------
# 1) DOWNLOAD DE bookTicker HISTORICO
# ---------------------------------------------------------------------------

def download_book_ticker(symbol: str, month: str, *, force: bool = False) -> Path | None:
    """Baixa e descompacta um mes de bookTicker. Retorna o caminho do CSV ou None.

    Idempotente: se o CSV ja existe e nao for `force`, nao rebaixa. Em falha de
    rede NAO levanta — loga o aviso e retorna None (DADOS PENDENTES).
    """
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _csv_cache_path(symbol, month)
    if out.exists() and not force:
        logger.info("cache hit %s %s -> %s", symbol, month, out)
        return out

    url = _vision_url(symbol, month)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            logger.warning("404 (mes indisponivel ainda?) %s", url)
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                data = fh.read()
        out.write_bytes(data)
        logger.info("baixado %s %s (%d bytes csv)", symbol, month, len(data))
        return out
    except Exception as exc:  # rede bloqueada / timeout / zip ruim
        logger.warning("falha ao baixar %s %s: %s", symbol, month, exc)
        return None


def download_universe(
    symbols: list[str], months: list[str], *, force: bool = False
) -> dict[str, list[Path]]:
    """Baixa varios simbolos x meses. Retorna mapa symbol -> [csv paths baixados]."""
    out: dict[str, list[Path]] = {}
    for sym in symbols:
        paths: list[Path] = []
        for m in months:
            p = download_book_ticker(sym, m, force=force)
            if p is not None:
                paths.append(p)
        out[sym] = paths
    return out


# ---------------------------------------------------------------------------
# 2) ANALISE DE SPREAD
# ---------------------------------------------------------------------------

@dataclass
class SpreadStats:
    symbol: str
    n_rows: int
    mean_bps: float
    median_bps: float
    p25_bps: float
    p75_bps: float
    p95_bps: float
    pct_le_1bp: float
    pct_le_2bp: float
    months: list[str]

    def as_text(self) -> str:
        return (
            f"{self.symbol:<10} n={self.n_rows:>10,d}  "
            f"mean={self.mean_bps:6.3f}bps  median={self.median_bps:6.3f}bps  "
            f"p25={self.p25_bps:6.3f}  p75={self.p75_bps:6.3f}  p95={self.p95_bps:6.3f}  "
            f"<=1bp={self.pct_le_1bp:5.1f}%  <=2bp={self.pct_le_2bp:5.1f}%"
        )


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Percentil simples (q em 0..1) sobre lista JA ordenada."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _spread_bps_from_csv(path: Path) -> list[float]:
    """Le um CSV de bookTicker e devolve a lista de spreads em bps.

    Colunas tipicas do bookTicker vision:
        update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
        transaction_time, event_time
    spread_bps = (ask - bid) / mid * 1e4, com mid = (ask + bid) / 2.
    Usa o csv stdlib (sem pandas) para nao estourar memoria em arquivos grandes.
    """
    import csv

    spreads: list[float] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return spreads
        # localiza colunas por nome (tolerante a variacao de schema)
        cols = {c.strip().lower(): i for i, c in enumerate(header)}
        bid_i = cols.get("best_bid_price")
        ask_i = cols.get("best_ask_price")
        # fallback posicional (schema classico)
        if bid_i is None or ask_i is None:
            bid_i, ask_i = 1, 3
            # se o header era na verdade uma linha de dados, reprocessa
            try:
                b0, a0 = float(header[bid_i]), float(header[ask_i])
                mid0 = (a0 + b0) / 2.0
                if mid0 > 0:
                    spreads.append((a0 - b0) / mid0 * 1e4)
            except (ValueError, IndexError):
                pass
        for row in reader:
            try:
                bid = float(row[bid_i])
                ask = float(row[ask_i])
            except (ValueError, IndexError):
                continue
            mid = (ask + bid) / 2.0
            if mid <= 0 or ask < bid:
                continue
            spreads.append((ask - bid) / mid * 1e4)
    return spreads


def analyze_symbol(symbol: str, csv_paths: list[Path]) -> SpreadStats | None:
    spreads: list[float] = []
    months: list[str] = []
    for p in csv_paths:
        if not p.exists():
            continue
        s = _spread_bps_from_csv(p)
        spreads.extend(s)
        # extrai o mes do nome do arquivo SYM-bookTicker-YYYY-MM.csv
        stem = p.stem
        if "-bookTicker-" in stem:
            months.append(stem.split("-bookTicker-")[-1])
    if not spreads:
        return None
    spreads.sort()
    n = len(spreads)
    n_le_1 = sum(1 for x in spreads if x <= 1.0)
    n_le_2 = sum(1 for x in spreads if x <= 2.0)
    return SpreadStats(
        symbol=symbol,
        n_rows=n,
        mean_bps=_stats.fmean(spreads),
        median_bps=_percentile(spreads, 0.5),
        p25_bps=_percentile(spreads, 0.25),
        p75_bps=_percentile(spreads, 0.75),
        p95_bps=_percentile(spreads, 0.95),
        pct_le_1bp=100.0 * n_le_1 / n,
        pct_le_2bp=100.0 * n_le_2 / n,
        months=sorted(set(months)),
    )


def _discover_cached_csvs(symbol: str) -> list[Path]:
    if not CACHE_DIR.exists():
        return []
    return sorted(CACHE_DIR.glob(f"{symbol}-bookTicker-*.csv"))


def analyze_universe(symbols: list[str]) -> dict[str, SpreadStats | None]:
    out: dict[str, SpreadStats | None] = {}
    for sym in symbols:
        paths = _discover_cached_csvs(sym)
        out[sym] = analyze_symbol(sym, paths)
    return out


def _verdict(results: dict[str, SpreadStats | None]) -> str:
    """Veredito preliminar maker/market-making a partir do spread observado.

    Heuristica: maker de futuros USDS-M na Binance paga ~0..2bps de fee (tier 0
    ~1.8bps; pode ser rebate em tiers altos). Para o spread "valer a pena":
    capturar metade do spread (half-spread) precisa cobrir a fee maker + risco de
    adverse selection. Se o spread mediano for <= ~1bp na maior parte do tempo, o
    half-spread (~0.5bp) NAO cobre a fee maker -> capturar o spread NAO paga as
    contas; o jogo vira queue priority / rebate / adverse-selection, nao "ganhar
    o spread".
    """
    have = {k: v for k, v in results.items() if v is not None}
    if not have:
        return (
            "VEREDITO: PENDENTE — sem dados de bookTicker em cache. Rode o "
            "downloader (ver comando no topo do report) quando houver rede."
        )
    med = _stats.fmean([v.median_bps for v in have.values()])
    pct1 = _stats.fmean([v.pct_le_1bp for v in have.values()])
    lines = [
        f"VEREDITO PRELIMINAR (maker / market-making):",
        f"  spread mediano medio entre simbolos = {med:.3f} bps; "
        f"% do tempo com spread <= 1bp = {pct1:.1f}%.",
    ]
    half = med / 2.0
    maker_fee_bps = 1.8  # tier 0 USDS-M futures maker
    if med <= 1.2:
        lines.append(
            f"  Spread MUITO estreito: half-spread ~{half:.3f}bps < fee maker "
            f"(~{maker_fee_bps}bps tier 0). 'Ganhar o spread' NAO paga as contas "
            f"sem rebate de tier alto. Market-making puro INVIAVEL para varejo; o "
            f"edge real estaria em rebate/queue/adverse-selection, nao no spread."
        )
    elif med <= 3.0:
        lines.append(
            f"  Spread moderado: half-spread ~{half:.3f}bps cobre a fee maker so "
            f"em tiers com rebate. Maker MARGINAL — depende de fill ratio e de "
            f"adverse selection. Vale modelar com a amostra L2 ao vivo antes."
        )
    else:
        lines.append(
            f"  Spread largo (~{med:.3f}bps): half-spread ~{half:.3f}bps > fee "
            f"maker. Capturar o spread via maker PODE valer — modelar fill e "
            f"adverse selection com a amostra L2 para confirmar."
        )
    lines.append(
        "  Para a versao MAKER do tribunal de momentum: use o spread mediano por "
        "simbolo acima como custo de cruzar/postar, em vez do slippage fixo."
    )
    return "\n".join(lines)


def write_report(results: dict[str, SpreadStats | None]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("RELATORIO DE SPREAD — Binance Futures USDS-M bookTicker")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Fonte: data.binance.vision/data/futures/um/monthly/bookTicker/<SYM>/")
    lines.append("spread_bps = (ask - bid) / mid * 1e4, mid = (ask + bid)/2, por tick de top-of-book.")
    lines.append("")
    lines.append("Comando p/ baixar (quando houver rede):")
    lines.append("  uv run python -m simulation.binance_book --download \\")
    lines.append("      --symbols BTCUSDT,ETHUSDT,SOLUSDT --months 2026-03,2026-04,2026-05")
    lines.append("  uv run python -m simulation.binance_book --analyze")
    lines.append("")
    lines.append("-" * 78)
    lines.append("SPREAD POR SIMBOLO")
    lines.append("-" * 78)
    any_data = False
    for sym, st in results.items():
        if st is None:
            lines.append(f"{sym:<10} DADOS PENDENTES (sem CSV em cache)")
        else:
            any_data = True
            lines.append(st.as_text())
            lines.append(f"{'':<10} meses: {', '.join(st.months) or '-'}")
    lines.append("")
    if not any_data:
        lines.append("STATUS: DADOS PENDENTES — nenhum bookTicker baixado ainda.")
        lines.append("")
    lines.append("-" * 78)
    lines.append(_verdict(results))
    lines.append("-" * 78)
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    logger.info("report escrito em %s", REPORT_PATH)
    return REPORT_PATH


# ---------------------------------------------------------------------------
# 3) COLETOR L2 AO VIVO (websocket depth)
# ---------------------------------------------------------------------------

def _l2_out_path(symbol: str) -> Path:
    return CACHE_DIR / f"l2_{symbol}_live.jsonl"


async def _collect_async(symbols: list[str], minutes: float) -> dict[str, int]:
    """Conecta no fstream depth5@100ms e grava cada update em JSONL.

    Cada linha: {ts_local, symbol, bids:[[p,q]x5], asks:[[p,q]x5]}.
    Grava append em data/binance_book/l2_<SYM>_live.jsonl. Retorna contagem por
    simbolo. Em falha de rede NAO levanta — loga e retorna o que coletou.
    """
    import asyncio
    import time

    import websockets

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    streams = "/".join(f"{s.lower()}@depth5@100ms" for s in symbols)
    url = WS_BASE + streams
    counts: dict[str, int] = {s: 0 for s in symbols}
    files = {s: _l2_out_path(s).open("a") for s in symbols}
    deadline = time.monotonic() + minutes * 60.0
    try:
        async with websockets.connect(url, open_timeout=20, close_timeout=5) as ws:
            logger.info("conectado ao depth ao vivo: %s", url)
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                data = msg.get("data", msg)
                stream = msg.get("stream", "")
                sym = stream.split("@")[0].upper() if stream else None
                if sym is None:
                    # stream unico: deduz pelo simbolo do payload
                    sym = (data.get("s") or "").upper() or (symbols[0] if len(symbols) == 1 else None)
                if sym not in files:
                    continue
                rec = {
                    "ts_local": time.time(),
                    "symbol": sym,
                    "bids": data.get("b", [])[:5],
                    "asks": data.get("a", [])[:5],
                }
                files[sym].write(json.dumps(rec) + "\n")
                counts[sym] += 1
    except Exception as exc:
        logger.warning("coletor L2 interrompido/falhou: %s", exc)
    finally:
        for fh in files.values():
            fh.close()
    logger.info("coleta L2 encerrada: %s", counts)
    return counts


def collect_live(symbols: list[str], minutes: float) -> dict[str, int]:
    import asyncio

    return asyncio.run(_collect_async(symbols, minutes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Microestrutura Binance: spread historico (bookTicker) + coletor L2 ao vivo"
    )
    parser.add_argument("--download", action="store_true", help="baixa bookTicker historico")
    parser.add_argument("--analyze", action="store_true", help="analisa o que estiver em cache e escreve o report")
    parser.add_argument("--collect", action="store_true", help="coletor L2 ao vivo (websocket depth)")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="lista separada por virgula")
    parser.add_argument("--months", default="", help="YYYY-MM separados por virgula (download)")
    parser.add_argument("--minutes", type=float, default=1.0, help="duracao da coleta L2 (min)")
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo com cache")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.download:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
        if not months:
            logger.error("--download exige --months YYYY-MM[,YYYY-MM...]")
            return 2
        download_universe(symbols, months, force=args.force)

    if args.collect:
        counts = collect_live(symbols, args.minutes)
        total = sum(counts.values())
        logger.info("coletados %d updates L2 no total: %s", total, counts)
        if total == 0:
            logger.warning("nenhum update coletado (rede bloqueada?) — coletor pronto, DADOS PENDENTES")

    # analyze roda por padrao se nada mais foi pedido, ou se --analyze
    if args.analyze or not (args.download or args.collect):
        results = analyze_universe(symbols)
        write_report(results)
        for sym, st in results.items():
            if st is None:
                logger.info("%s: DADOS PENDENTES", sym)
            else:
                logger.info(st.as_text())

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
