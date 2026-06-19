"""Loader de OHLC INTRADAY M15 (Dukascopy) para o tribunal da FIMATHE REAL.

A FIMATHE canonica e DAY TRADE intraday em FOREX/XAU no M15/M1 (ver
docs/FIMATHE_SPEC.md). yfinance so entrega ~60 dias de M15 -> inutil. Este
loader usa o historico intraday GRATUITO da Dukascopy (`dukascopy-python`),
baixa OHLC M15 do lado BID, normaliza para colunas minusculas
(open/high/low/close[/volume]) com indice datetime UTC, e cacheia por
instrumento em data/fimathe_intraday_cache/*.csv — o tribunal roda offline
apos um unico fetch.

INSTRUMENTOS (os de Marcelo): XAUUSD (PRIMARIO, 99% das operacoes dele),
EURUSD, GBPUSD.

PROFUNDIDADE: alvo de 3 anos. Baixa em blocos mensais (limite de 30k barras/
chamada da lib; ~2900 barras M15/mes em forex) com retry, concatena e
deduplica. Dukascopy entrega o que existe; se um bloco falhar a rede, o loader
loga e segue (e o tribunal reporta a profundidade real obtida).

SPREAD por instrumento (pips), para o tribunal cruzar na entrada e na saida
(intraday -> SO spread, sem swap):
  XAUUSD ~2.5 pips, EURUSD ~0.6 pip, GBPUSD ~0.9 pip.
1 pip = pip_size em preco: 0.01 no ouro (cotacao ~milhares), 0.0001 nos majors.

SEM LOOK-AHEAD: o loader so entrega OHLC bruto; toda a logica de canal de
abertura / rompimento / saida intraday mora em fimathe/canal_abertura.py e
simulation/fimathe_intraday.py.

Uso:
    uv run python -m data.fimathe_intraday_data                 # universo, 3 anos
    uv run python -m data.fimathe_intraday_data --years 2
    uv run python -m data.fimathe_intraday_data --force         # ignora cache

Se a rede bloquear totalmente, o fetch devolve vazio e o tribunal reporta
"DADOS PENDENTES" com este comando — sem inventar numeros.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("data.fimathe_intraday_data")

CACHE_DIR = Path("data/fimathe_intraday_cache")

# Instrumentos de Marcelo. dukascopy_python.instruments.<const>.
# pip_size: 0.01 no ouro (XAU/USD cota ~4000; 1 pip = 0.01), 0.0001 nos majors.
# spread_pips: estimativa realista de spread intraday (cruzado na entrada E saida).
INSTRUMENTS: dict[str, dict] = {
    "XAUUSD": {  # PRIMARIO — 99% das operacoes dele
        "duka": "INSTRUMENT_FX_METALS_XAU_USD",
        "pip_size": 0.01,
        "spread_pips": 2.5,
    },
    "EURUSD": {
        "duka": "INSTRUMENT_FX_MAJORS_EUR_USD",
        "pip_size": 0.0001,
        "spread_pips": 0.6,
    },
    "GBPUSD": {
        "duka": "INSTRUMENT_FX_MAJORS_GBP_USD",
        "pip_size": 0.0001,
        "spread_pips": 0.9,
    },
}
DEFAULT_INSTRUMENTS = list(INSTRUMENTS.keys())

# M15: ~96 barras/dia (forex ~24h, 5d/sem) -> ~252*5*... Anualizacao do Sharpe
# do tribunal e feita em RETORNOS DIARIOS (1 sequencia/dia), entao usamos dias
# de pregao por ano (~252) la. Aqui so guardamos a constante de barras/dia.
BARS_PER_DAY_M15 = 96
TRADING_DAYS_FX = 252


def cache_path(instrument: str) -> Path:
    return CACHE_DIR / f"{instrument}_M15.csv"


def pip_size(instrument: str) -> float:
    return INSTRUMENTS[instrument]["pip_size"]


def spread_pips(instrument: str) -> float:
    return INSTRUMENTS[instrument]["spread_pips"]


def spread_price(instrument: str) -> float:
    """Spread em PRECO (full spread). Meio-spread = isto/2, cruzado em cada ponta."""
    info = INSTRUMENTS[instrument]
    return info["spread_pips"] * info["pip_size"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Dukascopy -> DataFrame(open,high,low,close[,volume]) indice datetime UTC,
    ordenado e sem duplicatas/barras de range zero (mercado fechado)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep].copy()
    idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index = idx
    out.index.name = "timestamp"
    out = out[~out.index.isna()]
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    # barras de range zero (fim de semana / feriado): high==low e open==close.
    flat = (out["high"] == out["low"]) & (out["open"] == out["close"])
    out = out[~flat]
    out = out[~out.index.duplicated(keep="first")]
    return out.sort_index()


def _fetch_blocks(
    duka_instrument: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Baixa M15 [start, end) em blocos mensais (resiliente a falha de bloco).

    A lib limita ~30k barras/chamada; M15 da ~2900 barras/mes em forex -> 1 mes
    por bloco e folgado. Concatena e deduplica. Loga blocos que falham e segue.
    """
    import dukascopy_python as d
    from dukascopy_python import instruments as ins

    instrument = getattr(ins, duka_instrument)
    frames: list[pd.DataFrame] = []
    cur = start
    n_ok = 0
    n_fail = 0
    while cur < end:
        # proximo "fim de mes" (aprox): avanca ~31 dias e trunca ao alvo final.
        nxt = min(cur + timedelta(days=31), end)
        try:
            df = d.fetch(instrument, d.INTERVAL_MIN_15, d.OFFER_SIDE_BID, cur, nxt)
            if df is not None and not df.empty:
                frames.append(df)
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:  # noqa: BLE001 — rede/duka; loga e segue
            logger.warning("Bloco %s [%s..%s] falhou: %s", duka_instrument, cur.date(), nxt.date(), exc)
            n_fail += 1
        cur = nxt
    logger.info("%s: %d blocos OK, %d vazios/falha", duka_instrument, n_ok, n_fail)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def load_instrument(
    instrument: str,
    *,
    years: float = 3.0,
    end: datetime | None = None,
    force: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """OHLC M15 de UM instrumento. Usa cache; baixa via Dukascopy se ausente/force.

    Devolve DataFrame possivelmente vazio se a rede falhar por completo (o
    tribunal trata vazio como DADOS PENDENTES). NAO inventa dados.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Instrumento desconhecido: {instrument}. {DEFAULT_INSTRUMENTS}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(instrument)
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "timestamp"
        return df[~df.index.isna()]

    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=int(round(years * 365.25)))
    raw = _fetch_blocks(INSTRUMENTS[instrument]["duka"], start, end)
    df = _normalize(raw)
    if write and not df.empty:
        df.to_csv(p)
        logger.info(
            "%s M15: %d barras (%s..%s) -> %s",
            instrument, len(df),
            df.index[0].date() if len(df) else "-",
            df.index[-1].date() if len(df) else "-", p,
        )
    return df


def load_universe(
    instruments: list[str] | None = None,
    *,
    years: float = 3.0,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """{instrumento: DataFrame M15}. Pula instrumentos que falharem por completo."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    out: dict[str, pd.DataFrame] = {}
    for inst in instruments:
        try:
            df = load_instrument(inst, years=years, force=force)
            if not df.empty:
                out[inst] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha total em %s: %s", inst, exc)
    return out


def data_status(instruments: list[str] | None = None) -> list[str]:
    """Linhas de status (profundidade real em cache) para o cabecalho do veredito."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    lines: list[str] = []
    for inst in instruments:
        p = cache_path(inst)
        if not p.exists():
            lines.append(f"  {inst:7s}: AUSENTE (rode o loader)")
            continue
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        if df.empty:
            lines.append(f"  {inst:7s}: VAZIO")
            continue
        span_days = (df.index[-1] - df.index[0]).days
        lines.append(
            f"  {inst:7s}: {len(df):>7d} barras M15  "
            f"{df.index[0].date()}..{df.index[-1].date()}  (~{span_days/365.25:.2f} anos)"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download de OHLC M15 intraday (Dukascopy) + cache p/ tribunal FIMATHE"
    )
    parser.add_argument("--instruments", default="", help="CSV; vazio = XAUUSD,EURUSD,GBPUSD")
    parser.add_argument("--years", type=float, default=3.0, help="profundidade em anos (default 3)")
    parser.add_argument("--force", action="store_true", help="ignora cache e rebaixa")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    insts = (
        [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
        if args.instruments else DEFAULT_INSTRUMENTS
    )
    res = load_universe(insts, years=args.years, force=args.force)
    if not res:
        logger.error(
            "NENHUM dado baixado. Rede/Dukascopy. Comando: "
            "uv run python -m data.fimathe_intraday_data --force"
        )
        return 1
    total = sum(len(d) for d in res.values())
    logger.info("Concluido: %d instrumentos, %d barras M15 no total.", len(res), total)
    for ln in data_status(insts):
        logger.info(ln)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
