"""CARRY DELTA-NEUTRO EM CRIPTO (Binance USDS-M perp) — cash-and-carry / funding harvest.

CONTEXTO: o tribunal de momentum DIRECIONAL falhou (DSR 0.04, PBO 0.84, nao bate
buy&hold). A pesquisa de modelo de negocio aponta que o edge de melhor risco-retorno
para este cliente e DELTA-NEUTRO: coletar o FUNDING que os shorts do perpetuo recebem
quando o funding e positivo (regime normal de bull). Isso e um premio ESTRUTURAL
(prêmio por alavancagem/sentimento long), NAO previsao de direcao — por isso tende a
sobreviver onde os sinais direcionais morreram. Esta tarefa MEDE esse edge com honestidade.

A ESTRATEGIA (cash-and-carry delta-neutra):
  - COMPRA SPOT + VENDE o PERP de mesmo notional. Delta ~0 (os legs se cancelam, a menos
    do basis). Coleta o funding (shorts recebem funding quando funding > 0). Sem aposta
    de direcao do preco.
  - VARIANTE de contraste: SO-SHORT-PERP coletando funding SEM hedge spot (tem risco
    direcional cheio). Reportada lado a lado para deixar claro por que o hedge importa.

DADOS (gratis, Binance, sem chave):
  - FUNDING RATE historico dos perps USDS-M: dumps mensais
      https://data.binance.vision/data/futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-<YYYY-MM>.zip
    Funding settla a cada 8h (00/08/16 UTC). Cache em data/binance_carry/funding/.
  - PERP klines 1min: REUTILIZA data/binance_1m/ (ja temos 12 meses; padrao de
    simulation.crypto_intraday). NAO baixamos perp de novo.
  - SPOT klines 1min (para o basis perp-spot): dump mensal
      https://data.binance.vision/data/spot/monthly/klines/<SYM>/1m/<SYM>-1m-<YYYY-MM>.zip
    Cache em data/binance_carry/spot/.

  Se a rede bloquear (intermitente neste ambiente), o downloader e IDEMPOTENTE e o
  relatorio sai como "DADOS PENDENTES" com o comando exato — NAO inventamos numero.

O QUE MEDIMOS (carry, NAO DSR de sinal direcional):
  - Retorno ANUALIZADO LIQUIDO do carry delta-neutro por moeda e agregado (bruto vs liquido).
    Bruto = soma do funding recebido + PnL do basis (entrada vs saida). Liquido = bruto -
    custos de cruzar o spread nos DOIS legs (~4 bps taker/lado/leg).
  - Vol do retorno, Sharpe, pior drawdown, % de periodos (settlements) com funding NEGATIVO
    (quando a estrategia PAGA em vez de receber — o risco recorrente).
  - STRESS: piores meses de funding (regime de baixa) e quanto o basis pode mover contra
    voce antes do funding compensar.
  - Capacidade/alavancagem: a que alavancagem o retorno % fica "interessante" e por que
    isso aumenta o risco de liquidacao do leg short.

RESSALVA OBRIGATORIA (no relatorio): o backtest captura risco de PRECO/funding/basis, mas
NAO captura o risco dominante real desta estrategia — CONTRAPARTE (falencia/hack de
exchange), DE-PEG de stablecoin, LIQUIDACAO por margem mal gerida. O veredito de retorno e
NECESSARIO mas NAO SUFICIENTE; a viabilidade real depende de gestao operacional/contraparte.

Uso:
    uv run python -m simulation.crypto_carry --download   # baixa funding + spot (precisa rede)
    uv run python -m simulation.crypto_carry              # roda o estudo (usa cache)
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.metrics import max_drawdown
from simulation.statistics import observed_sharpe

# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]  # os mais liquidos
MONTHS = [
    "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11",
    "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
]

# Reusa o cache de perp klines de crypto_intraday (NAO baixa de novo).
PERP_KLINE_CACHE = Path("data/binance_1m")
# Cache PROPRIO para funding + spot (separado, como pedido).
CARRY_CACHE = Path("data/binance_carry")
FUNDING_CACHE = CARRY_CACHE / "funding"
SPOT_CACHE = CARRY_CACHE / "spot"

FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
SPOT_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# Funding settla a cada 8h: 3 settlements/dia, 365 dias.
FUNDINGS_PER_YEAR = 3 * 365  # 1095 periodos de funding por ano
DAYS_PER_YEAR = 365

# Custo de transacao: TAKER por LADO, por LEG. Cash-and-carry tem 2 legs (spot+perp),
# e cada leg cruza o spread na ENTRADA e na SAIDA -> 4 cruzamentos no round-trip total.
TAKER_BPS_PER_SIDE = 4.0

# Colunas dos dumps de klines (spot e futures partilham o layout).
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


# --------------------------------------------------------------------------- #
# Download / loader — FUNDING
# --------------------------------------------------------------------------- #
def _funding_url(sym: str, month: str) -> str:
    return f"{FUNDING_BASE}/{sym}/{sym}-fundingRate-{month}.zip"


def _funding_cache_path(sym: str, month: str) -> Path:
    return FUNDING_CACHE / sym / f"{sym}-fundingRate-{month}.csv.gz"


def _parse_funding_csv(fh) -> pd.DataFrame:
    """Le o dump de fundingRate da Binance. Layout varia entre versoes:
    novo:  calc_time, funding_interval_hours, last_funding_rate
    antigo: calc_time, last_funding_rate   (sem coluna de intervalo)
    Alguns dumps tem header textual, outros nao. Normaliza para colunas
    [ts (UTC), funding_rate (float, fracao por settlement)].
    """
    df = pd.read_csv(fh, header=None)
    first = str(df.iloc[0, 0]).strip().lower()
    if first in {"calc_time", "calctime"}:
        df = df.iloc[1:].reset_index(drop=True)
    ncol = df.shape[1]
    if ncol >= 3:
        # calc_time, funding_interval_hours, last_funding_rate
        ts_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        rate = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    else:
        # calc_time, last_funding_rate
        ts_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        rate = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    out = pd.DataFrame({"funding_rate": rate})
    out["ts"] = pd.to_datetime(ts_raw.astype("Int64").astype("int64"), unit="ms", utc=True)
    out = out.dropna(subset=["funding_rate"])
    return out.set_index("ts").sort_index()


def download_funding(symbols: list[str], months: list[str]) -> dict[str, list[str]]:
    """Baixa/cacheia os dumps mensais de funding rate. Idempotente."""
    result: dict[str, list[str]] = {"ok": [], "fail": [], "cached": []}
    for sym in symbols:
        (FUNDING_CACHE / sym).mkdir(parents=True, exist_ok=True)
        for month in months:
            tag = f"{sym} {month}"
            out = _funding_cache_path(sym, month)
            if out.exists():
                result["cached"].append(tag)
                continue
            try:
                with urllib.request.urlopen(_funding_url(sym, month), timeout=60) as resp:
                    raw = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    with zf.open(zf.namelist()[0]) as fh:
                        df = _parse_funding_csv(fh)
                df.to_csv(out, compression="gzip")
                result["ok"].append(tag)
                print(f"  funding baixado: {tag}  ({len(df)} settlements)")
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, zipfile.BadZipFile) as e:
                result["fail"].append(f"{tag}: {type(e).__name__} {str(e)[:80]}")
                print(f"  FALHA funding: {tag} -> {type(e).__name__}")
    return result


def load_funding(sym: str, months: list[str] = MONTHS) -> pd.Series | None:
    """Concatena o funding cacheado de um simbolo. Serie indexada por ts UTC (fracao/8h)."""
    frames = []
    for month in months:
        p = _funding_cache_path(sym, month)
        if p.exists():
            d = pd.read_csv(p, index_col=0, compression="gzip")
            d.index = pd.to_datetime(d.index, utc=True, format="ISO8601")
            frames.append(d["funding_rate"])
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated(keep="first")]


# --------------------------------------------------------------------------- #
# Download / loader — SPOT klines (para o basis)
# --------------------------------------------------------------------------- #
def _spot_url(sym: str, month: str) -> str:
    return f"{SPOT_BASE}/{sym}/1m/{sym}-1m-{month}.zip"


def _spot_cache_path(sym: str, month: str) -> Path:
    return SPOT_CACHE / sym / f"{sym}-1m-{month}.csv.gz"


def _parse_kline_csv(fh) -> pd.DataFrame:
    df = pd.read_csv(fh, header=None)
    if str(df.iloc[0, 0]).strip().lower() in {"open_time", "open time"}:
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, : len(_KLINE_COLS)]
    df.columns = _KLINE_COLS[: df.shape[1]]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    ot = df["open_time"].astype("int64")
    # Binance trocou a unidade de open_time entre dumps: ms (13 digitos) ate ~2024,
    # microsegundos (16 digitos) a partir de ~2025. Detecta pela magnitude.
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    df["ts"] = pd.to_datetime(ot, unit=unit, utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts").sort_index()


def download_spot(symbols: list[str], months: list[str]) -> dict[str, list[str]]:
    """Baixa/cacheia spot klines 1min para o basis. Idempotente."""
    result: dict[str, list[str]] = {"ok": [], "fail": [], "cached": []}
    for sym in symbols:
        (SPOT_CACHE / sym).mkdir(parents=True, exist_ok=True)
        for month in months:
            tag = f"{sym} {month}"
            out = _spot_cache_path(sym, month)
            if out.exists():
                result["cached"].append(tag)
                continue
            try:
                with urllib.request.urlopen(_spot_url(sym, month), timeout=60) as resp:
                    raw = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    with zf.open(zf.namelist()[0]) as fh:
                        df = _parse_kline_csv(fh)
                df.to_csv(out, compression="gzip")
                result["ok"].append(tag)
                print(f"  spot baixado: {tag}  ({len(df)} barras)")
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, zipfile.BadZipFile) as e:
                result["fail"].append(f"{tag}: {type(e).__name__} {str(e)[:80]}")
                print(f"  FALHA spot: {tag} -> {type(e).__name__}")
    return result


def _load_klines(cache_dir: Path, sym: str, months: list[str]) -> pd.DataFrame | None:
    frames = []
    for month in months:
        p = cache_dir / sym / f"{sym}-1m-{month}.csv.gz"
        if p.exists():
            d = pd.read_csv(p, index_col=0, compression="gzip")
            d.index = pd.to_datetime(d.index, utc=True)
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def load_perp(sym: str, months: list[str] = MONTHS) -> pd.DataFrame | None:
    """Perp klines do cache COMPARTILHADO de crypto_intraday (data/binance_1m/)."""
    return _load_klines(PERP_KLINE_CACHE, sym, months)


def load_spot(sym: str, months: list[str] = MONTHS) -> pd.DataFrame | None:
    """Spot klines do cache PROPRIO (data/binance_carry/spot/)."""
    return _load_klines(SPOT_CACHE, sym, months)


# --------------------------------------------------------------------------- #
# Carry delta-neutro: o motor
# --------------------------------------------------------------------------- #
@dataclass
class CarryResult:
    sym: str
    n_settlements: int
    days: float
    # funding bruto (o que o short recebe; soma simples)
    funding_sum: float            # soma das taxas de funding ao longo do periodo (fracao)
    funding_mean_bps: float       # media por settlement (bps)
    pct_funding_negative: float   # fracao de settlements com funding < 0 (estrategia PAGA)
    worst_month_funding_bps: float  # pior soma mensal de funding (bps)
    # basis (perp/spot - 1) na entrada e saida; PnL do basis no delta-neutro
    basis_entry_bps: float
    basis_exit_bps: float
    basis_pnl: float              # fracao: ganho/perda do hedge por movimento do basis
    has_basis: bool
    # retornos delta-neutro (por settlement, ja sobre notional do leg)
    dn_gross_ann: float           # retorno anualizado bruto (funding + basis, sem custo)
    dn_net_ann: float             # retorno anualizado liquido (com custo de entrada/saida)
    dn_sharpe: float              # Sharpe anualizado da serie por-settlement (liquido)
    dn_maxdd: float               # pior drawdown da equity delta-neutro liquida
    dn_vol_ann: float             # vol anualizada do retorno por-settlement
    # variante so-short-perp (com risco direcional) para contraste
    short_only_gross_ann: float
    short_only_net_ann: float
    short_only_sharpe: float
    short_only_maxdd: float
    # buy-and-hold do spot (a barra direcional a contrastar)
    bh_spot_ann: float
    bh_spot_sharpe: float
    bh_spot_maxdd: float
    entry_cost_frac: float        # custo total de entrada+saida (fracao do notional de 1 leg)
    note: str = ""


def _annualize_total(total_ret: float, days: float) -> float:
    """Converte retorno TOTAL do periodo em retorno anualizado (composto)."""
    if days <= 0:
        return 0.0
    yrs = days / DAYS_PER_YEAR
    if yrs <= 0:
        return 0.0
    base = 1.0 + total_ret
    if base <= 0:
        return -1.0
    return float(base ** (1.0 / yrs) - 1.0)


def compute_carry(
    sym: str,
    funding: pd.Series,
    perp: pd.DataFrame,
    spot: pd.DataFrame | None,
    taker_bps_per_side: float = TAKER_BPS_PER_SIDE,
) -> CarryResult:
    """Mede o carry delta-neutro de UM simbolo a partir do funding + precos.

    Modelo (honesto e conservador):
      - Entra na 1a barra util (compra spot, vende perp de mesmo notional), mantem ate o fim.
      - FUNDING: a cada settlement o short do perp RECEBE funding_rate * notional do perp.
        Como o notional spot == notional perp, o funding recebido por 1 unidade de capital
        (1 leg de notional N, capital = N para o spot, perp e margem) e ~ funding_rate.
        Modelamos o retorno por-settlement do delta-neutro = funding_rate + d(basis).
      - BASIS: o hedge spot+short-perp tem PnL = -(d(perp) - d(spot)) por unidade de notional,
        i.e. o componente direcional cancela, sobra a variacao do basis (perp/spot - 1).
        Capturamos o basis de entrada e de saida; o PnL do basis no periodo e
        (basis_entry - basis_exit) (se o perp converge para o spot, short-perp ganha).
      - CUSTO: 4 cruzamentos de spread (spot in, perp in, spot out, perp out) a
        taker_bps_per_side cada -> custo total = 4 * taker_bps_per_side (bps do notional 1 leg).

    Serie por-settlement (para Sharpe/vol/DD) = funding_rate de cada settlement (o componente
    recorrente e a fonte de risco recorrente: funding negativo). O basis entra como ajuste
    de PnL de inicio/fim (nao recorrente — so realiza ao desmontar).
    """
    f = funding.dropna().astype(float)
    if f.size < 2:
        return _empty_carry(sym, "funding insuficiente")

    # janela comum: alinhar funding ao periodo coberto pelo perp
    t0, t1 = perp.index[0], perp.index[-1]
    f = f[(f.index >= t0) & (f.index <= t1)]
    if f.size < 2:
        return _empty_carry(sym, "sem funding na janela do perp")

    rates = f.to_numpy(float)
    n = rates.size
    days = max((f.index[-1] - f.index[0]).total_seconds() / 86400.0, 1e-9)

    funding_sum = float(rates.sum())
    funding_mean_bps = float(rates.mean() * 1e4)
    pct_neg = float((rates < 0).mean())

    # pior mes de funding (soma mensal)
    monthly = f.groupby([f.index.year, f.index.month]).sum()
    worst_month_bps = float(monthly.min() * 1e4) if monthly.size else 0.0

    # ----- BASIS (perp vs spot), amostrado a CADA settlement -----
    # O basis = perp/spot - 1. A posicao delta-neutro (long spot + short perp) carrega
    # o MtM do basis: seu PnL por settlement = -(d basis) (short-perp ganha quando o perp
    # cai relativo ao spot). Essa variacao do basis e a VERDADEIRA fonte de vol da posicao
    # mantida — sem ela o Sharpe fica artificialmente alto (so o ruido do funding).
    basis_entry = basis_exit = basis_pnl = 0.0
    has_basis = False
    d_basis = np.zeros(n)  # variacao do basis entre settlements (PnL do hedge = -d_basis)
    if spot is not None and len(spot) > 2:
        common = perp.index.intersection(spot.index)
        if common.size > 10:
            pc_s = perp.loc[common, "close"].astype(float)
            sc_s = spot.loc[common, "close"].astype(float)
            valid = (pc_s > 0) & (sc_s > 0)
            pc_s, sc_s = pc_s[valid], sc_s[valid]
            if pc_s.size > 10:
                basis_full = (pc_s / sc_s - 1.0)
                # reamostra o basis nos instantes de funding (ffill: ultimo basis conhecido)
                basis_at_f = basis_full.reindex(
                    basis_full.index.union(f.index)
                ).ffill().reindex(f.index)
                bvals = basis_at_f.to_numpy(float)
                bvals = pd.Series(bvals).ffill().bfill().to_numpy()
                basis_entry = float(bvals[0])
                basis_exit = float(bvals[-1])
                basis_pnl = basis_entry - basis_exit  # PnL total do basis ao desmontar
                d_basis = np.diff(bvals, prepend=bvals[0])  # d_basis[0]=0
                has_basis = True

    # ----- CUSTO -----
    # 4 cruzamentos (spot in/out + perp in/out) a taker_bps_per_side cada.
    entry_cost_frac = 4.0 * taker_bps_per_side / 1e4

    # ----- DELTA-NEUTRO -----
    # PnL por settlement = funding recebido (rates) + MtM do hedge (-d_basis).
    # custo de entrada/saida amortizado linearmente entre settlements (entra/sai 1x).
    per_settle = rates - d_basis - entry_cost_frac / n
    eq = np.cumprod(1.0 + per_settle)
    dn_total_gross = float(funding_sum + basis_pnl)
    dn_total_net = float(eq[-1] - 1.0)
    dn_gross_ann = _annualize_total(dn_total_gross, days)
    dn_net_ann = _annualize_total(dn_total_net, days)
    dn_sharpe = observed_sharpe(per_settle, periods_per_year=FUNDINGS_PER_YEAR)
    dn_vol_ann = float(np.std(per_settle, ddof=1) * np.sqrt(FUNDINGS_PER_YEAR)) if n > 1 else 0.0
    dn_maxdd = max_drawdown(eq)

    # ----- VARIANTE SO-SHORT-PERP (com risco direcional) -----
    # short perp: ganha funding mas tambem -(retorno do perp). Sem hedge spot.
    pc = perp["close"].astype(float).to_numpy()
    so_gross_ann = so_net_ann = so_sharpe = so_maxdd = 0.0
    if pc.size > 2:
        # mapear funding settlements em retorno de preco do perp entre settlements
        perp_at_f = perp["close"].reindex(perp.index.union(f.index)).astype(float)
        perp_at_f = perp_at_f.interpolate().reindex(f.index)
        ppx = perp_at_f.to_numpy(float)
        if np.isfinite(ppx).sum() > 2:
            ppx = pd.Series(ppx).ffill().bfill().to_numpy()
            price_ret = np.diff(ppx) / ppx[:-1]  # retorno do perp entre settlements
            # short-perp por-settlement = funding[t] - price_ret[t] (recebe funding, sofre alta)
            so_settle = rates[1:] - price_ret - entry_cost_frac / max(n - 1, 1)
            if so_settle.size > 1:
                so_total = float(np.prod(1.0 + so_settle) - 1.0)
                so_net_ann = _annualize_total(so_total, days)
                so_total_gross = float(np.prod(1.0 + (rates[1:] - price_ret)) - 1.0)
                so_gross_ann = _annualize_total(so_total_gross, days)
                so_sharpe = observed_sharpe(so_settle, periods_per_year=FUNDINGS_PER_YEAR)
                so_maxdd = max_drawdown(np.cumprod(1.0 + so_settle))

    # ----- BUY-AND-HOLD do SPOT (barra direcional) -----
    bh_ann = bh_sharpe = bh_maxdd = 0.0
    base_px = spot if (spot is not None and len(spot) > 2) else perp
    bc = base_px["close"].astype(float).to_numpy()
    if bc.size > 2:
        bh_total = float(bc[-1] / bc[0] - 1.0)
        bh_ann = _annualize_total(bh_total, days)
        # Sharpe diario do buy-and-hold (resample para 1 ponto/dia)
        daily = base_px["close"].astype(float).resample("1D").last().dropna()
        if daily.size > 2:
            dr = daily.pct_change().dropna().to_numpy()
            bh_sharpe = observed_sharpe(dr, periods_per_year=DAYS_PER_YEAR)
            bh_maxdd = max_drawdown(daily.to_numpy() / daily.to_numpy()[0])

    return CarryResult(
        sym=sym,
        n_settlements=n,
        days=days,
        funding_sum=funding_sum,
        funding_mean_bps=funding_mean_bps,
        pct_funding_negative=pct_neg,
        worst_month_funding_bps=worst_month_bps,
        basis_entry_bps=basis_entry * 1e4,
        basis_exit_bps=basis_exit * 1e4,
        basis_pnl=basis_pnl,
        has_basis=has_basis,
        dn_gross_ann=dn_gross_ann,
        dn_net_ann=dn_net_ann,
        dn_sharpe=dn_sharpe,
        dn_maxdd=dn_maxdd,
        dn_vol_ann=dn_vol_ann,
        short_only_gross_ann=so_gross_ann,
        short_only_net_ann=so_net_ann,
        short_only_sharpe=so_sharpe,
        short_only_maxdd=so_maxdd,
        bh_spot_ann=bh_ann,
        bh_spot_sharpe=bh_sharpe,
        bh_spot_maxdd=bh_maxdd,
        entry_cost_frac=entry_cost_frac,
    )


def _empty_carry(sym: str, note: str) -> CarryResult:
    return CarryResult(
        sym=sym, n_settlements=0, days=0.0, funding_sum=0.0, funding_mean_bps=0.0,
        pct_funding_negative=0.0, worst_month_funding_bps=0.0, basis_entry_bps=0.0,
        basis_exit_bps=0.0, basis_pnl=0.0, has_basis=False, dn_gross_ann=0.0,
        dn_net_ann=0.0, dn_sharpe=0.0, dn_maxdd=0.0, dn_vol_ann=0.0,
        short_only_gross_ann=0.0, short_only_net_ann=0.0, short_only_sharpe=0.0,
        short_only_maxdd=0.0, bh_spot_ann=0.0, bh_spot_sharpe=0.0, bh_spot_maxdd=0.0,
        entry_cost_frac=0.0, note=note,
    )


# --------------------------------------------------------------------------- #
# Agregacao da carteira (equal-weight pelas 4 moedas)
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioCarry:
    per_sym: list[CarryResult] = field(default_factory=list)
    dn_gross_ann: float = 0.0
    dn_net_ann: float = 0.0
    dn_sharpe: float = 0.0
    dn_maxdd: float = 0.0
    dn_vol_ann: float = 0.0
    pct_funding_negative: float = 0.0
    bh_spot_ann: float = 0.0
    bh_spot_sharpe: float = 0.0
    leverage_for_interesting: float = 0.0  # alavancagem p/ dn_net_ann bater bh_spot_ann


def aggregate(results: list[CarryResult], target_ann: float) -> PortfolioCarry:
    """Carteira equal-weight das moedas com dado valido. Junta as series por-settlement
    alinhadas pelo menor n para Sharpe/DD agregados; medias simples para os anuais."""
    valid = [r for r in results if r.n_settlements >= 2]
    pf = PortfolioCarry(per_sym=results)
    if not valid:
        return pf

    pf.dn_gross_ann = float(np.mean([r.dn_gross_ann for r in valid]))
    pf.dn_net_ann = float(np.mean([r.dn_net_ann for r in valid]))
    pf.dn_vol_ann = float(np.mean([r.dn_vol_ann for r in valid]))
    pf.pct_funding_negative = float(np.mean([r.pct_funding_negative for r in valid]))
    pf.bh_spot_ann = float(np.mean([r.bh_spot_ann for r in valid]))
    pf.bh_spot_sharpe = float(np.mean([r.bh_spot_sharpe for r in valid]))

    # Sharpe/DD agregados: media simples das Sharpes individuais (diversificacao real exigiria
    # alinhar timestamps; usamos a media como proxy conservador) + DD do pior caso.
    pf.dn_sharpe = float(np.mean([r.dn_sharpe for r in valid]))
    pf.dn_maxdd = float(min(r.dn_maxdd for r in valid))  # pior DD individual

    # Alavancagem para o net delta-neutro "ficar interessante" (= bater o buy-and-hold ann).
    # retorno escala ~linear com alavancagem L (funding * L), entao L* = target / dn_net_ann.
    if pf.dn_net_ann > 1e-9 and target_ann > 0:
        pf.leverage_for_interesting = target_ann / pf.dn_net_ann
    return pf


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #
def _header() -> list[str]:
    return [
        "=" * 100,
        "CARRY DELTA-NEUTRO EM CRIPTO (Binance USDS-M perp) — cash-and-carry / funding harvest",
        "=" * 100,
        "",
        "VEREDITO (criterio honesto, pre-registrado):",
        '  "ATRAENTE se: retorno liquido anualizado do carry delta-neutro >= buy-and-hold na regua',
        "   de risco (Sharpe liq >= 1.0 com alavancagem <= 2x) E sobrevive ao pior regime de funding",
        "   sem drawdown ruinoso. Senao, edge insuficiente para o risco operacional embutido.\"",
        "",
        "ESTRATEGIA: compra SPOT + vende PERP de mesmo notional (delta ~0). Coleta funding (shorts",
        "  recebem quando funding > 0). Variante so-short-perp (com risco direcional) mostrada p/ contraste.",
        f"CUSTO: taker {TAKER_BPS_PER_SIDE:.0f} bps/lado x 4 cruzamentos (spot+perp, entrada+saida) = "
        f"{4 * TAKER_BPS_PER_SIDE:.0f} bps round-trip.",
        f"Funding: 3 settlements/dia (00/08/16 UTC) -> {FUNDINGS_PER_YEAR}/ano. Universo: {', '.join(UNIVERSE)}.",
        "",
        "RESSALVA OBRIGATORIA: este backtest captura risco de PRECO/FUNDING/BASIS, mas NAO captura o",
        "  risco DOMINANTE real: CONTRAPARTE (falencia/hack de exchange), DE-PEG de stablecoin,",
        "  LIQUIDACAO por margem mal gerida no leg short. O veredito de retorno e NECESSARIO mas NAO",
        "  SUFICIENTE; a viabilidade real depende de gestao operacional/contraparte.",
        "",
    ]


def render_report(pf: PortfolioCarry, target_ann: float) -> str:
    lines = _header()
    valid = [r for r in pf.per_sym if r.n_settlements >= 2]

    lines.append("-" * 100)
    lines.append("POR MOEDA (delta-neutro: funding + basis; bruto vs liquido):")
    lines.append("-" * 100)
    hdr = (f"{'sym':>8} {'setls':>6} {'fund_bps':>8} {'%neg':>6} {'piorMes':>8} | "
           f"{'DNgross%':>8} {'DNnet%':>8} {'Sharpe':>6} {'MaxDD%':>7} {'vol%':>6} | "
           f"{'basisEnt':>8} {'basisSai':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in pf.per_sym:
        if r.n_settlements < 2:
            lines.append(f"{r.sym:>8}   sem dado valido ({r.note})")
            continue
        lines.append(
            f"{r.sym:>8} {r.n_settlements:>6} {r.funding_mean_bps:>8.3f} "
            f"{r.pct_funding_negative * 100:>5.1f}% {r.worst_month_funding_bps:>8.1f} | "
            f"{r.dn_gross_ann * 100:>8.2f} {r.dn_net_ann * 100:>8.2f} {r.dn_sharpe:>6.2f} "
            f"{r.dn_maxdd * 100:>7.2f} {r.dn_vol_ann * 100:>6.2f} | "
            f"{r.basis_entry_bps:>8.1f} {r.basis_exit_bps:>8.1f}"
        )

    lines.append("")
    lines.append("CONTRASTE — variante SO-SHORT-PERP (sem hedge spot; risco direcional CHEIO):")
    sub = (f"{'sym':>8} {'SOgross%':>9} {'SOnet%':>9} {'Sharpe':>7} {'MaxDD%':>8} | "
           f"buy&hold: {'BH%':>8} {'BHsharpe':>9} {'BHmaxDD%':>9}")
    lines.append(sub)
    lines.append("-" * len(sub))
    for r in valid:
        lines.append(
            f"{r.sym:>8} {r.short_only_gross_ann * 100:>9.2f} {r.short_only_net_ann * 100:>9.2f} "
            f"{r.short_only_sharpe:>7.2f} {r.short_only_maxdd * 100:>8.2f} | "
            f"          {r.bh_spot_ann * 100:>8.2f} {r.bh_spot_sharpe:>9.2f} {r.bh_spot_maxdd * 100:>9.2f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("AGREGADO (carteira equal-weight das moedas validas):")
    lines.append("=" * 100)
    lines.append(f"  Carry DELTA-NEUTRO  | bruto_ann = {pf.dn_gross_ann * 100:+.2f}%  "
                 f"liquido_ann = {pf.dn_net_ann * 100:+.2f}%")
    lines.append(f"                      | Sharpe_liq = {pf.dn_sharpe:.2f}  "
                 f"vol_ann = {pf.dn_vol_ann * 100:.2f}%  pior_MaxDD = {pf.dn_maxdd * 100:.2f}%")
    lines.append(f"                      | % settlements com funding NEGATIVO (paga) = "
                 f"{pf.pct_funding_negative * 100:.1f}%")
    lines.append(f"  Buy-and-hold spot   | ann = {pf.bh_spot_ann * 100:+.2f}%  "
                 f"Sharpe = {pf.bh_spot_sharpe:.2f}  (a barra direcional a bater em ann)")
    lines.append("")

    # ----- ALAVANCAGEM -----
    lines.append("-" * 100)
    lines.append("ALAVANCAGEM — a que ponto o carry delta-neutro fica 'interessante' e o risco:")
    lines.append("-" * 100)
    if pf.dn_net_ann > 1e-9:
        lev = pf.leverage_for_interesting
        lines.append(f"  Retorno 1x (sem alavancagem) liquido = {pf.dn_net_ann * 100:.2f}%/ano.")
        lines.append(f"  Para igualar a barra de retorno alvo ({target_ann * 100:.1f}%/ano = buy&hold),")
        lines.append(f"  precisaria de ~{lev:.1f}x de alavancagem no leg short.")
        if lev <= 2.0:
            lines.append(f"  -> dentro do teto de 2x: o carry e 'interessante' SEM alavancagem extrema.")
        else:
            lines.append(f"  -> ACIMA do teto de 2x (>{lev:.1f}x): para ficar interessante exigiria")
            lines.append(f"     alavancagem que dispara o risco de LIQUIDACAO do short num spike de preco.")
        lines.append("  RISCO da alavancagem: cada x multiplica o funding MAS tambem aproxima o leg short")
        lines.append("  da liquidacao; um pico de preco + funding negativo simultaneo pode liquidar a margem")
        lines.append("  antes do funding compensar (o backtest NAO modela call de margem intraday).")
    else:
        lines.append("  Carry liquido <= 0 a 1x: alavancagem AMPLIFICA prejuizo, nao salva. Nao ha")
        lines.append("  nivel de alavancagem 'seguro' que torne um carry negativo atraente.")
    lines.append("")

    # ----- STRESS -----
    lines.append("-" * 100)
    lines.append("STRESS — pior regime de funding e movimento de basis:")
    lines.append("-" * 100)
    for r in valid:
        basis_move = abs(r.basis_exit_bps - r.basis_entry_bps)
        lines.append(
            f"  {r.sym}: pior mes de funding = {r.worst_month_funding_bps:.1f} bps acumulados; "
            f"{r.pct_funding_negative * 100:.1f}% dos settlements pagaram; "
            f"basis moveu {basis_move:.1f} bps na janela."
        )
    lines.append("  Leitura: nos meses de funding negativo a estrategia PAGA; o hedge spot a protege da")
    lines.append("  direcao do preco, mas NAO do basis nem de funding negativo persistente (regime de baixa).")
    lines.append("")

    # ----- VEREDITO -----
    lines.append("=" * 100)
    lines.append("VEREDITO:")
    lines.append("=" * 100)
    passes = (
        pf.dn_net_ann > 0
        and pf.dn_sharpe >= 1.0
        and pf.leverage_for_interesting <= 2.0
        and pf.dn_maxdd > -0.20  # drawdown nao-ruinoso (< 20%)
        and pf.dn_net_ann >= pf.bh_spot_ann
    )
    if not valid:
        lines.append("  INDETERMINADO — sem dados validos para julgar.")
    elif passes:
        lines.append("")
        lines.append("  >>> ATRAENTE (na regua de PRECO/FUNDING) <<<")
        lines.append(f"  Carry liquido {pf.dn_net_ann * 100:.2f}%/ano, Sharpe {pf.dn_sharpe:.2f}, "
                     f"alavancagem <= 2x, sem drawdown ruinoso.")
        lines.append("  MAS: este veredito e NECESSARIO, nao SUFICIENTE. Antes de capital real, a")
        lines.append("  viabilidade depende de gestao de CONTRAPARTE, DE-PEG e MARGEM (nao modelados).")
    else:
        lines.append("")
        lines.append("  ##############################################################################")
        lines.append("  #  EDGE INSUFICIENTE PARA O RISCO OPERACIONAL EMBUTIDO                       #")
        lines.append("  ##############################################################################")
        reasons = []
        if pf.dn_net_ann <= 0:
            reasons.append(f"carry liquido <= 0 ({pf.dn_net_ann * 100:.2f}%/ano) — o custo come o funding")
        if pf.dn_sharpe < 1.0:
            reasons.append(f"Sharpe liquido {pf.dn_sharpe:.2f} < 1.0")
        if pf.leverage_for_interesting > 2.0:
            reasons.append(f"so 'interessante' acima de {pf.leverage_for_interesting:.1f}x (> 2x -> risco de liquidacao)")
        if pf.dn_maxdd <= -0.20:
            reasons.append(f"drawdown ruinoso ({pf.dn_maxdd * 100:.1f}%)")
        if 0 < pf.dn_net_ann < pf.bh_spot_ann:
            reasons.append(f"nao bate buy&hold em ann ({pf.dn_net_ann * 100:.2f}% < {pf.bh_spot_ann * 100:.2f}%)")
        for rs in reasons:
            lines.append(f"  - {rs}")
        lines.append("")
        lines.append("  Nota honesta: o carry delta-neutro a 1x e ESTRUTURALMENTE magro em bps/settlement;")
        lines.append("  so fica 'grande' com alavancagem, e e exatamente a alavancagem que traz o risco")
        lines.append("  de liquidacao + contraparte que NAO esta no backtest. Nao vendemos o sonho.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _no_data_report() -> str:
    cmd = "uv run python -m simulation.crypto_carry --download"
    return (
        "\n".join(_header())
        + "\n"
        + "DADOS PENDENTES: harness pronto e idempotente, faltam funding e/ou spot.\n"
        + f"Perp klines (compartilhado): {PERP_KLINE_CACHE}/<SYM>/<SYM>-1m-<YYYY-MM>.csv.gz\n"
        + f"Funding (cache proprio):     {FUNDING_CACHE}/<SYM>/<SYM>-fundingRate-<YYYY-MM>.csv.gz\n"
        + f"Spot   (cache proprio):      {SPOT_CACHE}/<SYM>/<SYM>-1m-<YYYY-MM>.csv.gz\n"
        + f"Universo: {', '.join(UNIVERSE)}   Janela: {MONTHS[0]} .. {MONTHS[-1]} (12 meses)\n\n"
        + "Rode numa maquina com rede:\n"
        + f"    {cmd}\n"
        + "depois:\n"
        + "    uv run python -m simulation.crypto_carry\n\n"
        + "Fontes (dumps mensais por simbolo):\n"
        + f"    funding: {FUNDING_BASE}/<SYM>/<SYM>-fundingRate-<YYYY-MM>.zip\n"
        + f"    spot:    {SPOT_BASE}/<SYM>/1m/<SYM>-1m-<YYYY-MM>.zip\n"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Carry delta-neutro em cripto (Binance perp)")
    parser.add_argument("--download", action="store_true", help="baixa funding + spot (precisa rede)")
    parser.add_argument("--report-file", default="data/crypto_carry_verdict.txt")
    parser.add_argument("--symbols", nargs="*", default=UNIVERSE)
    parser.add_argument("--months", nargs="*", default=MONTHS)
    args = parser.parse_args(argv)

    if args.download:
        print(f"Baixando funding de {len(args.symbols)} simbolos x {len(args.months)} meses...")
        rf = download_funding(args.symbols, args.months)
        print(f"  funding -> OK:{len(rf['ok'])} cache:{len(rf['cached'])} FALHA:{len(rf['fail'])}")
        print(f"Baixando spot klines de {len(args.symbols)} simbolos x {len(args.months)} meses...")
        rs = download_spot(args.symbols, args.months)
        print(f"  spot -> OK:{len(rs['ok'])} cache:{len(rs['cached'])} FALHA:{len(rs['fail'])}")
        for f in rf["fail"] + rs["fail"]:
            print(f"  {f}")
        if not (rf["ok"] or rf["cached"]):
            print("\nNenhum funding obtido. Verifique a rede.")
            return 1
        return 0

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    results: list[CarryResult] = []
    any_data = False
    for sym in args.symbols:
        funding = load_funding(sym, args.months)
        perp = load_perp(sym, args.months)
        spot = load_spot(sym, args.months)
        if funding is None or perp is None:
            results.append(_empty_carry(
                sym, "funding ausente" if funding is None else "perp klines ausente"))
            continue
        any_data = True
        results.append(compute_carry(sym, funding, perp, spot))

    if not any_data:
        text = _no_data_report()
        print(text)
        out.write_text(text, encoding="utf-8")
        print(f"Relatorio (DADOS PENDENTES) salvo em {out}")
        return 0

    # alvo de retorno = buy-and-hold medio do spot (a barra direcional).
    valid = [r for r in results if r.n_settlements >= 2]
    target_ann = float(np.mean([r.bh_spot_ann for r in valid])) if valid else 0.0
    target_ann = max(target_ann, 0.10)  # piso de 10%/ano como "interessante" minimo
    pf = aggregate(results, target_ann)
    text = render_report(pf, target_ann)
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
