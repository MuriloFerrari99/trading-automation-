"""TRIBUNAL INTRADIARIO DE CRIPTO (Binance USDS-M perp) — H1: momentum curto.

Objetivo NAO e achar algo que funcione — e descobrir a VERDADE sobre se sobra
edge depois do custo real de maker/taker. Mesma barra de rigor do tribunal diario
(simulation.statistics: DSR, PSR, PBO).

Dados (gratis): klines de 1min dos dumps mensais da Binance via data.binance.vision.
  https://data.binance.vision/data/futures/um/monthly/klines/<SYM>/1m/<SYM>-1m-<YYYY-MM>.zip
Cache em data/binary_1m/<SYM>/. Universo + janela definidos abaixo. Loader estruturado
para aceitar aggTrades/bookTicker em fases futuras (liquidacao/market-making), mas AQUI
so klines.

HIPOTESE H1 — MOMENTUM CURTO:
  - Sinal no FECHAMENTO da barra t: retorno acumulado dos ultimos k min (k em {5,10,15,30})
    normalizado em z-score de vol, acima de um limiar (varre 2-3 limiares).
  - Entrada no OPEN da barra t+1 (sem look-ahead). Saida apos holding H (em {5,10,15} min).
    Intraday, long E short (perp permite short).
  - n_trials do DSR = k x limiar x H (sem trapaca de multiple-testing).

CUSTO (o que mata ou salva):
  (a) TAKER: cruza o spread, 4 bps/lado (8 bps round-trip).
  (b) MAKER: 2 bps/lado, mas ASSUME execucao passiva sem adverse selection — otimista,
      a confirmar com order book. Se SO passa no maker => veredito PENDENTE-DE-ORDER-BOOK.
  - Funding: se o hold cruza horario de funding (a cada 8h: 00/08/16 UTC), desconta a taxa.
    Default 1 bp / 8h (aproximacao — sem o dado real de funding).

Uso:
    uv run python -m simulation.crypto_intraday --download   # baixa/cacheia klines (precisa rede)
    uv run python -m simulation.crypto_intraday               # roda o tribunal (usa cache)
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.metrics import max_drawdown
from simulation.statistics import (
    CRYPTO_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
# 12 meses: 2024-06 .. 2025-05 (mes anterior ao dump corrente — meses fechados).
MONTHS = [
    "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11",
    "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
]
CACHE = Path("data/binance_1m")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

MINUTES_PER_YEAR = 365 * 24 * 60  # cripto 24/7, barra de 1min -> anualizacao da Sharpe

# Custos (fracao de preco, por LADO)
TAKER_BPS = 4.0
MAKER_BPS = 2.0
FUNDING_BPS_PER_8H = 1.0  # aproximacao (sem dado real)
FUNDING_HOURS = (0, 8, 16)  # horarios de settlement de funding (UTC)

# Grade de hipoteses H1 (conta TODA no n_trials)
LOOKBACKS_K = (5, 10, 15, 30)        # minutos de retorno acumulado para o sinal
Z_THRESHOLDS = (1.0, 1.5, 2.0)       # limiares em z-score de vol
HOLDINGS_H = (5, 10, 15)             # minutos de holding
VOL_WINDOW = 60                      # janela (min) para estimar a vol do z-score

# Colunas dos dumps de klines da Binance (futures UM monthly)
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


# --------------------------------------------------------------------------- #
# Download / loader
# --------------------------------------------------------------------------- #
def _kline_url(sym: str, month: str) -> str:
    return f"{BASE_URL}/{sym}/1m/{sym}-1m-{month}.zip"


def _kline_cache_path(sym: str, month: str) -> Path:
    # CSV gzip: sem dependencia nova (pyarrow/fastparquet podem nao estar instalados).
    return CACHE / sym / f"{sym}-1m-{month}.csv.gz"


def download_klines(symbols: list[str], months: list[str]) -> dict[str, list[str]]:
    """Baixa e cacheia os dumps mensais de klines 1min. Idempotente (pula o que ja existe).

    Retorna {"ok": [...], "fail": [...]} com tags "<SYM> <MONTH>" para diagnostico.
    """
    result: dict[str, list[str]] = {"ok": [], "fail": [], "cached": []}
    for sym in symbols:
        (CACHE / sym).mkdir(parents=True, exist_ok=True)
        for month in months:
            tag = f"{sym} {month}"
            out = _kline_cache_path(sym, month)
            if out.exists():
                result["cached"].append(tag)
                continue
            url = _kline_url(sym, month)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    raw = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    name = zf.namelist()[0]
                    with zf.open(name) as fh:
                        df = _parse_kline_csv(fh)
                df.to_csv(out, compression="gzip")
                result["ok"].append(tag)
                print(f"  baixado: {tag}  ({len(df)} barras)")
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, zipfile.BadZipFile) as e:
                result["fail"].append(f"{tag}: {type(e).__name__} {str(e)[:80]}")
                print(f"  FALHA: {tag} -> {type(e).__name__}")
    return result


def _parse_kline_csv(fh) -> pd.DataFrame:
    """Le o CSV de klines da Binance. Alguns dumps tem header, outros nao."""
    df = pd.read_csv(fh, header=None)
    # se a 1a linha for o header textual ('open_time'), recarrega pulando
    if str(df.iloc[0, 0]).strip().lower() in {"open_time", "open time"}:
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, : len(_KLINE_COLS)]
    df.columns = _KLINE_COLS[: df.shape[1]]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    # open_time em ms -> indice datetime UTC
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts").sort_index()


def load_symbol(sym: str, months: list[str] = MONTHS) -> pd.DataFrame | None:
    """Concatena os meses cacheados de um simbolo. None se nada cacheado.

    AGORA so klines. Estruturado para no futuro fundir aggTrades (microprice/fluxo) e
    bookTicker (spread/imbalance) pelo mesmo indice de timestamp — ver load_microstructure().
    """
    frames = []
    for month in months:
        p = _kline_cache_path(sym, month)
        if p.exists():
            d = pd.read_csv(p, index_col=0, compression="gzip")
            d.index = pd.to_datetime(d.index, utc=True)
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def load_microstructure(sym: str, months: list[str] = MONTHS) -> None:
    """STUB de fase futura: aggTrades (fluxo/microprice) + bookTicker (spread/imbalance).

    O loader de klines (load_symbol) ja indexa por timestamp UTC; aggTrades e bookTicker
    serao reamostrados para 1min e fundidos pelo mesmo indice. Nao implementado nesta fase
    (so klines). Mantido para fixar o contrato da interface de microestrutura.
    """
    raise NotImplementedError(
        "Microestrutura (aggTrades/bookTicker) e fase futura — esta fase opera so klines."
    )


# --------------------------------------------------------------------------- #
# H1 — momentum curto
# --------------------------------------------------------------------------- #
@dataclass
class TradeResult:
    gross: float          # retorno bruto do trade (ja com sinal long/short)
    net_taker: float      # liquido de custo taker (round-trip) + funding
    net_maker: float      # liquido de custo maker (round-trip) + funding
    minutes_held: int


def _funding_crossings(entry_idx: int, exit_idx: int, hours: np.ndarray) -> int:
    """Quantos settlements de funding (00/08/16 UTC) caem dentro do hold (entry, exit]."""
    # hours[j] = hora UTC do open da barra j. Conta barras cujo open marca exatamente um
    # settlement e que estao dentro da janela mantida.
    if exit_idx <= entry_idx:
        return 0
    window = hours[entry_idx + 1: exit_idx + 1]
    if window.size == 0:
        return 0
    return int(np.isin(window, FUNDING_HOURS).sum())


def run_h1_config(
    df: pd.DataFrame, k: int, z_thr: float, hold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Roda UMA config (k, z_thr, hold) de H1 sobre um simbolo.

    Sem look-ahead: sinal no CLOSE de t, entrada no OPEN de t+1, saida no OPEN de t+1+hold.
    Long se momentum z > +z_thr; short se z < -z_thr. Posicoes NAO sobrepostas
    (entra na proxima barra livre apos a saida) -> trades independentes.

    Retorna (gross[], net_taker[], net_maker[], n_bars) — n_bars p/ trades/dia.
    """
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    hours = df.index.hour.to_numpy()
    n = len(c)
    if n < VOL_WINDOW + k + hold + 5:
        return np.array([]), np.array([]), np.array([]), n

    # retorno log por minuto (close-to-close)
    logc = np.log(c)
    r1 = np.diff(logc, prepend=logc[0])  # r1[t] = log(c[t]/c[t-1])
    # momentum acumulado dos ultimos k min ate o close de t
    mom_k = pd.Series(r1).rolling(k).sum().to_numpy()
    # vol de barra: desvio-padrao dos retornos de 1min na janela; escala p/ k min (~sqrt(k))
    vol_1m = pd.Series(r1).rolling(VOL_WINDOW).std(ddof=0).to_numpy()
    vol_k = vol_1m * np.sqrt(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(vol_k > 0, mom_k / vol_k, 0.0)

    gross: list[float] = []
    net_t: list[float] = []
    net_m: list[float] = []

    taker_rt = 2.0 * TAKER_BPS / 1e4
    maker_rt = 2.0 * MAKER_BPS / 1e4
    fund_unit = FUNDING_BPS_PER_8H / 1e4

    t = VOL_WINDOW + k  # 1a barra com sinal valido
    while t + 1 + hold < n:
        zt = z[t]
        if not np.isfinite(zt) or abs(zt) < z_thr:
            t += 1
            continue
        side = 1.0 if zt > 0 else -1.0
        entry = o[t + 1]
        exit_px = o[t + 1 + hold]
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_px) and exit_px > 0):
            t += 1
            continue
        g = side * (exit_px / entry - 1.0)
        # funding: posicao paga (long) / recebe sinal oposto — modelamos como CUSTO
        # absoluto por settlement cruzado (conservador: sempre desconta).
        n_fund = _funding_crossings(t + 1, t + 1 + hold, hours)
        fund_cost = n_fund * fund_unit
        gross.append(g)
        net_t.append(g - taker_rt - fund_cost)
        net_m.append(g - maker_rt - fund_cost)
        t = t + 1 + hold  # nao sobrepoe: proxima entrada apos a saida
    return np.asarray(gross), np.asarray(net_t), np.asarray(net_m), n


def buy_and_hold_sharpe(df: pd.DataFrame) -> tuple[float, float]:
    """Sharpe anualizado e retorno total do buy-and-hold (close-to-close) do ativo."""
    c = df["close"].to_numpy(float)
    if c.size < 3:
        return 0.0, 0.0
    r = np.diff(c) / c[:-1]
    sh = observed_sharpe(r, periods_per_year=MINUTES_PER_YEAR)
    total = float(c[-1] / c[0] - 1.0)
    return sh, total


# --------------------------------------------------------------------------- #
# Agregacao por config (carteira de todos os simbolos) + tribunal
# --------------------------------------------------------------------------- #
@dataclass
class ConfigVerdict:
    k: int
    z_thr: float
    hold: int
    n_trades: int
    trades_per_day: float
    win_rate: float
    gross_mean_bps: float
    net_taker_mean_bps: float
    net_maker_mean_bps: float
    sharpe_taker_annual: float
    sharpe_maker_annual: float
    sharpe_gross_annual: float
    dsr_taker: float
    dsr_maker: float
    maxdd_taker: float
    net_taker_total: float
    rets_taker: np.ndarray  # por-trade (p/ matriz PBO)


def evaluate_config(
    symbols_data: dict[str, pd.DataFrame], k: int, z_thr: float, hold: int, n_trials: int
) -> ConfigVerdict:
    """Roda a config em TODOS os simbolos, junta os trades numa unica serie por-trade e julga.

    Sharpe por-trade anualizado: cada trade dura `hold` min; periodos/ano efetivos para a
    serie por-trade = MINUTES_PER_YEAR / hold (cada observacao cobre `hold` minutos)."""
    all_g, all_nt, all_nm = [], [], []
    total_bars = 0
    for df in symbols_data.values():
        g, nt, nm, nbars = run_h1_config(df, k, z_thr, hold)
        all_g.append(g)
        all_nt.append(nt)
        all_nm.append(nm)
        total_bars += nbars
    g = np.concatenate(all_g) if all_g else np.array([])
    nt = np.concatenate(all_nt) if all_nt else np.array([])
    nm = np.concatenate(all_nm) if all_nm else np.array([])

    n = int(g.size)
    days = max(total_bars / (24 * 60), 1e-9)
    periods_per_year_trade = MINUTES_PER_YEAR / hold  # cada trade cobre `hold` min

    if n < 2:
        return ConfigVerdict(k, z_thr, hold, n, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nt)

    win_rate = float((nt > 0).mean())
    sh_t = observed_sharpe(nt, periods_per_year=periods_per_year_trade)
    sh_m = observed_sharpe(nm, periods_per_year=periods_per_year_trade)
    sh_g = observed_sharpe(g, periods_per_year=periods_per_year_trade)
    v_t = evaluate_edge(nt, n_trials=n_trials, periods_per_year=periods_per_year_trade)
    v_m = evaluate_edge(nm, n_trials=n_trials, periods_per_year=periods_per_year_trade)
    eq_t = np.cumprod(1.0 + nt)
    return ConfigVerdict(
        k=k, z_thr=z_thr, hold=hold,
        n_trades=n,
        trades_per_day=n / days,
        win_rate=win_rate,
        gross_mean_bps=float(g.mean() * 1e4),
        net_taker_mean_bps=float(nt.mean() * 1e4),
        net_maker_mean_bps=float(nm.mean() * 1e4),
        sharpe_taker_annual=sh_t,
        sharpe_maker_annual=sh_m,
        sharpe_gross_annual=sh_g,
        dsr_taker=v_t.dsr,
        dsr_maker=v_m.dsr,
        maxdd_taker=max_drawdown(eq_t),
        net_taker_total=float(eq_t[-1] - 1.0),
        rets_taker=nt,
    )


def run_tribunal(symbols_data: dict[str, pd.DataFrame]) -> tuple[list[ConfigVerdict], float, float, dict]:
    """Roda toda a grade H1. Retorna (verdicts, bh_sharpe_medio, bh_ret_medio, pbo_info)."""
    n_trials = len(LOOKBACKS_K) * len(Z_THRESHOLDS) * len(HOLDINGS_H)
    verdicts: list[ConfigVerdict] = []
    for k in LOOKBACKS_K:
        for z_thr in Z_THRESHOLDS:
            for hold in HOLDINGS_H:
                verdicts.append(evaluate_config(symbols_data, k, z_thr, hold, n_trials))

    # buy-and-hold medio do universo (a barra a bater)
    bh_sh, bh_ret = [], []
    for df in symbols_data.values():
        s, t = buy_and_hold_sharpe(df)
        bh_sh.append(s)
        bh_ret.append(t)
    bh_sharpe = float(np.mean(bh_sh)) if bh_sh else 0.0
    bh_total = float(np.mean(bh_ret)) if bh_ret else 0.0

    # PBO: matriz (trades alinhados por menor n) x configs, usando net_taker
    pbo_info = {"pbo": None, "note": "amostra insuficiente"}
    series = [v.rets_taker for v in verdicts if v.rets_taker.size >= 20]
    if len(series) >= 2:
        m = min(s.size for s in series)
        if m >= 20:
            mat = np.column_stack([s[:m] for s in series])
            pbo = probability_of_backtest_overfitting(mat, n_splits=10)
            pbo_info = {"pbo": pbo, "note": f"{len(series)} configs x {m} trades"}
    return verdicts, bh_sharpe, bh_total, pbo_info


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #
def _kill_criterion_header() -> list[str]:
    return [
        "=" * 100,
        "TRIBUNAL INTRADIARIO DE CRIPTO (Binance USDS-M perp) — H1: MOMENTUM CURTO (hold 5-15 min)",
        "=" * 100,
        "",
        "KILL-CRITERION (pre-registrado, antes dos numeros):",
        "  PASSA somente se (versao TAKER): DSR >= 0.95 E Sharpe_liq_anual >= 1.0 E",
        "  retorno liquido > 0 E supera buy-and-hold do mesmo periodo.",
        "  Senao FALHA e momentum curto e ARQUIVADO.",
        "  (Versao MAKER e INFORMATIVA: se SO passa no maker => PENDENTE-DE-ORDER-BOOK,",
        "   nao PASSA — o maker assume execucao passiva SEM adverse selection, otimista.)",
        "",
        "Custos: TAKER 4bps/lado (8bps RT) | MAKER 2bps/lado (4bps RT, otimista) |",
        f"  funding {FUNDING_BPS_PER_8H:.0f}bp/8h aprox (sem dado real de funding) em 00/08/16 UTC.",
        f"n_trials (anti-snooping) = {len(LOOKBACKS_K)}k x {len(Z_THRESHOLDS)}z x {len(HOLDINGS_H)}H = "
        f"{len(LOOKBACKS_K) * len(Z_THRESHOLDS) * len(HOLDINGS_H)} configs.",
        "Sharpe anualizado por-trade (periods/ano = minutos_ano / holding). Long+short, sem sobreposicao.",
        "",
    ]


def render_report(
    verdicts: list[ConfigVerdict],
    bh_sharpe: float,
    bh_total: float,
    pbo_info: dict,
    symbols_data: dict[str, pd.DataFrame],
) -> str:
    lines = _kill_criterion_header()
    lines.append(f"Dados: {len(symbols_data)} simbolos, "
                 f"{sum(len(d) for d in symbols_data.values()):,} barras 1min totais.")
    lines.append(f"Buy-and-hold medio do universo: Sharpe_anual={bh_sharpe:.2f}  "
                 f"ret_total_periodo={bh_total * 100:+.1f}%")
    lines.append("")
    hdr = (f"{'k':>3} {'z':>4} {'H':>3} | {'trades':>7} {'t/dia':>6} {'win%':>5} | "
           f"{'gross_bps':>9} {'netT_bps':>8} {'netM_bps':>8} | "
           f"{'ShT':>6} {'ShM':>6} | {'DSR_T':>6} {'DSR_M':>6} | {'MddT%':>6} | veredito")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    # ordena por Sharpe taker desc para leitura
    for v in sorted(verdicts, key=lambda x: x.sharpe_taker_annual, reverse=True):
        if v.n_trades < 2:
            lines.append(f"{v.k:>3} {v.z_thr:>4.1f} {v.hold:>3} |  amostra insuficiente ({v.n_trades})")
            continue
        passes_taker = (
            v.dsr_taker >= 0.95
            and v.sharpe_taker_annual >= 1.0
            and v.net_taker_mean_bps > 0
            and v.sharpe_taker_annual > bh_sharpe
        )
        passes_maker_only = (
            not passes_taker
            and v.dsr_maker >= 0.95
            and v.sharpe_maker_annual >= 1.0
            and v.net_maker_mean_bps > 0
        )
        flag = "PASSA" if passes_taker else ("PEND-OB" if passes_maker_only else "FALHA")
        lines.append(
            f"{v.k:>3} {v.z_thr:>4.1f} {v.hold:>3} | {v.n_trades:>7} {v.trades_per_day:>6.1f} "
            f"{v.win_rate * 100:>5.1f} | {v.gross_mean_bps:>9.2f} {v.net_taker_mean_bps:>8.2f} "
            f"{v.net_maker_mean_bps:>8.2f} | {v.sharpe_taker_annual:>6.2f} {v.sharpe_maker_annual:>6.2f} | "
            f"{v.dsr_taker:>6.3f} {v.dsr_maker:>6.3f} | {v.maxdd_taker * 100:>6.1f} | {flag}"
        )

    lines.append("")
    if pbo_info["pbo"] is not None:
        lines.append(f"PBO (overfitting na selecao de config, net taker): {pbo_info['pbo']:.2f}  "
                     f"({pbo_info['note']})  [<0.5 = aceitavel]")
    else:
        lines.append(f"PBO: {pbo_info['note']}")
    lines.append("")

    # --- VEREDITO ---
    any_pass_taker = any(
        v.dsr_taker >= 0.95 and v.sharpe_taker_annual >= 1.0
        and v.net_taker_mean_bps > 0 and v.sharpe_taker_annual > bh_sharpe
        for v in verdicts if v.n_trades >= 2
    )
    any_pass_maker = any(
        v.dsr_maker >= 0.95 and v.sharpe_maker_annual >= 1.0 and v.net_maker_mean_bps > 0
        for v in verdicts if v.n_trades >= 2
    )
    lines.append("=" * 100)
    lines.append("VEREDITO H1 — MOMENTUM CURTO:")
    lines.append("=" * 100)
    if any_pass_taker:
        lines.append("")
        lines.append("  >>> PASSA (TAKER) <<<  ao menos uma config supera o kill-criterion liquido de taker.")
        lines.append("  Confirmar com walk-forward OOS e custo estressado antes de capital real.")
    elif any_pass_maker:
        lines.append("")
        lines.append("  >>> PENDENTE-DE-ORDER-BOOK <<<  passa SO no maker (execucao passiva idealizada).")
        lines.append("  NAO e PASSA: precisa de bookTicker p/ medir adverse selection do fill passivo.")
    else:
        lines.append("")
        lines.append("  ##############################################################################")
        lines.append("  #                                                                            #")
        lines.append("  #   FALHA                                                                    #")
        lines.append("  #                                                                            #")
        lines.append("  #   Momentum curto (5-15 min) NAO sobrevive ao custo real de taker.          #")
        lines.append("  #   Nenhuma config bate DSR>=0.95 + Sharpe>=1.0 + ret>0 + > buy-and-hold.     #")
        lines.append("  #   ARQUIVADO. Vitoria: economizou capital real.                             #")
        lines.append("  #                                                                            #")
        lines.append("  ##############################################################################")
        # diagnostico: quanto o custo come
        best_g = max(verdicts, key=lambda x: x.sharpe_gross_annual if x.n_trades >= 2 else -1e9)
        if best_g.n_trades >= 2:
            lines.append("")
            lines.append(f"  Diagnostico (melhor config BRUTA k={best_g.k} z={best_g.z_thr} H={best_g.hold}):")
            lines.append(f"    Sharpe BRUTO={best_g.sharpe_gross_annual:.2f}  -> "
                         f"Sharpe TAKER={best_g.sharpe_taker_annual:.2f}  -> "
                         f"Sharpe MAKER={best_g.sharpe_maker_annual:.2f}")
            lines.append(f"    ret/trade: bruto={best_g.gross_mean_bps:.2f}bps  "
                         f"taker={best_g.net_taker_mean_bps:.2f}bps  maker={best_g.net_maker_mean_bps:.2f}bps")
            lines.append("    -> o custo de cruzar o spread come o edge bruto por-trade.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _no_data_report() -> str:
    cmd = "uv run python -m simulation.crypto_intraday --download"
    return (
        "\n".join(_kill_criterion_header())
        + "\n"
        + "REDE BLOQUEADA / SEM DADOS: harness pronto, faltam os klines.\n"
        + f"Cache esperado: {CACHE}/<SYM>/<SYM>-1m-<YYYY-MM>.csv.gz\n"
        + f"Universo: {', '.join(UNIVERSE)}\n"
        + f"Janela: {MONTHS[0]} .. {MONTHS[-1]} (12 meses)\n\n"
        + "Rode numa maquina com rede:\n"
        + f"    {cmd}\n"
        + "depois:\n"
        + "    uv run python -m simulation.crypto_intraday\n\n"
        + "Fonte (dump mensal por simbolo):\n"
        + f"    {BASE_URL}/<SYM>/1m/<SYM>-1m-<YYYY-MM>.zip\n"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tribunal intradiario de cripto (Binance perp) — H1 momentum curto")
    parser.add_argument("--download", action="store_true", help="baixa/cacheia os klines 1min (precisa rede)")
    parser.add_argument("--report-file", default="data/crypto_intraday_verdict.txt")
    parser.add_argument("--symbols", nargs="*", default=UNIVERSE)
    parser.add_argument("--months", nargs="*", default=MONTHS)
    args = parser.parse_args(argv)

    if args.download:
        print(f"Baixando klines 1min de {len(args.symbols)} simbolos x {len(args.months)} meses...")
        res = download_klines(args.symbols, args.months)
        print(f"\nOK: {len(res['ok'])} | cache: {len(res['cached'])} | FALHA: {len(res['fail'])}")
        for f in res["fail"]:
            print(f"  {f}")
        if not (res["ok"] or res["cached"]):
            print("\nNenhum dado baixado. Verifique a rede.")
            return 1
        return 0

    # carrega cache
    symbols_data: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        df = load_symbol(sym, args.months)
        if df is not None and len(df) > VOL_WINDOW + 100:
            symbols_data[sym] = df

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not symbols_data:
        text = _no_data_report()
        print(text)
        out.write_text(text, encoding="utf-8")
        print(f"Relatorio (SEM DADOS) salvo em {out}")
        print(f"\nPara baixar: uv run python -m simulation.crypto_intraday --download")
        return 0

    print(f"Carregados {len(symbols_data)} simbolos. Rodando tribunal H1...")
    verdicts, bh_sharpe, bh_total, pbo_info = run_tribunal(symbols_data)
    text = render_report(verdicts, bh_sharpe, bh_total, pbo_info, symbols_data)
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
