"""CRYPTO TREND-FOLLOWING — TSMOM diario em majors (Binance spot, klines 1d gratis).

ESTRATEGIA (time-series momentum, classico Moskowitz/Ooi/Pedersen aplicado a cripto):
  - Universo: majors liquidos com historia longa na Binance spot
    (BTC, ETH, BNB, XRP, LTC, ADA; SOL/DOGE entram quando ha dado).
  - Sinal por ativo i, dia t:  s_i,t = sign(retorno passado em janela L)  (ou breakout
    de canal de L dias). Calculado com dado <= t-1 (close de t-1), aplicado ao retorno
    t->t+1. SEM look-ahead.
  - Position sizing: vol-scaled por ativo (alvo de vol por leg) e depois portfolio
    vol-target (escala o book inteiro para um alvo de vol anualizada). Long/short.
  - Custo real: rebalance diario paga turnover * custo. Cripto Alpaca = 25 bps/lado +
    slippage (CRYPTO_BASE). Tambem roda cenario ESTRESSADO (2x).

BARRA (tribunal): PASSA so se DSR>=0.95 E Sharpe_liq robusto E (bate buy&hold BTC OU
diversifica com correlacao BAIXA ao SPY melhorando o conjunto), robusto OOS.

n_trials HONESTO: contamos TODA a grade testada (sinais x lookbacks x variantes) no DSR
via expected_max_sharpe, e medimos PBO via CSCV sobre a matriz de todas as configs.

DADOS (gratis, sem chave): data.binance.vision dumps mensais de klines 1d spot:
  https://data.binance.vision/data/spot/monthly/klines/<SYM>/1d/<SYM>-1d-<YYYY-MM>.zip
  Cache em data/tsmom_cache/. Idempotente. Se a rede bloquear, reporta DADOS PENDENTES.
SPY (corr ao mercado): yfinance (gratis). Se faltar, corr fica None e a barra usa
buy&hold BTC.

Uso:
  uv run python -m simulation.crypto_tsmom --download   # baixa/cacheia os 1d
  uv run python -m simulation.crypto_tsmom              # roda o tribunal (usa cache)
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import CRYPTO_BASE, CostModel
from simulation.metrics import max_drawdown
from simulation.statistics import (
    CRYPTO_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "tsmom_cache"
REPORT = ROOT / "data" / "crypto_tsmom_verdict.txt"
KLINE_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# Universo: majors com historia longa. Inicio individual e detectado pelo dado disponivel.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]

# Janela de estudo: 2019-01 ate 2025-05 (alinha com o cache 1m existente no fim).
START = pd.Timestamp("2019-01-01", tz="UTC")
END = pd.Timestamp("2025-06-01", tz="UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [d.strftime("%Y-%m") for d in pd.period_range(start, end, freq="M").to_timestamp()]


MONTHS = _months(START, END)

ANN = float(np.sqrt(CRYPTO_PERIODS))  # cripto 24/7 -> 365

# --------------------------------------------------------------------------- #
# Download / cache (idempotente)
# --------------------------------------------------------------------------- #


def _cache_path(sym: str, month: str) -> Path:
    return CACHE / sym / f"{sym}-1d-{month}.csv.gz"


_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "qav", "trades", "tbbav", "tbqav", "ignore",
]


def download(symbols: list[str] = UNIVERSE, months: list[str] = MONTHS) -> dict:
    result = {"ok": [], "cached": [], "fail": []}
    for sym in symbols:
        for month in months:
            out = _cache_path(sym, month)
            tag = f"{sym}/{month}"
            if out.exists():
                result["cached"].append(tag)
                continue
            url = f"{KLINE_BASE}/{sym}/1d/{sym}-1d-{month}.zip"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    name = zf.namelist()[0]
                    df = pd.read_csv(io.BytesIO(zf.read(name)), header=None)
                # Binance mudou header em 2025: detecta se 1a linha e header textual.
                if str(df.iloc[0, 0]).lower().startswith("open"):
                    df = df.iloc[1:].reset_index(drop=True)
                df = df.iloc[:, : len(_COLS)]
                df.columns = _COLS
                ot = df["open_time"].astype("int64")
                # Binance mudou a unidade do timestamp: ms (13 digitos) ate ~2024,
                # us (16 digitos) a partir de 2025. Detecta pela magnitude.
                unit = "us" if ot.iloc[0] > 1e15 else "ms"
                df["ts"] = pd.to_datetime(ot, unit=unit, utc=True)
                df = df.set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, compression="gzip")
                result["ok"].append(tag)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    result["fail"].append(f"{tag}:404")  # ativo nao existia nesse mes
                else:
                    result["fail"].append(f"{tag}:HTTP{e.code}")
            except Exception as e:  # noqa: BLE001
                result["fail"].append(f"{tag}:{type(e).__name__}")
    return result


def load_daily(sym: str, months: list[str] = MONTHS) -> pd.Series | None:
    """Serie de close diario (UTC) de um simbolo, do cache. None se nao houver dado."""
    frames = []
    for month in months:
        p = _cache_path(sym, month)
        if p.exists():
            d = pd.read_csv(p, index_col=0)
            if d.empty:
                continue
            d.index = pd.to_datetime(d.index, utc=True)
            frames.append(d["close"])
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.loc[(s.index >= START) & (s.index < END)]


def build_panel() -> pd.DataFrame:
    """Painel de closes diarios (colunas = simbolos). Apenas simbolos com dado."""
    cols = {}
    for sym in UNIVERSE:
        s = load_daily(sym)
        if s is not None and s.size > 200:
            cols[sym] = s
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    # Indice diario regular (evita buracos de timezone); ffill curto so para alinhar.
    panel = panel.resample("1D").last()
    return panel


def load_spy_daily() -> pd.Series | None:
    """Retornos diarios do SPY (yfinance) alinhados a janela cripto. None se indisponivel."""
    try:
        import yfinance as yf

        spy = yf.download(
            "SPY", start=str(START.date()), end=str(END.date()),
            progress=False, auto_adjust=True,
        )
        if spy is None or spy.empty:
            return None
        close = spy["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index, utc=True)
        return close.pct_change().dropna()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Engine TSMOM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    signal: str  # "sign" | "breakout"
    lookback: int  # dias
    name: str


# Grade HONESTA: 2 sinais x 4 lookbacks = 8 configs. (Esse e o n_trials.)
LOOKBACKS = [10, 20, 50, 100]
SIGNALS = ["sign", "breakout"]
GRID = [Config(s, lb, f"{s}{lb}") for s in SIGNALS for lb in LOOKBACKS]

# Parametros de sizing (FIXOS, nao varridos -> nao entram no n_trials de selecao de sinal).
VOL_LOOKBACK = 30  # dias p/ estimar vol por ativo
PER_ASSET_VOL_TARGET = 0.40  # vol anual alvo por leg (cripto e volatil)
PORTFOLIO_VOL_TARGET = 0.20  # vol anual alvo do book agregado
MAX_LEVERAGE = 2.0  # teto de alavancagem bruta


def _signal_series(close: pd.Series, cfg: Config) -> pd.Series:
    """Sinal em {-1,0,+1} por dia, usando SO dado <= t-1 (shift(1) no fim)."""
    if cfg.signal == "sign":
        past_ret = close / close.shift(cfg.lookback) - 1.0
        sig = np.sign(past_ret)
    else:  # breakout de canal de L dias (excluindo hoje)
        roll_max = close.shift(1).rolling(cfg.lookback).max()
        roll_min = close.shift(1).rolling(cfg.lookback).min()
        sig = pd.Series(0.0, index=close.index)
        sig[close >= roll_max] = 1.0
        sig[close <= roll_min] = -1.0
        # Mantem a ultima direcao ate romper o lado oposto (stop-and-reverse).
        sig = sig.replace(0.0, np.nan).ffill().fillna(0.0)
    return sig.shift(1).fillna(0.0)  # decisao com info de t-1 -> aplica em t


def backtest_config(
    panel: pd.DataFrame, cfg: Config, cost: CostModel
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (retornos liquidos diarios do portfolio, retornos brutos) para uma config.

    Vol-scaled por ativo + portfolio vol-target + teto de leverage. Custo por turnover.
    """
    rets = panel.pct_change()
    # vol por ativo (anual), estimada com dado <= t-1
    vol = rets.rolling(VOL_LOOKBACK).std().shift(1) * ANN
    vol = vol.replace(0.0, np.nan)

    # peso bruto por ativo = sinal * (alvo_vol_leg / vol_realizada)
    weights = {}
    for sym in panel.columns:
        sig = _signal_series(panel[sym], cfg)
        w = sig * (PER_ASSET_VOL_TARGET / vol[sym])
        weights[sym] = w
    W = pd.DataFrame(weights).reindex(panel.index)

    # portfolio vol-target: escala o book p/ alvo, com base na vol realizada do book
    # (estimada ex-ante com pesos de t-1 e cov diagonal -> simples e sem look-ahead).
    raw_port_ret = (W * rets).sum(axis=1)
    realized_port_vol = raw_port_ret.rolling(VOL_LOOKBACK).std().shift(1) * ANN
    scale = (PORTFOLIO_VOL_TARGET / realized_port_vol).clip(upper=MAX_LEVERAGE)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    W_final = W.mul(scale, axis=0).fillna(0.0)
    # teto de alavancagem bruta total
    gross = W_final.abs().sum(axis=1).replace(0.0, np.nan)
    lev_cap = (MAX_LEVERAGE / gross).clip(upper=1.0).fillna(1.0)
    W_final = W_final.mul(lev_cap, axis=0)

    gross_ret = (W_final.shift(0) * rets).sum(axis=1)
    # turnover = soma das mudancas absolutas de peso (t-1 -> t)
    turnover = W_final.diff().abs().sum(axis=1).fillna(W_final.abs().sum(axis=1))
    cost_drag = turnover * (cost.per_side_bps / 1e4)
    net_ret = gross_ret - cost_drag

    valid = ~net_ret.isna() & (W_final.abs().sum(axis=1) > 0).shift(0).fillna(False)
    # corta o burn-in (primeiros max(lookback,vol_lb) dias)
    burn = max(cfg.lookback, VOL_LOOKBACK) + 2
    mask = np.zeros(len(net_ret), dtype=bool)
    mask[burn:] = True
    valid = valid & pd.Series(mask, index=net_ret.index)

    return net_ret[valid].to_numpy(), gross_ret[valid].to_numpy(), net_ret[valid].index


# --------------------------------------------------------------------------- #
# Tribunal
# --------------------------------------------------------------------------- #


def _equity(rets: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + rets)


def _cagr(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    eq = _equity(rets)
    years = rets.size / CRYPTO_PERIODS
    return float(eq[-1] ** (1.0 / max(years, 1e-9)) - 1.0)


def run_tribunal() -> dict:
    panel = build_panel()
    if panel.empty or panel.shape[1] < 2:
        return {"data_status": "blocked", "reason": "sem painel diario (cache vazio)"}

    syms = list(panel.columns)
    n_days = panel.dropna(how="all").shape[0]

    cost = CRYPTO_BASE
    cost_stress = CRYPTO_BASE.stressed(2.0)

    # roda a grade inteira (base e estressado)
    results = {}
    net_matrix_cols = []
    trial_sharpes_period = []
    common_index = None
    for cfg in GRID:
        net, gross, idx = backtest_config(panel, cfg, cost)
        net_s, _, _ = backtest_config(panel, cfg, cost_stress)
        if net.size < 50:
            continue
        results[cfg.name] = {
            "cfg": cfg,
            "net": net,
            "net_stress": net_s,
            "gross": gross,
            "idx": idx,
            "sharpe_net": observed_sharpe(net) * ANN,
            "sharpe_stress": observed_sharpe(net_s) * ANN,
            "sharpe_gross": observed_sharpe(gross) * ANN,
        }
        trial_sharpes_period.append(observed_sharpe(net))
        net_matrix_cols.append(pd.Series(net, index=idx, name=cfg.name))

    if not results:
        return {"data_status": "limited", "reason": "nenhuma config com amostra suficiente"}

    n_trials = len(GRID)  # HONESTO: toda a grade conta

    # MELHOR config in-sample por Sharpe estressado (escolha conservadora)
    best_name = max(results, key=lambda k: results[k]["sharpe_stress"])
    best = results[best_name]

    # ---- equal-weight ENSEMBLE das 8 configs (combate snooping de escolher 1) ----
    ens = pd.concat(net_matrix_cols, axis=1).dropna()
    ens_ret = ens.mean(axis=1).to_numpy() if not ens.empty else np.array([])

    # ---- PBO via CSCV sobre a matriz de configs (alinhadas) ----
    mat = pd.concat(net_matrix_cols, axis=1).dropna()
    pbo = (
        probability_of_backtest_overfitting(mat.to_numpy(), n_splits=10)
        if mat.shape[1] >= 2 and mat.shape[0] >= 20
        else float("nan")
    )

    # ---- benchmark: buy&hold BTC (mesma janela do melhor) ----
    btc = panel["BTCUSDT"].dropna()
    btc_ret_full = btc.pct_change().dropna()
    bh_ret = btc_ret_full.reindex(best["idx"]).dropna().to_numpy()
    bh_sharpe = observed_sharpe(bh_ret) * ANN
    bh_cagr = _cagr(bh_ret)
    bh_maxdd = max_drawdown(_equity(bh_ret))

    # ---- DSR/PSR do MELHOR (estressado) e do ENSEMBLE (estressado p/ as configs) ----
    verdict_best = evaluate_edge(
        best["net_stress"], n_trials=n_trials,
        trial_sharpes=trial_sharpes_period, periods_per_year=CRYPTO_PERIODS,
        min_sharpe_annual=0.8, dsr_threshold=0.95,
    )

    # ensemble estressado
    ens_stress = pd.concat(
        [pd.Series(results[k]["net_stress"], index=results[k]["idx"], name=k) for k in results],
        axis=1,
    ).dropna()
    ens_stress_ret = ens_stress.mean(axis=1).to_numpy() if not ens_stress.empty else np.array([])
    verdict_ens = (
        evaluate_edge(
            ens_stress_ret, n_trials=n_trials, periods_per_year=CRYPTO_PERIODS,
            min_sharpe_annual=0.8, dsr_threshold=0.95,
        )
        if ens_stress_ret.size > 50
        else None
    )

    # ---- correlacao ao SPY (mercado) usando o ENSEMBLE liquido base ----
    spy_ret = load_spy_daily()
    corr_spy = None
    if spy_ret is not None and ens.shape[0] > 20:
        ens_base = ens.mean(axis=1)
        joined = pd.concat(
            [ens_base.rename("strat"), spy_ret.rename("spy")], axis=1, join="inner"
        ).dropna()
        if joined.shape[0] > 30:
            corr_spy = float(joined["strat"].corr(joined["spy"]))

    # corr do MELHOR a BTC buy&hold (e isto so beta de cripto?)
    corr_btc = None
    bb = pd.concat(
        [pd.Series(best["net"], index=best["idx"], name="strat"),
         btc_ret_full.rename("btc")], axis=1, join="inner"
    ).dropna()
    if bb.shape[0] > 30:
        corr_btc = float(bb["strat"].corr(bb["btc"]))

    # ---- OOS split (primeira metade vs segunda) p/ robustez temporal ----
    half = best["net"].size // 2
    oos_sharpe_h1 = observed_sharpe(best["net"][:half]) * ANN
    oos_sharpe_h2 = observed_sharpe(best["net"][half:]) * ANN

    # metricas do melhor (liquido base)
    best_cagr = _cagr(best["net"])
    best_maxdd = max_drawdown(_equity(best["net"]))
    best_sharpe_net = best["sharpe_net"]
    best_sharpe_stress = best["sharpe_stress"]

    # ---- BARRA DE GRADUACAO ----
    beats_bh = (best_sharpe_stress > bh_sharpe) and (best_cagr > bh_cagr * 0.8)
    low_corr_diversifies = (
        corr_spy is not None and abs(corr_spy) < 0.3 and best_sharpe_stress > 0.5
    )
    dsr_ok = verdict_best.dsr >= 0.95
    sharpe_robust = best_sharpe_stress >= 0.8 and oos_sharpe_h1 > 0 and oos_sharpe_h2 > 0
    pbo_ok = (not np.isnan(pbo)) and pbo < 0.5

    passed = bool(dsr_ok and sharpe_robust and pbo_ok and (beats_bh or low_corr_diversifies))

    return {
        "data_status": "real",
        "symbols": syms,
        "n_days": n_days,
        "n_obs_best": int(best["net"].size),
        "n_trials": n_trials,
        "best_name": best_name,
        "best_sharpe_gross": best["sharpe_gross"],
        "best_sharpe_net": best_sharpe_net,
        "best_sharpe_stress": best_sharpe_stress,
        "best_cagr": best_cagr,
        "best_maxdd": best_maxdd,
        "dsr": verdict_best.dsr,
        "psr": verdict_best.psr,
        "sr_hurdle_annual": verdict_best.sr_benchmark_annual,
        "pbo": pbo,
        "corr_spy": corr_spy,
        "corr_btc": corr_btc,
        "oos_sharpe_h1": oos_sharpe_h1,
        "oos_sharpe_h2": oos_sharpe_h2,
        "bh_sharpe": bh_sharpe,
        "bh_cagr": bh_cagr,
        "bh_maxdd": bh_maxdd,
        "ens_sharpe_net": observed_sharpe(ens_ret) * ANN if ens_ret.size else float("nan"),
        "ens_sharpe_stress": observed_sharpe(ens_stress_ret) * ANN if ens_stress_ret.size else float("nan"),
        "ens_dsr": verdict_ens.dsr if verdict_ens else float("nan"),
        "ens_cagr": _cagr(ens_ret) if ens_ret.size else float("nan"),
        "ens_maxdd": max_drawdown(_equity(ens_ret)) if ens_ret.size else float("nan"),
        "grid": {k: (v["sharpe_net"], v["sharpe_stress"]) for k, v in results.items()},
        "beats_bh": beats_bh,
        "low_corr_diversifies": low_corr_diversifies,
        "dsr_ok": dsr_ok,
        "sharpe_robust": sharpe_robust,
        "pbo_ok": pbo_ok,
        "passed": passed,
    }


def write_report(res: dict) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("VEREDITO — CRYPTO TREND-FOLLOWING (TSMOM em majors, diario)")
    lines.append("=" * 78)
    lines.append("")
    if res.get("data_status") != "real":
        lines.append(f"DATA STATUS: {res.get('data_status')}")
        lines.append(f"Motivo: {res.get('reason')}")
        lines.append("")
        lines.append("Para obter os dados (gratis, sem chave):")
        lines.append("  uv run python -m simulation.crypto_tsmom --download")
        REPORT.write_text("\n".join(lines))
        return

    lines.append(f"Universo:        {', '.join(res['symbols'])}")
    lines.append(f"Dias no painel:  {res['n_days']}  | obs (melhor config): {res['n_obs_best']}")
    lines.append(f"n_trials (HONESTO, toda a grade 2 sinais x 4 lookbacks): {res['n_trials']}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("GRADE COMPLETA — Sharpe anual (liquido base / estressado 2x):")
    for name, (sn, ss) in sorted(res["grid"].items(), key=lambda x: -x[1][1]):
        lines.append(f"  {name:<12} net={sn:6.2f}   stress2x={ss:6.2f}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"MELHOR CONFIG (por Sharpe estressado): {res['best_name']}")
    lines.append(f"  Sharpe bruto:        {res['best_sharpe_gross']:.2f}")
    lines.append(f"  Sharpe liquido base: {res['best_sharpe_net']:.2f}")
    lines.append(f"  Sharpe estressado2x: {res['best_sharpe_stress']:.2f}")
    lines.append(f"  CAGR (liq base):     {res['best_cagr']*100:.1f}%")
    lines.append(f"  MaxDD (liq base):    {res['best_maxdd']*100:.1f}%")
    lines.append(f"  OOS Sharpe 1a/2a metade: {res['oos_sharpe_h1']:.2f} / {res['oos_sharpe_h2']:.2f}")
    lines.append("")
    lines.append("TRIBUNAL ESTATISTICO (sobre o melhor, estressado):")
    lines.append(f"  DSR = {res['dsr']:.3f}   (barra >= 0.95)")
    lines.append(f"  PSR = {res['psr']:.3f}")
    lines.append(f"  Obstaculo data-snooping (Sharpe anual): {res['sr_hurdle_annual']:.2f}")
    lines.append(f"  PBO (CSCV sobre as {res['n_trials']} configs) = {res['pbo']:.3f}   (barra < 0.5)")
    lines.append("")
    lines.append("ENSEMBLE equal-weight das 8 configs:")
    lines.append(f"  Sharpe net base={res['ens_sharpe_net']:.2f}  stress2x={res['ens_sharpe_stress']:.2f}  DSR={res['ens_dsr']:.3f}")
    lines.append(f"  CAGR={res['ens_cagr']*100:.1f}%  MaxDD={res['ens_maxdd']*100:.1f}%")
    lines.append("")
    lines.append("BENCHMARK buy&hold BTC (mesma janela):")
    lines.append(f"  Sharpe={res['bh_sharpe']:.2f}  CAGR={res['bh_cagr']*100:.1f}%  MaxDD={res['bh_maxdd']*100:.1f}%")
    lines.append("")
    lines.append("ESTRESSE DE CORRELACAO:")
    cs = res["corr_spy"]
    cb = res["corr_btc"]
    lines.append(f"  corr ensemble vs SPY: {cs if cs is None else round(cs,3)}")
    lines.append(f"  corr melhor vs BTC buy&hold: {cb if cb is None else round(cb,3)}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("CHECKLIST DA BARRA:")
    lines.append(f"  [{'x' if res['dsr_ok'] else ' '}] DSR >= 0.95")
    lines.append(f"  [{'x' if res['sharpe_robust'] else ' '}] Sharpe estressado >= 0.8 E OOS positivo nas 2 metades")
    lines.append(f"  [{'x' if res['pbo_ok'] else ' '}] PBO < 0.5")
    lines.append(f"  [{'x' if res['beats_bh'] else ' '}] bate buy&hold BTC (Sharpe e ~CAGR)")
    lines.append(f"  [{'x' if res['low_corr_diversifies'] else ' '}] OU diversifica (|corr SPY|<0.3 e Sharpe>0.5)")
    lines.append("")
    verdict = "PASSA" if res["passed"] else "FALHA"
    lines.append("=" * 78)
    lines.append(f"VEREDITO FINAL: {verdict}")
    lines.append("=" * 78)
    REPORT.write_text("\n".join(lines))


def main() -> None:
    if "--download" in sys.argv:
        print("Baixando klines 1d (data.binance.vision)...")
        r = download()
        print(f"  OK:{len(r['ok'])}  cache:{len(r['cached'])}  FALHA:{len(r['fail'])}")
        if r["fail"]:
            print("  amostra de falhas:", r["fail"][:10])
        return
    res = run_tribunal()
    write_report(res)
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
