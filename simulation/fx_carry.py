"""TRIBUNAL DE FX CARRY (G10) — long juro alto / short juro baixo, vs USD.

A ESTRATEGIA (carry de moedas, o premio mais classico de FX):
  Toda moeda de G10 carrega um juro de curto prazo. Comprar a moeda de juro ALTO e
  vender a de juro BAIXO captura o diferencial de juro (o "carry"). A teoria (UIP) diz
  que a moeda de juro alto deveria se DEPRECIAR o suficiente para anular o ganho de juro;
  na pratica isso NAO acontece em media (forward premium puzzle) -> sobra um premio. O
  preco desse premio e a cauda: em RISK-OFF as moedas de carry (AUD/NZD) desabam juntas
  enquanto JPY/CHF (funding) disparam -> o "unwind" do carry. Por isso testamos o estresse.

CONSTRUCAO (cross-sectional, sem look-ahead):
  - Universo G10 vs USD: EUR, JPY, GBP, AUD, NZD, CAD, CHF, SEK, NOK (+ USD como base).
  - Sinal em t = ranking pelo diferencial de juro de t-1 (carry conhecido no fim do mes
    anterior). Retorno realizado em t+1 (mes seguinte). NENHUM dado de t entra no sinal de t.
  - Retorno total de manter a moeda i por 1 mes vs USD =
        variacao do spot (FX_i em USD) + carry mensal (r_i - r_us)/12.
    (CIP: forward premium ~ diferencial de juro; usamos o diferencial de juro de 3 meses
     como proxy do carry tradeable. Para G10 a CIP segura bem historicamente.)
  - Long-short: long top-k moedas de maior carry, short bottom-k (equal weight). Dollar-neutro.
  - SIZING POR VOL: escala a carteira para uma vol-alvo anual (vol estimada de t-1, sem look-ahead).
  - Variante long-only-top tambem reportada (mais beta, contraste).

DADOS (GRATIS, REAIS):
  - FX spot mensal (fim de mes): yfinance pares USD.
  - Juros de 3 meses (interbank/money-market, OECD via FRED, endpoint CSV publico SEM chave):
        IR3TIB01{CC}M156N  (US, EZ, JP, GB, AU, NZ, CA, CH, SE, NO).
  Se a rede bloquear: data_status="blocked" e o relatorio diz o comando — NAO inventamos numero.

CUSTO: G10 spot tem spread baixo; modelamos custo por rebalance sobre o turnover.
  Base: ~2 bps/lado de fricao de execucao em G10 (spread+slippage institucional). Estresse 2x.

VEREDITO (barra de graduacao, pre-registrada): PASSA so se
  DSR >= 0.95 (n_trials HONESTO conta todas as variantes) E Sharpe liq robusto E
  (bate USD/buy&hold OU diversifica com correlacao BAIXA ao SPY melhorando o conjunto),
  robusto OOS (split temporal). Estresse de unwind tem de ser sobrevivivel.

Uso:
    uv run python -m simulation.fx_carry --download   # baixa FX + juros (precisa rede)
    uv run python -m simulation.fx_carry              # roda o tribunal (usa cache)
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.metrics import max_drawdown
from simulation.statistics import (
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# --------------------------------------------------------------------------- #
# Universo e config
# --------------------------------------------------------------------------- #
# G10 vs USD. Mapeia cada moeda -> (ticker yahoo, "invert": True se o ticker e USDxxx,
# i.e. precisa inverter para virar xxx/USD = valor de 1 unidade da moeda em USD).
# EURUSD=X, GBPUSD=X, AUDUSD=X, NZDUSD=X ja sao xxx/USD (subir = moeda forte).
# JPY=X, CAD=X, CHF=X, SEK=X, NOK=X sao USD/xxx (subir = moeda fraca) -> inverter.
FX_TICKERS = {
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "AUD": ("AUDUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "JPY": ("JPY=X", True),
    "CAD": ("CAD=X", True),
    "CHF": ("CHF=X", True),
    "SEK": ("SEK=X", True),
    "NOK": ("NOK=X", True),
}
# Juros de 3 meses (OECD interbank, FRED), incluindo US (a base).
RATE_FRED = {
    "US": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
    "SEK": "IR3TIB01SEM156N",
    "NOK": "IR3TIB01NOM156N",
}

CACHE = Path("data/fx_carry_cache")
FX_CACHE = CACHE / "fx_monthly.csv.gz"
RATE_CACHE = CACHE / "rates_monthly.csv.gz"
SPY_CACHE = CACHE / "spy_monthly.csv.gz"

START = "2003-01-01"
END = "2024-12-31"
MONTHS_PER_YEAR = 12
COST_BPS_PER_SIDE_BASE = 2.0  # G10 spot, institucional
VOL_TARGET_ANN = 0.10  # 10%/ano (carteira long-short)


# --------------------------------------------------------------------------- #
# Download (REAL, gratis)
# --------------------------------------------------------------------------- #
def _fred_csv(series_id: str) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={START}&coed={END}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def download_all() -> dict[str, list[str]]:
    """Baixa FX mensal (yfinance) + juros 3m (FRED) + SPY mensal. Idempotente."""
    CACHE.mkdir(parents=True, exist_ok=True)
    res: dict[str, list[str]] = {"ok": [], "fail": []}

    # ---- juros (FRED) ----
    try:
        rates = {}
        for cur, sid in RATE_FRED.items():
            s = _fred_csv(sid)
            s.index = s.index.to_period("M").to_timestamp("M")  # fim de mes
            rates[cur] = s
            print(f"  juro {cur} ({sid}): {len(s)} meses")
        rdf = pd.DataFrame(rates).sort_index()
        rdf.to_csv(RATE_CACHE, compression="gzip")
        res["ok"].append(f"rates {rdf.shape}")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        res["fail"].append(f"rates: {type(e).__name__} {str(e)[:80]}")
        print(f"  FALHA juros: {type(e).__name__}")

    # ---- FX (yfinance) ----
    try:
        import yfinance as yf

        tickers = [t for t, _ in FX_TICKERS.values()]
        raw = yf.download(
            tickers, start=START, end=END, interval="1d",
            progress=False, auto_adjust=True,
        )["Close"]
        # fim de mes
        monthly = raw.resample("ME").last()
        # converter cada ticker para xxx/USD (valor de 1 unidade da moeda em USD)
        fx = {}
        for cur, (tic, invert) in FX_TICKERS.items():
            if tic not in monthly.columns:
                res["fail"].append(f"fx {cur}: ticker {tic} ausente")
                continue
            ser = monthly[tic].astype(float)
            fx[cur] = (1.0 / ser) if invert else ser
        fxdf = pd.DataFrame(fx).sort_index()
        fxdf.to_csv(FX_CACHE, compression="gzip")
        res["ok"].append(f"fx {fxdf.shape}")
        print(f"  fx: {fxdf.shape}")

        spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)["Close"]
        spym = spy.resample("ME").last()
        spym.to_csv(SPY_CACHE, compression="gzip")
        res["ok"].append("spy")
    except Exception as e:  # yfinance levanta varios tipos
        res["fail"].append(f"fx/spy: {type(e).__name__} {str(e)[:80]}")
        print(f"  FALHA fx/spy: {type(e).__name__} {e}")

    return res


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.Series | None]:
    fx = rates = spy = None
    if FX_CACHE.exists():
        fx = pd.read_csv(FX_CACHE, index_col=0, parse_dates=True)
    if RATE_CACHE.exists():
        rates = pd.read_csv(RATE_CACHE, index_col=0, parse_dates=True)
    if SPY_CACHE.exists():
        spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    return fx, rates, spy


# --------------------------------------------------------------------------- #
# Motor do carry — cross-sectional, sem look-ahead
# --------------------------------------------------------------------------- #
@dataclass
class StratResult:
    name: str
    monthly_returns: np.ndarray   # liquido de custo, ja vol-scaled (se aplicavel)
    n_months: int
    sharpe_ann: float
    cagr: float
    maxdd: float
    vol_ann: float
    turnover_mean: float
    corr_spy: float


def _build_monthly_panel(
    fx: pd.DataFrame, rates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Alinha FX e juros num indice mensal comum.

    Retorna:
      fx_ret[t, cur]   = retorno do SPOT da moeda vs USD no mes t (FX_t/FX_{t-1} - 1)
      carry[t, cur]    = carry mensal conhecido NO FIM de t (r_cur - r_us)/1200  (% -> fracao/mes)
      tot_ret[t, cur]  = fx_ret[t] + carry[t-1]  (carry conhecido em t-1, ganho realizado em t)
    """
    idx = fx.index.intersection(rates.index)
    fx = fx.loc[idx].sort_index()
    rates = rates.loc[idx].sort_index()
    curs = [c for c in FX_TICKERS if c in fx.columns and c in rates.columns]

    fx = fx[curs]
    fx_ret = fx.pct_change()

    # carry mensal = (juro moeda - juro US)/12, em fracao. rates em % ao ano.
    r_us = rates["US"]
    carry = pd.DataFrame(
        {c: (rates[c] - r_us) / 1200.0 for c in curs}, index=idx
    )
    # retorno total de SEGURAR a moeda no mes t = spot ret(t) + carry conhecido em t-1
    tot_ret = fx_ret + carry.shift(1)
    return fx_ret, carry, tot_ret


def _vol_scale(raw_ret: pd.Series, target_ann: float, lookback: int = 12) -> tuple[np.ndarray, pd.Series]:
    """Escala a serie para vol-alvo usando vol realizada de t-1 (sem look-ahead)."""
    realized = raw_ret.rolling(lookback).std().shift(1) * np.sqrt(MONTHS_PER_YEAR)
    lev = (target_ann / realized).clip(upper=5.0)  # teto de alavancagem 5x
    scaled = (raw_ret * lev)
    return scaled.to_numpy(float), lev


def run_strategy(
    tot_ret: pd.DataFrame,
    carry: pd.DataFrame,
    fx_ret: pd.DataFrame,
    spy_ret: pd.Series,
    *,
    k: int,
    long_only: bool,
    vol_scaled: bool,
    cost_bps_per_side: float,
) -> StratResult:
    """Long top-k carry / short bottom-k (ou long-only top-k). Sinal de t-1, retorno t.

    k: quantas moedas em cada perna. long_only: so a perna long (top-k).
    vol_scaled: escala para VOL_TARGET_ANN com vol de t-1.
    """
    name = (
        f"{'LONGONLY' if long_only else 'LS'}_top{k}"
        f"{'_vol' if vol_scaled else '_raw'}"
    )
    # ranking pelo carry CONHECIDO no fim de t-1 (carry.shift(1) ja esta defasado no tot_ret;
    # aqui rankeamos por carry de t-1 explicitamente para montar pesos aplicados em t).
    signal = carry.shift(1)  # carry de t-1 -> decide a posicao para o mes t

    rets = []
    weights_prev = None
    turnovers = []
    idx = tot_ret.index
    for t in range(len(idx)):
        sig = signal.iloc[t]
        rr = tot_ret.iloc[t]
        valid = sig.dropna().index.intersection(rr.dropna().index)
        if len(valid) < 2 * k if not long_only else len(valid) < k:
            rets.append(0.0)
            continue
        sig_v = sig[valid].sort_values(ascending=False)
        longs = list(sig_v.index[:k])
        w = pd.Series(0.0, index=valid)
        if long_only:
            w[longs] = 1.0 / k
        else:
            shorts = list(sig_v.index[-k:])
            w[longs] = 1.0 / k
            w[shorts] = -1.0 / k
        # turnover vs mes anterior
        if weights_prev is not None:
            allc = w.index.union(weights_prev.index)
            wn = w.reindex(allc).fillna(0.0)
            wp = weights_prev.reindex(allc).fillna(0.0)
            turnover = float((wn - wp).abs().sum())
        else:
            turnover = float(w.abs().sum())
        turnovers.append(turnover)
        gross = float((w * rr[valid]).sum())
        cost = turnover * cost_bps_per_side / 1e4
        rets.append(gross - cost)
        weights_prev = w

    raw = pd.Series(rets, index=idx)
    if vol_scaled:
        scaled_arr, _ = _vol_scale(raw, VOL_TARGET_ANN)
        ser = pd.Series(scaled_arr, index=idx)
    else:
        ser = raw
    ser = ser.dropna()
    if ser.size < 12:
        return StratResult(name, ser.to_numpy(), ser.size, 0, 0, 0, 0, 0, 0)

    arr = ser.to_numpy(float)
    sharpe = observed_sharpe(arr, periods_per_year=MONTHS_PER_YEAR)
    eq = np.cumprod(1.0 + arr)
    years = ser.size / MONTHS_PER_YEAR
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if eq[-1] > 0 else -1.0
    mdd = max_drawdown(eq)
    vol = float(arr.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    corr = 0.0
    if spy_ret is not None:
        common = ser.index.intersection(spy_ret.index)
        if len(common) > 12:
            a = ser.loc[common].to_numpy(float)
            b = spy_ret.loc[common].to_numpy(float)
            if a.std() > 0 and b.std() > 0:
                corr = float(np.corrcoef(a, b)[0, 1])

    return StratResult(
        name=name, monthly_returns=arr, n_months=ser.size, sharpe_ann=sharpe,
        cagr=cagr, maxdd=mdd, vol_ann=vol,
        turnover_mean=float(np.mean(turnovers)) if turnovers else 0.0, corr_spy=corr,
    )


# --------------------------------------------------------------------------- #
# Stress: unwind do carry em risk-off (piores meses do SPY)
# --------------------------------------------------------------------------- #
def stress_unwind(ser: pd.Series, spy_ret: pd.Series) -> dict:
    """Mede o comportamento do carry nos piores meses de risco (risk-off)."""
    common = ser.index.intersection(spy_ret.index)
    if len(common) < 24:
        return {}
    a = ser.loc[common]
    b = spy_ret.loc[common]
    worst = b.nsmallest(max(int(len(b) * 0.10), 6)).index  # pior decil de risk-off
    carry_in_riskoff = a.loc[worst]
    # beta nos risk-off vs geral
    return {
        "carry_mean_riskoff": float(carry_in_riskoff.mean()),
        "carry_mean_all": float(a.mean()),
        "worst_single_month": float(a.min()),
        "carry_in_worst_spy_month": float(a.loc[b.idxmin()]),
        "n_riskoff": int(len(worst)),
        "downside_capture": float(
            carry_in_riskoff.mean() / b.loc[worst].mean()
        ) if b.loc[worst].mean() != 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Relatorio + veredito
# --------------------------------------------------------------------------- #
def render_report(
    results: list[StratResult],
    primary: StratResult,
    verdict,
    pbo: float,
    oos: dict,
    stress: dict,
    spy_stats: dict,
    n_trials: int,
    cost_label: str,
    passed: bool,
    reasons: list[str],
) -> str:
    L = []
    L.append("=" * 100)
    L.append("TRIBUNAL DE FX CARRY (G10) — long juro alto / short juro baixo, vs USD")
    L.append("=" * 100)
    L.append("")
    L.append("DADOS: REAIS. FX spot mensal (yfinance) + juros 3m interbank OECD (FRED, sem chave).")
    L.append("  Proxy: carry = diferencial de juro 3m / 12 (CIP segura bem em G10). Sem look-ahead:")
    L.append("  sinal de t usa carry/vol de t-1; retorno realizado em t.")
    L.append(f"CUSTO: {cost_label} sobre o turnover. Sizing por vol (alvo {VOL_TARGET_ANN*100:.0f}%/ano, vol de t-1).")
    L.append("")
    L.append("VEREDITO (barra pre-registrada): PASSA so se DSR>=0.95 (n_trials honesto) E Sharpe")
    L.append("  liq robusto E (bate USD/buy&hold OU diversifica c/ corr BAIXA ao SPY) E robusto OOS")
    L.append("  E unwind sobrevivivel.")
    L.append("")
    L.append("-" * 100)
    L.append("VARIANTES TESTADAS (todas contam no n_trials do DSR):")
    L.append("-" * 100)
    hdr = (f"{'variante':>20} {'meses':>6} {'Sharpe':>7} {'CAGR%':>7} {'vol%':>6} "
           f"{'MaxDD%':>8} {'turnov':>7} {'corrSPY':>8}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in results:
        flag = "  <== PRIMARIA" if r.name == primary.name else ""
        L.append(
            f"{r.name:>20} {r.n_months:>6} {r.sharpe_ann:>7.2f} {r.cagr*100:>7.2f} "
            f"{r.vol_ann*100:>6.2f} {r.maxdd*100:>8.2f} {r.turnover_mean:>7.2f} "
            f"{r.corr_spy:>8.2f}{flag}"
        )
    L.append("")
    L.append(f"Benchmark USD (cash, sleeve sem risco) ~ 0% real; SPY buy&hold: "
             f"Sharpe={spy_stats.get('sharpe',0):.2f} CAGR={spy_stats.get('cagr',0)*100:.2f}%")
    L.append("")
    L.append("=" * 100)
    L.append(f"TRIBUNAL ESTATISTICO sobre a PRIMARIA ({primary.name}), custo ESTRESSADO:")
    L.append("=" * 100)
    L.append(f"  n_obs={verdict.n_obs}  n_trials(honesto)={n_trials}")
    L.append(f"  Sharpe_anual = {verdict.sharpe_annual:.3f}  (obstaculo data-snooping = {verdict.sr_benchmark_annual:.3f})")
    L.append(f"  PSR = {verdict.psr:.3f}   DSR = {verdict.dsr:.3f}   (barra DSR>=0.95)")
    L.append(f"  skew = {verdict.skew:.2f}  kurtose = {verdict.kurtosis:.2f}  (carry: skew NEGATIVO esperado)")
    L.append(f"  PBO (CSCV) = {pbo:.3f}   (barra < 0.5; quanto menor, melhor)")
    L.append("")
    L.append("OOS (split temporal 60/40):")
    L.append(f"  IS  : Sharpe={oos.get('is_sharpe',0):.2f}  CAGR={oos.get('is_cagr',0)*100:.2f}%")
    L.append(f"  OOS : Sharpe={oos.get('oos_sharpe',0):.2f}  CAGR={oos.get('oos_cagr',0)*100:.2f}%")
    L.append("")
    L.append("STRESS — UNWIND do carry em RISK-OFF (pior decil de meses do SPY):")
    if stress:
        L.append(f"  retorno medio do carry em risk-off = {stress['carry_mean_riskoff']*100:+.2f}%/mes "
                 f"(vs {stress['carry_mean_all']*100:+.2f}%/mes no geral)")
        L.append(f"  pior mes isolado do carry = {stress['worst_single_month']*100:+.2f}%")
        L.append(f"  carry no PIOR mes do SPY = {stress['carry_in_worst_spy_month']*100:+.2f}%")
        L.append(f"  downside capture vs SPY (risk-off) = {stress['downside_capture']:.2f}")
        L.append("  Leitura: carry tem cauda esquerda (desaba junto com risco). Se downside_capture > 0")
        L.append("  e alto, o carry AMPLIFICA o risk-off (o classico unwind) — pessimo p/ diversificar.")
    else:
        L.append("  (dados insuficientes p/ estresse)")
    L.append("")
    L.append("=" * 100)
    L.append("VEREDITO FINAL:")
    L.append("=" * 100)
    if passed:
        L.append("  >>> PASSA <<<  o FX carry (G10) demonstra edge robusto na barra de graduacao.")
    else:
        L.append("  ##############################################################################")
        L.append("  #  REPROVADO — FX carry (G10) NAO cumpre a barra de graduacao                #")
        L.append("  ##############################################################################")
    for rs in reasons:
        L.append(f"  - {rs}")
    L.append("")
    return "\n".join(L) + "\n"


def _no_data_report() -> str:
    return (
        "=" * 100 + "\n"
        "TRIBUNAL DE FX CARRY (G10) — DADOS PENDENTES\n"
        + "=" * 100 + "\n\n"
        "Harness pronto e idempotente, faltam FX e/ou juros. Rode com rede:\n"
        "    uv run python -m simulation.fx_carry --download\n"
        "depois:\n"
        "    uv run python -m simulation.fx_carry\n\n"
        f"Cache: {CACHE}\n"
        "Fontes (gratis, sem chave):\n"
        "  FX  : yfinance pares USD (EURUSD=X, JPY=X, ...)\n"
        "  juro: FRED CSV publico IR3TIB01{CC}M156N (3m interbank OECD)\n"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tribunal de FX carry (G10)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--report-file", default="data/fx_carry_verdict.txt")
    args = p.parse_args(argv)

    if args.download:
        CACHE.mkdir(parents=True, exist_ok=True)
        r = download_all()
        print(f"OK={r['ok']}  FALHA={r['fail']}")
        return 0 if r["ok"] else 1

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    fx, rates, spy = load_data()
    if fx is None or rates is None:
        text = _no_data_report()
        print(text)
        out.write_text(text, encoding="utf-8")
        return 0

    fx_ret, carry, tot_ret = _build_monthly_panel(fx, rates)
    spy_ret = spy.pct_change().dropna() if spy is not None else None

    spy_stats = {}
    if spy_ret is not None and spy_ret.size > 12:
        sa = spy_ret.to_numpy(float)
        eqs = np.cumprod(1.0 + sa)
        spy_stats = {
            "sharpe": observed_sharpe(sa, periods_per_year=MONTHS_PER_YEAR),
            "cagr": float(eqs[-1] ** (MONTHS_PER_YEAR / sa.size) - 1.0),
        }

    base_cost = COST_BPS_PER_SIDE_BASE
    stress_cost = base_cost * 2.0

    # ---- grade de variantes (TODAS contam no n_trials) ----
    # k in {1,2,3} x {long-short, long-only} x {vol-scaled, raw} = 12 variantes
    results: list[StratResult] = []
    matrix_cols: list[np.ndarray] = []  # p/ PBO: series alinhadas das variantes LS vol-scaled em k
    for k in (1, 2, 3):
        for long_only in (False, True):
            for vol_scaled in (True, False):
                r = run_strategy(
                    tot_ret, carry, fx_ret, spy_ret,
                    k=k, long_only=long_only, vol_scaled=vol_scaled,
                    cost_bps_per_side=base_cost,
                )
                results.append(r)
    n_trials = len(results)

    # primaria pre-registrada: long-short, k=2, vol-scaled (a config canonica de carry)
    primary_base = next(r for r in results if r.name == "LS_top2_vol")

    # PBO sobre a matriz das variantes (alinhar pelo menor tamanho comum)
    minlen = min(r.monthly_returns.size for r in results if r.monthly_returns.size > 12)
    mat = np.column_stack([r.monthly_returns[-minlen:] for r in results
                           if r.monthly_returns.size >= minlen])
    pbo = probability_of_backtest_overfitting(mat, n_splits=10)

    # tribunal na PRIMARIA com CUSTO ESTRESSADO (honesto)
    primary_stress = run_strategy(
        tot_ret, carry, fx_ret, spy_ret,
        k=2, long_only=False, vol_scaled=True, cost_bps_per_side=stress_cost,
    )
    verdict = evaluate_edge(
        primary_stress.monthly_returns,
        n_trials=n_trials,
        periods_per_year=MONTHS_PER_YEAR,
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # OOS split temporal 60/40 na primaria (custo base)
    pr = primary_base.monthly_returns
    cut = int(len(pr) * 0.6)
    is_r, oos_r = pr[:cut], pr[cut:]
    def _st(a):
        if a.size < 6:
            return 0.0, 0.0
        sh = observed_sharpe(a, periods_per_year=MONTHS_PER_YEAR)
        eq = np.cumprod(1.0 + a)
        cg = float(eq[-1] ** (MONTHS_PER_YEAR / a.size) - 1.0) if eq[-1] > 0 else -1.0
        return sh, cg
    is_sh, is_cg = _st(is_r)
    oos_sh, oos_cg = _st(oos_r)
    oos = {"is_sharpe": is_sh, "is_cagr": is_cg, "oos_sharpe": oos_sh, "oos_cagr": oos_cg}

    # stress unwind na primaria
    primary_ser = pd.Series(primary_base.monthly_returns, index=tot_ret.index[-primary_base.n_months:])
    stress = stress_unwind(primary_ser, spy_ret) if spy_ret is not None else {}

    # ---- decisao da barra ----
    reasons: list[str] = []
    passes_dsr = verdict.dsr >= 0.95
    passes_sharpe = primary_stress.sharpe_ann >= 0.8
    passes_pbo = pbo < 0.5
    passes_oos = oos_sh > 0.3  # OOS Sharpe positivo e nao-trivial
    # diversifica? corr baixa ao SPY E Sharpe positivo, OU bate buy&hold em risco
    low_corr = abs(primary_base.corr_spy) < 0.3
    beats_bh = primary_base.sharpe_ann > spy_stats.get("sharpe", 0)
    diversifies = (low_corr and primary_stress.sharpe_ann > 0.5) or beats_bh
    # unwind sobrevivivel: pior mes nao-ruinoso vs vol-alvo (carry tem cauda; exigimos < -15%/mes)
    unwind_ok = bool(stress) and stress["worst_single_month"] > -0.15

    if not passes_dsr:
        reasons.append(f"DSR={verdict.dsr:.3f} < 0.95 (Sharpe nao supera o obstaculo de data-snooping)")
    if not passes_sharpe:
        reasons.append(f"Sharpe liq estressado={primary_stress.sharpe_ann:.2f} < 0.8")
    if not passes_pbo:
        reasons.append(f"PBO={pbo:.3f} >= 0.5 (selecao escolhe sorte, nao skill)")
    if not passes_oos:
        reasons.append(f"OOS fraco (Sharpe OOS={oos_sh:.2f})")
    if not diversifies:
        reasons.append(f"nao diversifica nem bate buy&hold (corrSPY={primary_base.corr_spy:.2f}, "
                       f"Sharpe={primary_base.sharpe_ann:.2f} vs SPY {spy_stats.get('sharpe',0):.2f})")
    if not unwind_ok:
        wm = stress.get("worst_single_month", 0) if stress else 0
        reasons.append(f"unwind perigoso (pior mes {wm*100:.1f}%)")

    passed = passes_dsr and passes_sharpe and passes_pbo and passes_oos and diversifies and unwind_ok
    if passed:
        reasons.append(f"DSR={verdict.dsr:.3f}, Sharpe_liq={primary_stress.sharpe_ann:.2f}, "
                       f"PBO={pbo:.3f}, OOS Sharpe={oos_sh:.2f}, corrSPY={primary_base.corr_spy:.2f}")

    cost_label = (f"base {base_cost:.0f}bps/lado, ESTRESSADO {stress_cost:.0f}bps/lado")
    text = render_report(
        results, primary_base, verdict, pbo, oos, stress, spy_stats,
        n_trials, cost_label, passed, reasons,
    )
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")

    # imprime um bloco-resumo legivel por maquina
    print("\n@@VERDICT@@")
    print(f"data_status=real")
    print(f"passed={passed}")
    print(f"sharpe={primary_stress.sharpe_ann:.4f}")
    print(f"cagr={primary_base.cagr:.4f}")
    print(f"maxdd={primary_base.maxdd:.4f}")
    print(f"dsr={verdict.dsr:.4f}")
    print(f"pbo={pbo:.4f}")
    print(f"corr_spy={primary_base.corr_spy:.4f}")
    print(f"n_trials={n_trials}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
