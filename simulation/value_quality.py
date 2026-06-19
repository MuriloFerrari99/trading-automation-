"""Tribunal de FATORES FUNDAMENTALISTAS: VALUE + QUALITY (R&D offline, isolado).

PERGUNTA (imparcial, MEDIR): um sleeve cross-seccional de VALUE (P/E, P/B baixos)
+ QUALITY (ROE, margens altas, baixa alavancagem) num universo liquido US tem edge
risco-ajustado ROBUSTO **e** bate/diversifica o benchmark (equal-weight / SPY)?

R&D PURO, em paralelo ao beta vivo. NAO toca em beta_*, main.py, nav_history nem
config de producao. Arquivo NOVO. Importa (read-only) apenas:
  - simulation.costs      : EQUITY_BASE (custo real de equity) + cenario estressado 2x
  - simulation.statistics : DSR/PSR/PBO/observed_sharpe (tribunal anti-overfitting)
  - simulation.metrics    : max_drawdown

DADOS (gratis, yfinance, cacheado em data/value_quality_cache/):
  - Precos: close ajustado, period=max.
  - FUNDAMENTOS: demonstracoes TRIMESTRAIS (income + balanco) datadas, do yfinance.
    Importante: yfinance NAO entrega ratios point-in-time historicos; entrega as
    DEMONSTRACOES datadas (Net Income, Revenue, Equity, Total Debt, Shares, Assets).
    Construimos os fatores a partir DELAS, na data de cada demonstracao.

LIMITACAO DE DADO (honesta): demonstracoes trimestrais gratis do yfinance vao ~5-7
trimestres p/ tras; anuais ~4-5 anos. NAO ha dado PIT profundo gratuito. Por isso o
historico fundamentalista util e CURTO (~3-5 anos), o que reduz o N do tribunal.
Reportamos data_status conforme o que o fetch realmente trouxer.

SEM LOOK-AHEAD (rigoroso):
  - Cada demonstracao (data fiscal d) so vira disponivel apos um LAG de reporte
    (REPORT_LAG_DAYS=90, conservador). Na data de rebalance t, usamos a demonstracao
    mais recente com (d + lag) <= t.
  - P/E e P/B usam EARNINGS/EQUITY da demonstracao (passado) e PRECO em t (conhecido).
  - Pesos decididos em t aplicados aos retornos t+1..proximo rebalance (weights.shift).

FATORES (cross-seccional, z-score, rebalance TRIMESTRAL):
  VALUE   = media(z(-P/E), z(-P/B))      [barato = score alto]
  QUALITY = media(z(ROE), z(net margin), z(-debt/equity))
  COMBO   = media(z(VALUE), z(QUALITY))
  Variantes: LONG-ONLY (long top tercil/quartil, EW) e LONG-SHORT (top-bottom, $-neutral).

CUSTO: EQUITY_BASE (3 bps/lado) + estressado 2x sobre |Delta peso| (turnover). VEREDITO
no cenario ESTRESSADO.

n_TRIALS HONESTO: fatores {value, quality, combo} x variantes {long-only, long-short}
x cortes {tercil, quartil} = 12 configs. Contado no DSR; PBO sobre a matriz das 12.

BARRA: PASSA so se DSR>=0.95 E Sharpe liq robusto E (bate buy&hold/EW OU diversifica
com corr baixa ao SPY melhorando o conjunto), robusto. Senao FALHA.

Uso:
    uv run python -m simulation.value_quality            # usa cache
    uv run python -m simulation.value_quality --force    # re-baixa
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("simulation.value_quality")

CACHE_DIR = Path("data/value_quality_cache")
REPORT_PATH = Path("data/value_quality_verdict.txt")
TRADING_DAYS = EQUITY_PERIODS  # 252
REPORT_LAG_DAYS = 90  # atraso de divulgacao (conservador, anti look-ahead)

# Universo: ~44 nomes liquidos large/mega-cap US, varios setores (cross-section rico).
UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "ORCL",  # tech
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",                      # financials
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "BMY",                 # healthcare
    "PG", "KO", "PEP", "WMT", "COST", "MCD",                          # staples/cons
    "XOM", "CVX", "COP",                                              # energy
    "HD", "NKE", "DIS", "SBUX", "LOW",                                # consumer disc
    "CAT", "BA", "HON", "GE",                                         # industrials
    "VZ", "T", "INTC", "CSCO",                                        # telecom/tech
]
SPY = "SPY"


# ============================================================================
# DADOS
# ============================================================================
def _fetch_prices(force: bool) -> pd.DataFrame:
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "prices.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df
    series: dict[str, pd.Series] = {}
    for tk in [*UNIVERSE, SPY]:
        try:
            h = yf.Ticker(tk).history(period="max", auto_adjust=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preco %s falhou: %s", tk, exc)
            continue
        if h.empty:
            continue
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        series[tk] = s
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index().ffill(limit=2)
    panel.to_csv(cache)
    return panel


def _fetch_fundamentals(force: bool) -> dict[str, pd.DataFrame]:
    """Por ticker: DataFrame indexado por data fiscal com colunas dos fatores brutos.

    Combina income trimestral (Net Income TTM, Revenue TTM) com balanco trimestral
    (Stockholders Equity, Total Debt, Shares). TTM = soma dos 4 trimestres mais
    recentes ate a data, p/ evitar sazonalidade. Cacheado por ticker.
    """
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for tk in UNIVERSE:
        cache = CACHE_DIR / f"fund_{tk}.csv"
        if cache.exists() and not force:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if not df.empty:
                out[tk] = df
            continue
        try:
            t = yf.Ticker(tk)
            qinc = t.quarterly_financials
            qbs = t.quarterly_balance_sheet
            ainc = t.financials       # anual (~4-5 anos)
            abs_ = t.balance_sheet    # anual
        except Exception as exc:  # noqa: BLE001
            logger.warning("fund %s falhou: %s", tk, exc)
            continue

        def _row(df, name: str) -> pd.Series:
            if df is not None and not df.empty and name in df.index:
                s = df.loc[name]
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
            return pd.Series(dtype=float)

        # ---- TRIMESTRAL (recente, alta resolucao): income TTM + balanco do trimestre
        net_income_q = _row(qinc, "Net Income")
        revenue_q = _row(qinc, "Total Revenue")
        ni_ttm = (net_income_q.rolling(4, min_periods=2).sum()
                  * (4.0 / net_income_q.rolling(4, min_periods=2).count())) if not net_income_q.empty else pd.Series(dtype=float)
        rev_ttm = (revenue_q.rolling(4, min_periods=2).sum()
                   * (4.0 / revenue_q.rolling(4, min_periods=2).count())) if not revenue_q.empty else pd.Series(dtype=float)
        eq_q = _row(qbs, "Stockholders Equity")
        if eq_q.empty:
            eq_q = _row(qbs, "Common Stock Equity")
        debt_q = _row(qbs, "Total Debt")
        sh_q = _row(qbs, "Ordinary Shares Number")
        if sh_q.empty:
            sh_q = _row(qbs, "Share Issued")

        # ---- ANUAL (profundo, ~4-5 anos): income do ano (ja "TTM") + balanco anual
        ni_a = _row(ainc, "Net Income")
        rev_a = _row(ainc, "Total Revenue")
        eq_a = _row(abs_, "Stockholders Equity")
        if eq_a.empty:
            eq_a = _row(abs_, "Common Stock Equity")
        debt_a = _row(abs_, "Total Debt")
        sh_a = _row(abs_, "Ordinary Shares Number")
        if sh_a.empty:
            sh_a = _row(abs_, "Share Issued")

        # combina: anual (profundo) + trimestral (recente). Em datas conflitantes,
        # o trimestral (mais granular/recente) prevalece.
        def _combine(a: pd.Series, q: pd.Series) -> pd.Series:
            parts = [s for s in (a, q) if not s.empty]
            if not parts:
                return pd.Series(dtype=float)
            m = pd.concat(parts)
            m = m[~m.index.duplicated(keep="last")].sort_index()
            return m

        net_income = _combine(ni_a, ni_ttm)
        revenue = _combine(rev_a, rev_ttm)
        equity = _combine(eq_a, eq_q)
        total_debt = _combine(debt_a, debt_q)
        shares = _combine(sh_a, sh_q)

        if net_income.empty or equity.empty or shares.empty:
            continue

        idx = sorted(set(equity.index) | set(shares.index) | set(net_income.index))
        rec = pd.DataFrame(index=pd.DatetimeIndex(idx))
        rec["net_income_ttm"] = net_income.reindex(rec.index).ffill()
        rec["revenue_ttm"] = revenue.reindex(rec.index).ffill()
        rec["equity"] = equity.reindex(rec.index).ffill()
        rec["total_debt"] = total_debt.reindex(rec.index).ffill() if not total_debt.empty else 0.0
        rec["shares"] = shares.reindex(rec.index).ffill()
        rec = rec.dropna(subset=["net_income_ttm", "equity", "shares"])
        rec = rec[rec["shares"] > 0]
        if rec.empty:
            continue
        rec.to_csv(cache)
        out[tk] = rec
    return out


# ============================================================================
# FATORES (PIT, sem look-ahead)
# ============================================================================
def _pit_snapshot(fund: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    """Para cada ticker, a demonstracao mais recente com (data_fiscal + lag) <= as_of."""
    cutoff = as_of - pd.Timedelta(days=REPORT_LAG_DAYS)
    rows = {}
    for tk, df in fund.items():
        avail = df[df.index <= cutoff]
        if avail.empty:
            continue
        rows[tk] = avail.iloc[-1]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-3, 3)


def _factor_scores(snap: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """snap: demonstracoes PIT (index=tickers). price: close em as_of por ticker.

    Retorna DataFrame com colunas value, quality, combo (z-scores cross-seccionais).
    """
    tk = [t for t in snap.index if t in price.index and np.isfinite(price.get(t, np.nan))]
    snap = snap.loc[tk]
    px = price.loc[tk].astype(float)
    eps = (snap["net_income_ttm"] / snap["shares"]).astype(float)
    bvps = (snap["equity"] / snap["shares"]).astype(float)
    pe = px / eps.replace(0, np.nan)
    pb = px / bvps.replace(0, np.nan)
    # P/E negativo (prejuizo) = nao "barato"; trata como pessimo (NaN -> excluido do top).
    pe = pe.where(pe > 0, np.nan)
    pb = pb.where(pb > 0, np.nan)
    roe = (snap["net_income_ttm"] / snap["equity"].replace(0, np.nan)).astype(float)
    net_margin = (snap["net_income_ttm"] / snap["revenue_ttm"].replace(0, np.nan)).astype(float)
    de = (snap["total_debt"] / snap["equity"].replace(0, np.nan)).astype(float)

    value = pd.concat([_zscore(-pe), _zscore(-pb)], axis=1).mean(axis=1)
    quality = pd.concat([_zscore(roe), _zscore(net_margin), _zscore(-de)], axis=1).mean(axis=1)
    combo = pd.concat([_zscore(value), _zscore(quality)], axis=1).mean(axis=1)
    return pd.DataFrame({"value": value, "quality": quality, "combo": combo})


# ============================================================================
# SIMULACAO
# ============================================================================
def _quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    q = index.to_period("Q")
    last = s.groupby(q).last()
    return pd.DatetimeIndex(last.to_numpy())


def _build_weights(
    fund: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    factor: str,
    variant: str,
    cut: str,
) -> pd.DataFrame:
    """Pesos alvo (rebal x tickers). long-only: top EW. long-short: top - bottom, $-neutral."""
    cols = [c for c in prices.columns if c != SPY]
    W = pd.DataFrame(0.0, index=rebal, columns=cols)
    frac = 1.0 / 3.0 if cut == "tercil" else 0.25
    for t in rebal:
        loc = prices.index.get_indexer([t], method="ffill")[0]
        if loc < 0:
            continue
        px = prices.iloc[loc]
        snap = _pit_snapshot(fund, t)
        if snap.empty or len(snap) < 6:
            continue
        scores = _factor_scores(snap, px)[factor].dropna()
        if len(scores) < 6:
            continue
        scores = scores.sort_values(ascending=False)
        n = len(scores)
        k = max(1, int(round(n * frac)))
        top = scores.index[:k]
        bottom = scores.index[-k:]
        if variant == "long_only":
            W.loc[t, top] = 1.0 / len(top)
        else:  # long_short, dollar-neutral
            W.loc[t, top] = 0.5 / len(top)
            W.loc[t, bottom] = -0.5 / len(bottom)
    return W


def _simulate(
    weights_rebal: pd.DataFrame, prices: pd.DataFrame, cost_per_side_bps: float
) -> pd.Series:
    """Retornos diarios liquidos. Pesos em t valem a partir de t+1 (shift). Custo no turnover."""
    cols = list(weights_rebal.columns)
    rets = prices[cols].pct_change().fillna(0.0)
    # expande pesos de rebalance p/ diario (ffill ate proximo rebalance), aplica em t+1
    w_daily = weights_rebal.reindex(prices.index).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)
    gross = (w_eff * rets).sum(axis=1)
    # custo: |Delta peso| nas datas de rebalance efetivas (quando w_daily muda)
    turnover = w_daily.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_per_side_bps / 1e4)
    cost = cost.shift(1).fillna(0.0)  # custo incorre quando o trade entra em vigor
    return (gross - cost).rename("ret")


def _equal_weight_bench(prices: pd.DataFrame) -> pd.Series:
    cols = [c for c in prices.columns if c != SPY]
    rets = prices[cols].pct_change().fillna(0.0)
    return rets.mean(axis=1).rename("ew")


def _ann_sharpe(r: np.ndarray) -> float:
    return observed_sharpe(r) * np.sqrt(TRADING_DAYS)


def _cagr(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    yrs = max(len(r) / TRADING_DAYS, 1e-9)
    return float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0


@dataclass
class ConfigResult:
    name: str
    factor: str
    variant: str
    cut: str
    ret: pd.Series


def run() -> dict:
    force = "--force" in sys.argv
    prices = _fetch_prices(force)
    if prices.empty or SPY not in prices.columns:
        return {"data_status": "blocked", "reason": "sem precos / sem SPY"}
    fund = _fetch_fundamentals(force)
    logger.info("fundamentos obtidos p/ %d tickers", len(fund))
    if len(fund) < 15:
        return {"data_status": "blocked", "reason": f"so {len(fund)} tickers com fundamentos"}

    # janela util: a partir da 1a data fiscal+lag disponivel em >=15 nomes
    first_avail = []
    for df in fund.values():
        if not df.empty:
            first_avail.append(df.index.min() + pd.Timedelta(days=REPORT_LAG_DAYS))
    first_avail.sort()
    start = first_avail[14] if len(first_avail) > 14 else first_avail[0]
    prices = prices.loc[prices.index >= start]
    if len(prices) < 252:
        # ainda roda, mas reportamos limited
        pass

    rebal = _quarter_ends(prices.index)
    rebal = rebal[rebal >= start]
    logger.info("janela: %s -> %s | %d rebalances trimestrais | %d dias",
                prices.index[0].date(), prices.index[-1].date(), len(rebal), len(prices))

    factors = ["value", "quality", "combo"]
    variants = ["long_only", "long_short"]
    cuts = ["tercil", "quartil"]
    n_trials = len(factors) * len(variants) * len(cuts)  # 12, honesto

    base = EQUITY_BASE.per_side_bps
    stress = EQUITY_BASE.stressed(2.0).per_side_bps

    configs_stress: list[ConfigResult] = []
    configs_base: list[ConfigResult] = []
    for f in factors:
        for v in variants:
            for c in cuts:
                W = _build_weights(fund, prices, rebal, f, v, c)
                if (W.abs().sum(axis=1) == 0).all():
                    continue
                r_s = _simulate(W, prices, stress)
                r_b = _simulate(W, prices, base)
                nm = f"{f}_{v}_{c}"
                configs_stress.append(ConfigResult(nm, f, v, c, r_s))
                configs_base.append(ConfigResult(nm, f, v, c, r_b))

    if not configs_stress:
        return {"data_status": "limited", "reason": "nenhuma config gerou pesos (cross-section insuficiente)"}

    # benchmarks
    ew = _equal_weight_bench(prices)
    spy_ret = prices[SPY].pct_change().fillna(0.0)
    common = configs_stress[0].ret.index
    ew = ew.reindex(common).fillna(0.0)
    spy_ret = spy_ret.reindex(common).fillna(0.0)

    # matriz p/ PBO (estressado)
    mat = np.column_stack([cr.ret.reindex(common).fillna(0.0).to_numpy() for cr in configs_stress])
    pbo = probability_of_backtest_overfitting(mat, n_splits=10)

    # avalia cada config (estressado) e escolhe a "melhor" por Sharpe estressado
    rows = []
    for cs, cb in zip(configs_stress, configs_base):
        r_s = cs.ret.reindex(common).fillna(0.0).to_numpy()
        verdict = evaluate_edge(
            r_s, n_trials=n_trials, periods_per_year=TRADING_DAYS,
            min_sharpe_annual=0.8, dsr_threshold=0.95,
        )
        corr_spy = float(np.corrcoef(r_s, spy_ret.to_numpy())[0, 1]) if r_s.std() > 0 else 0.0
        corr_ew = float(np.corrcoef(r_s, ew.to_numpy())[0, 1]) if r_s.std() > 0 else 0.0
        eq = (1 + cs.ret.reindex(common).fillna(0.0)).cumprod().to_numpy()
        rows.append({
            "name": cs.name, "factor": cs.factor, "variant": cs.variant, "cut": cs.cut,
            "sharpe_stress": verdict.sharpe_annual,
            "sharpe_base": _ann_sharpe(cb.ret.reindex(common).fillna(0.0).to_numpy()),
            "dsr": verdict.dsr, "psr": verdict.psr,
            "cagr": _cagr(cs.ret.reindex(common).fillna(0.0)),
            "maxdd": max_drawdown(eq),
            "corr_spy": corr_spy, "corr_ew": corr_ew,
            "n_obs": verdict.n_obs,
        })
    res = pd.DataFrame(rows).sort_values("sharpe_stress", ascending=False)

    # benchmark metrics
    ew_sharpe = _ann_sharpe(ew.to_numpy())
    spy_sharpe = _ann_sharpe(spy_ret.to_numpy())
    ew_cagr = _cagr(ew)
    spy_cagr = _cagr(spy_ret)

    best = res.iloc[0]
    # criterio de PASSA
    beats_bench = (best["sharpe_stress"] > max(ew_sharpe, spy_sharpe)) and (best["cagr"] > max(ew_cagr, spy_cagr))
    diversifies = (abs(best["corr_spy"]) < 0.5) and (best["sharpe_stress"] > 0.3)
    passed = bool(
        (best["dsr"] >= 0.95)
        and (best["sharpe_stress"] >= 0.8)
        and (beats_bench or diversifies)
        and (pbo < 0.5)
        and (best["n_obs"] >= 252)  # >= ~1 ano de dados util
    )

    n_years = len(common) / TRADING_DAYS
    data_status = "real" if n_years >= 3.0 else "limited"

    return {
        "data_status": data_status,
        "n_tickers_fund": len(fund),
        "n_years": round(n_years, 2),
        "n_obs": int(len(common)),
        "n_rebalances": int(len(rebal)),
        "n_trials": n_trials,
        "pbo": round(float(pbo), 3),
        "ew_sharpe": round(ew_sharpe, 3), "spy_sharpe": round(spy_sharpe, 3),
        "ew_cagr": round(ew_cagr, 4), "spy_cagr": round(spy_cagr, 4),
        "best": best.to_dict(),
        "table": res,
        "passed": passed,
        "window": (str(common[0].date()), str(common[-1].date())),
    }


def _write_report(r: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("TRIBUNAL — VALUE + QUALITY (fatores fundamentalistas)  [R&D isolado]")
    lines.append("=" * 78)
    lines.append(f"data_status: {r.get('data_status')}")
    if "reason" in r:
        lines.append(f"motivo: {r['reason']}")
    if "best" in r:
        lines.append(f"janela: {r['window'][0]} -> {r['window'][1]}  ({r['n_years']} anos, {r['n_obs']} dias)")
        lines.append(f"tickers c/ fundamentos: {r['n_tickers_fund']} | rebalances trimestrais: {r['n_rebalances']}")
        lines.append(f"n_trials (honesto): {r['n_trials']} | PBO: {r['pbo']}")
        lines.append("")
        lines.append(f"Benchmarks: EW Sharpe={r['ew_sharpe']} CAGR={r['ew_cagr']:.2%} | "
                     f"SPY Sharpe={r['spy_sharpe']} CAGR={r['spy_cagr']:.2%}")
        lines.append("")
        lines.append("Configs (custo ESTRESSADO 2x, ordenado por Sharpe):")
        tbl = r["table"][["name", "sharpe_stress", "sharpe_base", "dsr", "cagr", "maxdd", "corr_spy", "n_obs"]]
        lines.append(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        lines.append("")
        b = r["best"]
        lines.append(f"MELHOR config: {b['name']}")
        lines.append(f"  Sharpe(stress)={b['sharpe_stress']:.3f}  DSR={b['dsr']:.3f}  "
                     f"CAGR={b['cagr']:.2%}  MaxDD={b['maxdd']:.2%}  corr_SPY={b['corr_spy']:.3f}")
        lines.append("")
        lines.append(f"VEREDITO: {'PASSA' if r['passed'] else 'FALHA'}")
        lines.append("Barra: DSR>=0.95 E Sharpe_stress>=0.8 E (bate EW/SPY OU diversifica corr<0.5) E PBO<0.5 E >=1ano")
    txt = "\n".join(lines)
    REPORT_PATH.write_text(txt)
    logger.info("\n%s", txt)
    logger.info("Relatorio salvo em %s", REPORT_PATH)


if __name__ == "__main__":
    result = run()
    _write_report(result)
