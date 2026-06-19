"""Tribunal de DUAL MOMENTUM / TAA (Faber GTAA + Antonacci dual momentum).

R&D PURO e ISOLADO. NAO toca em beta_*, main.py, nav_history nem config de
producao. Cria seu proprio cache (data/taa_cache) e relatorio (data/taa_dualmom_verdict.txt).
IMPORTA (read-only) o tribunal anti-overfitting do projeto:
  - simulation.statistics : evaluate_edge / DSR / PSR / PBO / observed_sharpe
  - simulation.metrics    : max_drawdown
  - simulation.costs      : EQUITY_BASE (custo real de equity, por lado) + stressed 2x

PERGUNTA (imparcial, MEDIR — nao assumir):
  Um TAA de momentum ABSOLUTO (so fica no ativo de risco se o seu retorno de
  lookback > cash/T-bills, senao vai p/ bonds/cash) combinado com momentum
  RELATIVO (escolhe os top-N ativos de risco por momentum) REDUZ o drawdown e
  melhora o risco-ajustado de um portfolio 60/40 buy&hold, de forma ROBUSTA OOS?

ESTRATEGIA (rebalance MENSAL, no fechamento do ultimo pregao do mes):
  - Universo de RISCO: ETFs de classes de ativo (acoes US/intl/EM, REITs,
    commodities, ouro, bond longo).
  - Sinal: momentum absoluto = retorno total de L meses de cada ativo.
    * RELATIVO: ranqueia o universo de risco por momentum, escolhe top-N.
    * ABSOLUTO: dos top-N, so mantem os que tem momentum > momentum do CASH
      (BIL/T-bill proxy). Os reprovados viram alocacao DEFENSIVA (IEF/AGG).
  - Pesos equal-weight entre os slots; cada slot reprovado no filtro absoluto
    vai p/ o ativo defensivo. Sem alavancagem, long-only (implementavel).

SEM LOOK-AHEAD (rigoroso):
  - O sinal do mes m usa SO closes ate o ultimo pregao de m (t-1 relativo ao
    primeiro retorno de m+1). Pesos do mes m+1 aplicados aos retornos DIARIOS de
    m+1. Garantido por weights.shift(1) na simulacao diaria.

CUSTO: simulation.costs.EQUITY_BASE (3 bps/lado) cobrado sobre |Delta peso| de
cada rebalance (turnover real). VEREDITO usa o cenario ESTRESSADO 2x.

n_TRIALS HONESTO (anti data-snooping): conta TODAS as variantes varridas:
  lookbacks {3,6,9,10,12} x top_N {1,2,3,4} x defensivo {IEF,AGG} x
  universos {core5, broad8} = 5*4*2*2 = 80 trials. Contado e passado ao DSR;
  PBO sobre a matriz de retornos mensais de TODAS as 80 configs.

BARRA (regua do programa): PASSA so se, no cenario ESTRESSADO:
  DSR >= 0.95  E  Sharpe liq robusto  E
  (bate 60/40 buy&hold  OU  diversifica: corr baixa com SPY melhorando o
   conjunto via MaxDD muito menor + Sharpe>=60/40), ROBUSTO OOS (PBO baixo).
  Senao: FALHA -> cemiterio. Imparcial.

Uso:
    uv run python -m simulation.dual_momentum            # usa cache
    uv run python -m simulation.dual_momentum --force    # re-baixa precos
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/taa_cache")
BETA_CACHE = Path("data/beta_cache")  # reuso read-only do cache existente
REPORT = Path("data/taa_dualmom_verdict.txt")

# Universo de classes de ativo (ETFs liquidos, dados gratis yfinance).
# core5 = Antonacci-ish (US/intl/REIT/commod/bond); broad8 acrescenta EM/ouro/US-tech.
RISK_CORE = ["SPY", "EFA", "VNQ", "DBC", "TLT"]
RISK_BROAD = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "DBC", "GLD", "TLT"]
DEFENSIVE = ["IEF", "AGG"]  # ativos defensivos testados
CASH = "BIL"  # proxy de T-bill p/ o filtro de momentum absoluto
SPY = "SPY"  # benchmark de mercado p/ correlacao

ALL_TICKERS = sorted(set(RISK_CORE + RISK_BROAD + DEFENSIVE + [CASH, SPY, "IEF"]))

TRADING_DAYS = EQUITY_PERIODS


# ============================================================================
# DADOS — yfinance, cache proprio em data/taa_cache (reusa data/beta_cache se houver)
# ============================================================================
def _yf_download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        ticker, period="max", interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        try:
            out = out.xs(ticker, axis=1, level=1)
        except KeyError:
            out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower() for c in out.columns]
    col = "close" if "close" in out.columns else (
        "adj close" if "adj close" in out.columns else None
    )
    if col is None:
        return pd.DataFrame()
    s = pd.to_numeric(out[col], errors="coerce")
    idx = pd.to_datetime(s.index, utc=True, errors="coerce")
    res = pd.DataFrame({"close": s.to_numpy()}, index=idx)
    res = res[~res.index.isna()].dropna()
    return res[res["close"] > 0].sort_index()


def load_close(ticker: str, *, force: bool = False) -> pd.DataFrame:
    """Close ajustado de UM ticker. Cache proprio; reusa beta_cache read-only se houver."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{ticker}_1d.csv"
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]
    # reuso do cache do beta (somente leitura) p/ tickers ja baixados
    bp = BETA_CACHE / f"{ticker}_1d.csv"
    if bp.exists() and not force:
        df = pd.read_csv(bp, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        if "close" in df.columns and not df.empty:
            df[["close"]].to_csv(p)
            return df[["close"]]
    df = _normalize(_yf_download(ticker), ticker)
    if not df.empty:
        df.to_csv(p)
        logger.info("%s: %d barras (%s..%s)", ticker, len(df),
                    df.index[0].date(), df.index[-1].date())
    return df


def load_panel(force: bool = False) -> pd.DataFrame:
    cols = {}
    for t in ALL_TICKERS:
        df = load_close(t, force=force)
        if df.empty:
            logger.warning("%s sem dados — pulado.", t)
            continue
        s = df["close"]
        cols[t] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()
    return panel


# ============================================================================
# SINAL E SIMULACAO — sem look-ahead
# ============================================================================
def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ultimo pregao de cada mes (datas de decisao de rebalance)."""
    s = pd.Series(idx, index=idx)
    grp = s.groupby([idx.year, idx.month]).last()
    return pd.DatetimeIndex(grp.values, tz=idx.tz)


def _momentum(panel: pd.DataFrame, asof: pd.Timestamp, lookback_m: int) -> pd.Series:
    """Retorno total de `lookback_m` meses ate `asof` (inclusive). Sem look-ahead:
    so usa closes <= asof. ~21 pregoes/mes."""
    win = lookback_m * 21
    sub = panel.loc[:asof]
    if len(sub) < win + 1:
        return pd.Series(dtype=float)
    end = sub.iloc[-1]
    start = sub.iloc[-(win + 1)]
    mom = end / start - 1.0
    return mom


def build_weights(
    panel: pd.DataFrame,
    *,
    risk_universe: list[str],
    lookback_m: int,
    top_n: int,
    defensive: str,
) -> pd.DataFrame:
    """Matriz de pesos DIARIA (linha t = peso aplicado ao retorno de t).

    Decisao no ultimo pregao de cada mes; pesos vigem no mes seguinte. Ao final
    aplicamos shift(1) p/ garantir que peso da linha t usa so dados ate t-1.
    """
    cols = list(panel.columns)
    me = _month_end_index(panel.index)
    w_monthly = pd.DataFrame(0.0, index=me, columns=cols)

    avail = [t for t in risk_universe if t in panel.columns]
    for dt in me:
        mom = _momentum(panel, dt, lookback_m)
        if mom.empty:
            continue
        # so ativos de risco com momentum valido
        rm = mom.reindex(avail).dropna()
        if rm.empty or pd.isna(mom.get(CASH, np.nan)):
            continue
        cash_mom = mom[CASH]
        # RELATIVO: top-N por momentum
        chosen = rm.sort_values(ascending=False).head(top_n)
        slot_w = 1.0 / top_n
        for tkr, m in chosen.items():
            # ABSOLUTO: so fica no ativo se momentum > cash; senao -> defensivo
            if m > cash_mom:
                w_monthly.loc[dt, tkr] += slot_w
            else:
                w_monthly.loc[dt, defensive] += slot_w
    return w_monthly


def _daily_weights(panel: pd.DataFrame, w_monthly: pd.DataFrame) -> pd.DataFrame:
    """Propaga pesos mensais (decididos no ME) p/ os dias do mes SEGUINTE, com
    shift(1). w_monthly index = ultimo pregao do mes m -> vige a partir do
    primeiro pregao de m+1."""
    full = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    full.loc[w_monthly.index] = w_monthly.values  # peso na data de decisao (ME)
    full = full.ffill().fillna(0.0)               # vige ate o proximo rebalance
    # shift(1): peso decidido no fechamento de t so atua a partir de t+1
    return full.shift(1).fillna(0.0)


def simulate(
    panel: pd.DataFrame,
    daily_w: pd.DataFrame,
    cost,
    *,
    start: pd.Timestamp | None = None,
) -> pd.Series:
    """Retornos DIARIOS liquidos da estrategia. Custo sobre |Delta peso|."""
    rets = panel.pct_change().fillna(0.0)
    if start is not None:
        rets = rets.loc[start:]
        daily_w = daily_w.loc[start:]
    daily_w = daily_w.reindex(rets.index).fillna(0.0)
    gross = (daily_w * rets).sum(axis=1)
    # turnover: mudanca de peso vs dia anterior
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    cost_frac = cost.per_side_bps / 10_000.0
    net = gross - turnover * cost_frac
    return net


def benchmark_6040(panel: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.Series:
    """60/40 SPY/IEF, rebalance mensal, liquido do mesmo custo equity."""
    cols = ["SPY", "IEF"]
    sub = panel[cols].dropna()
    me = _month_end_index(sub.index)
    w = pd.DataFrame(np.nan, index=sub.index, columns=cols)
    target = pd.Series({"SPY": 0.6, "IEF": 0.4})
    for dt in me:
        w.loc[dt] = target.values
    w = w.ffill().fillna(0.0).shift(1).fillna(0.0)
    rets = sub.pct_change().fillna(0.0)
    if start is not None:
        rets = rets.loc[start:]
        w = w.loc[start:]
    w = w.reindex(rets.index).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turn * (EQUITY_BASE.per_side_bps / 10_000.0)
    return net


def to_monthly(daily_returns: pd.Series) -> pd.Series:
    """Compoe retornos diarios em mensais (p/ DSR/PBO em escala mensal)."""
    return (1.0 + daily_returns).resample("ME").prod() - 1.0


def equity_curve(returns: pd.Series) -> np.ndarray:
    return (1.0 + returns).cumprod().to_numpy()


@dataclass
class ConfigResult:
    name: str
    lookback_m: int
    top_n: int
    defensive: str
    universe: str
    monthly_returns: pd.Series  # liquido (estressado)
    sharpe_ann: float
    cagr: float
    maxdd: float


def _cagr(returns: pd.Series, periods: int) -> float:
    eq = (1.0 + returns).prod()
    yrs = len(returns) / periods
    if yrs <= 0 or eq <= 0:
        return -1.0
    return float(eq ** (1.0 / yrs) - 1.0)


# ============================================================================
# TRIBUNAL
# ============================================================================
def run(force: bool = False) -> dict:
    panel = load_panel(force=force)
    if panel.empty:
        return {"data_status": "blocked", "reason": "painel vazio"}

    # Janela comum: a partir do primeiro dia em que todo o universo BROAD + cash
    # + defensivos estao vivos (senao o filtro absoluto/relativo nao e honesto).
    need = sorted(set(RISK_BROAD + DEFENSIVE + [CASH, SPY]))
    alive = panel[need].notna().all(axis=1)
    if not alive.any():
        return {"data_status": "limited", "reason": "universo broad nunca todo vivo"}
    common_start = panel.index[alive][0]
    logger.info("janela comum (broad): %s .. %s", common_start.date(),
                panel.index[-1].date())

    cost_base = EQUITY_BASE
    cost_stress = EQUITY_BASE.stressed(2.0)

    lookbacks = [3, 6, 9, 10, 12]
    top_ns = [1, 2, 3, 4]
    universes = {"core5": RISK_CORE, "broad8": RISK_BROAD}

    n_trials = len(lookbacks) * len(top_ns) * len(DEFENSIVE) * len(universes)

    results: list[ConfigResult] = []
    for uname, uni in universes.items():
        max_n = min(len(uni), max(top_ns))
        for lb in lookbacks:
            for tn in top_ns:
                if tn > len(uni):
                    continue
                for dfn in DEFENSIVE:
                    w_m = build_weights(
                        panel, risk_universe=uni, lookback_m=lb,
                        top_n=tn, defensive=dfn,
                    )
                    dw = _daily_weights(panel, w_m)
                    net_d = simulate(panel, dw, cost_stress, start=common_start)
                    m = to_monthly(net_d).dropna()
                    if len(m) < 24:
                        continue
                    results.append(ConfigResult(
                        name=f"{uname}_L{lb}_N{tn}_{dfn}",
                        lookback_m=lb, top_n=tn, defensive=dfn, universe=uname,
                        monthly_returns=m,
                        sharpe_ann=observed_sharpe(m.to_numpy()) * np.sqrt(12),
                        cagr=_cagr(m, 12),
                        maxdd=max_drawdown(equity_curve(m)),
                    ))

    if not results:
        return {"data_status": "limited", "reason": "nenhuma config valida"}

    # Matriz de retornos mensais alinhada (p/ PBO). Index comum.
    idx = results[0].monthly_returns.index
    for r in results:
        idx = idx.intersection(r.monthly_returns.index)
    mat = np.column_stack([r.monthly_returns.reindex(idx).to_numpy() for r in results])

    pbo = probability_of_backtest_overfitting(mat, n_splits=16)

    # Sharpes por periodo de todas as trials (p/ variancia do DSR).
    trial_sharpes = [observed_sharpe(r.monthly_returns.to_numpy()) for r in results]

    # MELHOR config in-sample por Sharpe (a que o data-snooping escolheria).
    best = max(results, key=lambda r: r.sharpe_ann)

    # benchmark 60/40 na mesma janela
    bench_d = benchmark_6040(panel, start=common_start)
    bench_m = to_monthly(bench_d).reindex(best.monthly_returns.index).dropna()
    # alinha best e bench
    cidx = best.monthly_returns.index.intersection(bench_m.index)
    best_m = best.monthly_returns.reindex(cidx)
    bench_m = bench_m.reindex(cidx)

    # SPY mensal p/ correlacao ao mercado
    spy_d = panel["SPY"].pct_change().loc[common_start:]
    spy_m = to_monthly(spy_d.fillna(0.0)).reindex(cidx).dropna()
    ci2 = cidx.intersection(spy_m.index)
    corr_spy = float(np.corrcoef(
        best_m.reindex(ci2).to_numpy(), spy_m.reindex(ci2).to_numpy()
    )[0, 1])

    bench_sharpe = observed_sharpe(bench_m.to_numpy()) * np.sqrt(12)
    bench_cagr = _cagr(bench_m, 12)
    bench_maxdd = max_drawdown(equity_curve(bench_m))

    # DSR/PSR do melhor, custo ESTRESSADO, n_trials honesto, escala mensal (252->12).
    verdict = evaluate_edge(
        best.monthly_returns.to_numpy(),
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        periods_per_year=12,
        min_sharpe_annual=0.7,
        dsr_threshold=0.95,
    )

    # BARRA: DSR>=0.95 E sharpe robusto E (bate 60/40 OU diversifica c/ MaxDD muito
    # menor + corr baixa + sharpe>=bench).
    beats_6040 = (best.sharpe_ann > bench_sharpe) and (best.cagr >= bench_cagr)
    diversifies = (
        abs(corr_spy) <= 0.7
        and best.maxdd > bench_maxdd  # menos negativo = melhor (DD menor)
        and best.sharpe_ann >= bench_sharpe
    )
    passed = bool(
        verdict.passes_dsr
        and verdict.passes_sharpe
        and (beats_6040 or diversifies)
        and pbo < 0.5
    )

    out = {
        "data_status": "real",
        "n_obs_months": int(len(best.monthly_returns)),
        "window": f"{common_start.date()}..{panel.index[-1].date()}",
        "n_trials": n_trials,
        "best_config": best.name,
        "sharpe_stress": round(best.sharpe_ann, 3),
        "cagr_stress": round(best.cagr, 4),
        "maxdd_stress": round(best.maxdd, 4),
        "dsr": round(verdict.dsr, 4),
        "psr": round(verdict.psr, 4),
        "pbo": round(pbo, 4),
        "corr_spy": round(corr_spy, 4),
        "bench_6040_sharpe": round(bench_sharpe, 3),
        "bench_6040_cagr": round(bench_cagr, 4),
        "bench_6040_maxdd": round(bench_maxdd, 4),
        "beats_6040": beats_6040,
        "diversifies": diversifies,
        "passes_dsr": verdict.passes_dsr,
        "passes_sharpe": verdict.passes_sharpe,
        "passed": passed,
        "verdict_summary": verdict.summary(),
        "all_results": results,
    }

    _write_report(out)
    return out


def _write_report(out: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("VEREDITO — DUAL MOMENTUM / TAA (absolute + relative, Faber/GTAA)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"DATA STATUS: {out['data_status']}")
    lines.append(f"Janela (comum, broad vivo): {out['window']}  "
                 f"({out['n_obs_months']} meses)")
    lines.append(f"n_trials (honesto): {out['n_trials']}  "
                 "(lookbacks 5 x top_N 4 x defensivo 2 x universos 2)")
    lines.append("")
    lines.append("--- TODAS AS CONFIGS (Sharpe ann, CAGR, MaxDD; custo ESTRESSADO 2x) ---")
    for r in sorted(out["all_results"], key=lambda x: -x.sharpe_ann):
        lines.append(
            f"  {r.name:<22} Sharpe={r.sharpe_ann:6.2f}  "
            f"CAGR={r.cagr*100:6.2f}%  MaxDD={r.maxdd*100:7.2f}%"
        )
    lines.append("")
    lines.append("--- MELHOR CONFIG IN-SAMPLE (a que o data-snooping escolheria) ---")
    lines.append(f"  config        : {out['best_config']}")
    lines.append(f"  Sharpe (2x)   : {out['sharpe_stress']}")
    lines.append(f"  CAGR (2x)     : {out['cagr_stress']*100:.2f}%")
    lines.append(f"  MaxDD (2x)    : {out['maxdd_stress']*100:.2f}%")
    lines.append("")
    lines.append("--- BENCHMARK 60/40 (SPY/IEF, mesma janela, mesmo custo) ---")
    lines.append(f"  Sharpe        : {out['bench_6040_sharpe']}")
    lines.append(f"  CAGR          : {out['bench_6040_cagr']*100:.2f}%")
    lines.append(f"  MaxDD         : {out['bench_6040_maxdd']*100:.2f}%")
    lines.append("")
    lines.append("--- TRIBUNAL ESTATISTICO (anti-overfitting) ---")
    lines.append(f"  DSR           : {out['dsr']}   (barra >= 0.95)")
    lines.append(f"  PSR           : {out['psr']}")
    lines.append(f"  PBO (CSCV)    : {out['pbo']}   (barra < 0.5)")
    lines.append(f"  corr c/ SPY   : {out['corr_spy']}")
    lines.append(f"  {out['verdict_summary']}")
    lines.append("")
    lines.append("--- DECISAO ---")
    lines.append(f"  bate 60/40?      {out['beats_6040']}")
    lines.append(f"  diversifica?     {out['diversifies']} "
                 "(|corr|<=0.7 E MaxDD melhor E Sharpe>=60/40)")
    lines.append(f"  passes_dsr?      {out['passes_dsr']}")
    lines.append(f"  passes_sharpe?   {out['passes_sharpe']}")
    lines.append("")
    lines.append(f"  >>> PASSA NA BARRA: {out['passed']}")
    lines.append("=" * 78)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res = run(force=args.force)
    if res.get("data_status") != "real":
        print(f"DATA STATUS: {res.get('data_status')} — {res.get('reason')}")
    else:
        print(f"\nRELATORIO: {REPORT}")
        print(f"PASSA: {res['passed']}  DSR={res['dsr']}  "
              f"Sharpe={res['sharpe_stress']}  MaxDD={res['maxdd_stress']*100:.1f}%  "
              f"corr_SPY={res['corr_spy']}")
