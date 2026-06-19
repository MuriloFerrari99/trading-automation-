"""TRIBUNAL: Residual (idiosyncratic) momentum em equities US.

TESE
----
Momentum cross-seccional sobre RESIDUOS de um modelo de fatores (CAPM ou
Fama-French 3F), em vez de retornos brutos. Ranqueia por momentum do residuo
(formacao 12-1m: pula o mes mais recente p/ evitar reversao de curto prazo),
long o top, short o bottom, market-neutral (dollar-neutral). A hipotese
(Blitz, Huij, Martens 2011) e que residual momentum tem Sharpe maior e e mais
ortogonal ao mercado que o momentum bruto (que ja reprovou neste projeto).

CONSTRUCAO (sem look-ahead)
---------------------------
- Frequencia: mensal (rebalance no ultimo dia util do mes). Retornos diarios
  agregados a mensais (compostos).
- Para cada mes-de-formacao t e cada ativo:
    * janela de estimacao do beta: 36 meses ate t (rolling), residuos = ret -
      alpha - beta*fatores.
    * score = soma dos residuos mensais da janela 12-1 (meses t-11..t-1),
      i.e. PULA o mes t (gap de 1 mes), padronizado pela vol dos residuos
      (Blitz et al.). Tudo usa dados <= t.
- Carteira: long top tercil/quintil, short bottom; pesos iguais; dollar-neutral.
- Retorno realizado no mes t+1 (sem look-ahead). Custo aplicado sobre o turnover
  de cada perna por rebalance.

BENCHMARKS DE GRADUACAO (barra do projeto)
------------------------------------------
PASSA so se: DSR>=0.95 E Sharpe_liq robusto E (bate buy&hold SPY OU diversifica
com corr BAIXA ao SPY melhorando o conjunto) E robusto OOS (PBO baixo, split).

n_trials HONESTO: contamos TODAS as variantes do grid (modelo de fator x
janela de formacao x corte top/bottom x janela de beta) avaliadas. O DSR usa a
variancia dos Sharpes dessas trials (anti-snooping de verdade).

Custo: EQUITY_BASE (3 bps/lado) e cenario ESTRESSADO 2x. O veredito usa o
cenario estressado para ser honesto.

Uso:
    uv run python -m simulation.residual_momentum_tribunal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data.residmom_data import load_closes, load_ff_factors, load_market
from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.residual_momentum_tribunal")

REPORT_PATH = Path("data/residual_momentum_verdict.txt")
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Preparacao de dados                                                         #
# --------------------------------------------------------------------------- #
def _to_monthly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Closes diarios ajustados -> retornos mensais simples (ultimo do mes)."""
    monthly_px = closes.resample("ME").last()
    return monthly_px.pct_change()


def _market_monthly(spy: pd.Series) -> pd.Series:
    px = spy.resample("ME").last()
    return px.pct_change()


def _ff_monthly(ff_daily: pd.DataFrame) -> pd.DataFrame:
    """Fatores diarios (fracao) -> mensais por composicao (1+r).prod()-1."""
    out = (1.0 + ff_daily).resample("ME").prod() - 1.0
    return out


# --------------------------------------------------------------------------- #
# Modelo de fator + residuos                                                  #
# --------------------------------------------------------------------------- #
def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residuos de OLS y ~ [1, X] via lstsq. X shape (n, k). Retorna (n,)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _residual_panel(
    rets: pd.DataFrame,
    factors: pd.DataFrame,
    rf: pd.Series,
    beta_window: int,
) -> pd.DataFrame:
    """Para cada mes t e ativo, residuo do excesso de retorno ~ fatores,
    estimado em janela rolling [t-beta_window+1 .. t] (inclui t, in-sample da
    janela de formacao — sem look-ahead pois usa so dados <= t).

    Retorna painel de residuos mensais alinhado a `rets` (NaN onde insuficiente).
    """
    idx = rets.index
    cols = rets.columns
    F = factors.reindex(idx).values  # (T, k)
    rf_v = rf.reindex(idx).values
    resid = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for j, c in enumerate(cols):
        y_full = rets[c].values - rf_v  # excesso
        for end in range(beta_window - 1, len(idx)):
            sl = slice(end - beta_window + 1, end + 1)
            y = y_full[sl]
            Xw = F[sl]
            mask = np.isfinite(y) & np.isfinite(Xw).all(axis=1)
            if mask.sum() < beta_window * 0.8:
                continue
            res = _ols_residuals(y[mask], Xw[mask])
            # o residuo do mes corrente t e o ultimo da janela
            if mask[-1]:
                resid.iat[end, j] = res[-1]
    return resid


# --------------------------------------------------------------------------- #
# Backtest de uma configuracao                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    model: str          # "capm" | "ff3"
    beta_window: int    # meses p/ estimar beta (rolling)
    form_lookback: int  # meses de formacao (12)
    gap: int            # meses pulados no fim (1)
    frac: float         # fracao long/short (0.2=quintil, 0.333=tercil)

    @property
    def label(self) -> str:
        return (f"{self.model}|bw{self.beta_window}|"
                f"L{self.form_lookback}g{self.gap}|f{self.frac:g}")


def _signal_from_residuals(
    resid: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Score de residual momentum por mes t: soma padronizada dos residuos da
    janela de formacao [t-L .. t-gap], usando SO dados <= t (no look-ahead)."""
    L, g = cfg.form_lookback, cfg.gap
    # rolling sum dos residuos sobre L meses, depois desloca g p/ pular o gap
    rsum = resid.rolling(L, min_periods=int(L * 0.8)).sum().shift(g)
    rstd = resid.rolling(L, min_periods=int(L * 0.8)).std().shift(g)
    score = rsum / rstd.replace(0.0, np.nan)
    return score


def _portfolio_returns(
    score: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    cfg: Config,
    cost_bps_per_side: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retornos mensais liquidos da carteira long-short market-neutral.

    score em t -> pesos em t -> retorno realizado fwd_rets em t+1.
    Retorna (retornos_mensais, weights_history_turnover, exposure).
    """
    dates = score.index
    rets_out: list[float] = []
    prev_w = pd.Series(0.0, index=score.columns)
    months_in_market = 0

    for i in range(len(dates) - 1):
        s = score.iloc[i].dropna()
        # exige universo minimo p/ formar pernas
        if len(s) < 10:
            rets_out.append(0.0)
            continue
        n_side = max(1, int(np.floor(len(s) * cfg.frac)))
        ranked = s.sort_values()
        shorts = ranked.index[:n_side]
        longs = ranked.index[-n_side:]
        w = pd.Series(0.0, index=score.columns)
        w[longs] = 0.5 / n_side       # perna long soma +0.5 (gross 1.0 total)
        w[shorts] = -0.5 / n_side     # perna short soma -0.5
        # retorno bruto no mes seguinte
        fwd = fwd_rets.iloc[i + 1].reindex(w.index)
        gross = float((w * fwd.fillna(0.0)).sum())
        # custo de turnover (soma das mudancas absolutas de peso) * custo/lado
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * (cost_bps_per_side / 1e4)
        rets_out.append(gross - cost)
        prev_w = w
        months_in_market += 1

    exposure = months_in_market / max(len(dates) - 1, 1)
    return np.asarray(rets_out, dtype=float), np.asarray([]), exposure


# --------------------------------------------------------------------------- #
# Grid + tribunal                                                             #
# --------------------------------------------------------------------------- #
def run_tribunal() -> dict:
    closes = load_closes()
    spy = load_market()
    ff = load_ff_factors()

    status = "real"
    notes: list[str] = []
    if closes.empty or spy.empty:
        return {"data_status": "blocked",
                "note": "yfinance nao retornou precos. Rode data.residmom_data --force"}

    rets_m = _to_monthly_returns(closes)
    spy_m = _market_monthly(spy)

    have_ff = not ff.empty
    if have_ff:
        ffm = _ff_monthly(ff)
        rf_m = ffm["RF"]
        mktrf_m = ffm["MktRF"]
    else:
        status = "limited"
        notes.append("FF indisponivel: so CAPM (market model com SPY).")
        rf_m = pd.Series(0.0, index=spy_m.index)
        mktrf_m = spy_m  # excesso ~ SPY (rf~0)

    # alinha tudo no mesmo calendario mensal
    common = rets_m.dropna(how="all").index
    common = common.intersection(mktrf_m.dropna().index)
    rets_m = rets_m.reindex(common)
    spy_m = spy_m.reindex(common)
    rf_m = rf_m.reindex(common).fillna(0.0)

    # matrizes de fatores por modelo
    factor_sets: dict[str, pd.DataFrame] = {}
    factor_sets["capm"] = pd.DataFrame({"mkt": mktrf_m.reindex(common)})
    if have_ff:
        factor_sets["ff3"] = pd.DataFrame({
            "mkt": ffm["MktRF"].reindex(common),
            "smb": ffm["SMB"].reindex(common),
            "hml": ffm["HML"].reindex(common),
        })

    # GRID honesto: todas as variantes que tocamos contam no n_trials
    models = ["capm"] + (["ff3"] if have_ff else [])
    beta_windows = [36, 24]
    fracs = [0.2, 0.333]
    form_lookback = 12
    gap = 1

    configs = [
        Config(model=m, beta_window=bw, form_lookback=form_lookback, gap=gap, frac=f)
        for m, bw, f in product(models, beta_windows, fracs)
    ]
    n_trials = len(configs)
    logger.info("Grid: %d configuracoes (n_trials honesto)", n_trials)

    # custo estressado (honesto p/ veredito)
    cost = EQUITY_BASE.stressed(2.0)
    cost_base = EQUITY_BASE

    # cacheia paineis de residuo por (model, beta_window) p/ nao refazer OLS
    resid_cache: dict[tuple[str, int], pd.DataFrame] = {}
    series: dict[str, np.ndarray] = {}
    series_base: dict[str, np.ndarray] = {}
    summaries: list[dict] = []

    for cfg in configs:
        key = (cfg.model, cfg.beta_window)
        if key not in resid_cache:
            resid_cache[key] = _residual_panel(
                rets_m, factor_sets[cfg.model], rf_m, cfg.beta_window)
        resid = resid_cache[key]
        score = _signal_from_residuals(resid, cfg)
        r_stress, _, expo = _portfolio_returns(
            score, rets_m, cfg, cost.per_side_bps)
        r_base, _, _ = _portfolio_returns(
            score, rets_m, cfg, cost_base.per_side_bps)
        series[cfg.label] = r_stress
        series_base[cfg.label] = r_base
        sr = observed_sharpe(r_stress[r_stress != 0.0] if (r_stress != 0).any() else r_stress)
        summaries.append({
            "label": cfg.label,
            "sharpe_ann_stress": observed_sharpe(_trim(r_stress)) * np.sqrt(MONTHS_PER_YEAR),
            "sharpe_ann_base": observed_sharpe(_trim(r_base)) * np.sqrt(MONTHS_PER_YEAR),
            "mean_m": float(np.mean(_trim(r_stress))),
            "n": int(_trim(r_stress).size),
            "exposure": expo,
        })

    # escolhe a config "primaria" = a CANONICA da literatura (ff3 se houver,
    # bw36, quintil), NAO a melhor in-sample, p/ evitar cherry-pick.
    primary_label = _canonical_label(have_ff, beta_windows[0], fracs[0],
                                     form_lookback, gap)
    if primary_label not in series:
        primary_label = summaries[0]["label"]
    r_primary = _trim(series[primary_label])
    r_primary_base = _trim(series_base[primary_label])

    # alinha retornos da carteira ao SPY no MESMO recorte temporal p/ corr/benchmark
    # r_primary cobre meses [start+? .. end-1]; recompoe datas validas
    port_dates = _port_dates(series[primary_label], rets_m.index)
    spy_aligned = spy_m.reindex(port_dates).fillna(0.0).values
    spy_aligned = spy_aligned[: len(r_primary)] if len(spy_aligned) >= len(r_primary) else spy_aligned

    # correlacao ao SPY
    corr = _safe_corr(r_primary, spy_aligned)

    # equity e drawdown
    eq = np.cumprod(1.0 + r_primary)
    eq = np.concatenate([[1.0], eq])
    mdd = max_drawdown(eq)
    cagr = float(eq[-1] ** (MONTHS_PER_YEAR / max(len(r_primary), 1)) - 1.0)

    # buy&hold SPY no mesmo recorte
    spy_eq = np.cumprod(1.0 + spy_aligned)
    spy_eq = np.concatenate([[1.0], spy_eq])
    spy_sharpe = observed_sharpe(spy_aligned) * np.sqrt(MONTHS_PER_YEAR)
    spy_cagr = float(spy_eq[-1] ** (MONTHS_PER_YEAR / max(len(spy_aligned), 1)) - 1.0)

    # tribunal: DSR/PSR com n_trials honesto e variancia dos Sharpes das trials
    trial_sharpes = [observed_sharpe(_trim(v)) for v in series.values()]
    verdict = evaluate_edge(
        r_primary,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        periods_per_year=MONTHS_PER_YEAR,
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # PBO via CSCV sobre a matriz de retornos das trials (alinhadas no menor T)
    minT = min(len(v) for v in series.values())
    M = np.column_stack([v[-minT:] for v in series.values()])
    pbo = probability_of_backtest_overfitting(M, n_splits=10)

    # OOS split: 70/30 — Sharpe na metade final (out-of-sample temporal)
    split = int(len(r_primary) * 0.7)
    sr_is = observed_sharpe(r_primary[:split]) * np.sqrt(MONTHS_PER_YEAR)
    sr_oos = observed_sharpe(r_primary[split:]) * np.sqrt(MONTHS_PER_YEAR)

    # ---- barra de graduacao ----
    sharpe_stress = verdict.sharpe_annual
    beats_bh = sharpe_stress > spy_sharpe and cagr > spy_cagr
    diversifies = abs(corr) < 0.3 and sharpe_stress >= 0.5  # corr baixa + add real
    robust_oos = pbo < 0.5 and sr_oos > 0.0
    passed = bool(
        verdict.passes_dsr
        and sharpe_stress >= 0.8
        and (beats_bh or diversifies)
        and robust_oos
    )

    result = {
        "data_status": status,
        "notes": notes,
        "n_trials": n_trials,
        "primary": primary_label,
        "n_months": int(len(r_primary)),
        "sharpe_stress": sharpe_stress,
        "sharpe_base": observed_sharpe(r_primary_base) * np.sqrt(MONTHS_PER_YEAR),
        "dsr": verdict.dsr,
        "psr": verdict.psr,
        "sr_benchmark_ann": verdict.sr_benchmark_annual,
        "pbo": pbo,
        "corr_spy": corr,
        "cagr": cagr,
        "maxdd": mdd,
        "spy_sharpe": spy_sharpe,
        "spy_cagr": spy_cagr,
        "sr_is": sr_is,
        "sr_oos": sr_oos,
        "skew": verdict.skew,
        "kurtosis": verdict.kurtosis,
        "beats_bh": beats_bh,
        "diversifies": diversifies,
        "robust_oos": robust_oos,
        "passed": passed,
        "summaries": summaries,
        "verdict_summary": verdict.summary(),
    }
    _write_report(result)
    return result


def _trim(r: np.ndarray) -> np.ndarray:
    """Remove zeros de aquecimento no inicio (meses sem sinal valido)."""
    r = np.asarray(r, dtype=float)
    nz = np.nonzero(r)[0]
    if nz.size == 0:
        return r
    return r[nz[0]:]


def _port_dates(r_full: np.ndarray, monthly_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Datas (t+1) correspondentes aos retornos realizados, apos o trim."""
    # r_full tem len = len(index)-1 ; retorno i corresponde a realizacao em index[i+1]
    nz = np.nonzero(r_full)[0]
    start = nz[0] if nz.size else 0
    return monthly_index[start + 1: start + 1 + (len(r_full) - start)]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _canonical_label(have_ff: bool, bw: int, frac: float, L: int, g: int) -> str:
    model = "ff3" if have_ff else "capm"
    return f"{model}|bw{bw}|L{L}g{g}|f{frac:g}"


def _write_report(r: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("TRIBUNAL — RESIDUAL (IDIOSYNCRATIC) MOMENTUM — EQUITIES US")
    lines.append("=" * 78)
    lines.append(f"data_status: {r['data_status']}")
    for n in r["notes"]:
        lines.append(f"  nota: {n}")
    lines.append("")
    lines.append("TESE: momentum cross-seccional sobre residuos de modelo de fator")
    lines.append("(CAPM/FF3), long top / short bottom, market-neutral. Hipotese: maior")
    lines.append("Sharpe e mais ortogonal que momentum bruto (que ja reprovou).")
    lines.append("")
    lines.append(f"Config primaria (canonica, NAO a melhor IS): {r['primary']}")
    lines.append(f"n_trials honesto (todo o grid): {r['n_trials']}")
    lines.append(f"meses avaliados: {r['n_months']}")
    lines.append("")
    lines.append("--- METRICAS (carteira long-short, custo ESTRESSADO 2x) ---")
    lines.append(f"Sharpe anual (estressado): {r['sharpe_stress']:.3f}")
    lines.append(f"Sharpe anual (custo base): {r['sharpe_base']:.3f}")
    lines.append(f"CAGR: {r['cagr']*100:.2f}%   MaxDD: {r['maxdd']*100:.2f}%")
    lines.append(f"corr ao SPY: {r['corr_spy']:.3f}")
    lines.append(f"skew: {r['skew']:.2f}  kurtosis: {r['kurtosis']:.2f}")
    lines.append("")
    lines.append("--- TRIBUNAL ESTATISTICO ---")
    lines.append(f"DSR: {r['dsr']:.4f}  (barra >= 0.95)")
    lines.append(f"PSR (SR>0): {r['psr']:.4f}")
    lines.append(f"obstaculo data-snooping (Sharpe anual): {r['sr_benchmark_ann']:.3f}")
    lines.append(f"PBO (CSCV): {r['pbo']:.3f}  (barra < 0.5)")
    lines.append(f"Sharpe IS (70%): {r['sr_is']:.3f}   Sharpe OOS (30%): {r['sr_oos']:.3f}")
    lines.append("")
    lines.append("--- BENCHMARK: BUY&HOLD SPY (mesmo recorte) ---")
    lines.append(f"SPY Sharpe anual: {r['spy_sharpe']:.3f}   SPY CAGR: {r['spy_cagr']*100:.2f}%")
    lines.append(f"bate buy&hold? {r['beats_bh']}")
    lines.append(f"diversifica (corr<0.3 & Sharpe>=0.5)? {r['diversifies']}")
    lines.append(f"robusto OOS (PBO<0.5 & SR_oos>0)? {r['robust_oos']}")
    lines.append("")
    lines.append("--- GRID (todas as trials; Sharpe anual) ---")
    for s in sorted(r["summaries"], key=lambda x: -x["sharpe_ann_stress"]):
        lines.append(f"  {s['label']:<28} SR_stress={s['sharpe_ann_stress']:+.2f} "
                     f"SR_base={s['sharpe_ann_base']:+.2f} n={s['n']}")
    lines.append("")
    lines.append("=" * 78)
    verdict = "PASSA" if r["passed"] else "REPROVA"
    lines.append(f"VEREDITO: {verdict}")
    lines.append("=" * 78)
    REPORT_PATH.write_text("\n".join(lines))
    logger.info("Relatorio -> %s", REPORT_PATH)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    r = run_tribunal()
    if r.get("data_status") == "blocked":
        print("BLOCKED:", r.get("note"))
        return 1
    print("\n".join([
        f"data_status={r['data_status']}",
        f"primary={r['primary']} n_trials={r['n_trials']} months={r['n_months']}",
        f"Sharpe_stress={r['sharpe_stress']:.3f} Sharpe_base={r['sharpe_base']:.3f}",
        f"DSR={r['dsr']:.4f} PSR={r['psr']:.4f} PBO={r['pbo']:.3f}",
        f"corr_SPY={r['corr_spy']:.3f} CAGR={r['cagr']*100:.2f}% MaxDD={r['maxdd']*100:.2f}%",
        f"SPY: Sharpe={r['spy_sharpe']:.3f} CAGR={r['spy_cagr']*100:.2f}%",
        f"IS={r['sr_is']:.3f} OOS={r['sr_oos']:.3f}",
        f"PASSED={r['passed']}",
    ]))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
