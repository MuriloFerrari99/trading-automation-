"""Tribunal R&D — Betting-Against-Beta (BAB) / defensive (low-beta anomaly).

PERGUNTA (imparcial, MEDIR): a anomalia low-beta (Frazzini-Pedersen 2014) entrega
edge risco-ajustado ROBUSTO num universo de ETFs/acoes liquidos? Construcao:
  - beta rolante de cada nome vs SPY (regressao OLS sobre retornos diarios),
  - rank cross-seccional por beta a cada rebalance MENSAL,
  - LONG low-beta (alavancado p/ beta do leg ~1) / SHORT high-beta (de-alavancado
    p/ beta ~1) -> portfolio BAB beta-neutral (a versao academica),
  - tambem testa LONG-ONLY low-beta (tilt defensivo, sem short).

R&D PURO e ISOLADO. NAO toca beta_*, main.py, nav_history nem config de producao.
Cria arquivo NOVO. Importa (read-only):
  - simulation.statistics : evaluate_edge / DSR / PSR / PBO / observed_sharpe
  - simulation.metrics    : max_drawdown
  - simulation.costs      : EQUITY_BASE + estresse 2x

UNIVERSO (dados GRATIS via yfinance, cacheado em data/bab_cache/*.csv):
  ETFs setoriais/regionais + fatores, alta liquidez, historico longo (>~15 anos).

SEM LOOK-AHEAD (rigoroso):
  - beta de cada nome calculado com retornos ATE o fim do mes m (t <= ult. dia de m).
  - pesos do mes m aplicados aos retornos do mes m+1 (weights.shift(1) na sim diaria).
  - SPY (benchmark de beta) usado so com dados <= t-1.

CUSTO: EQUITY_BASE (3 bps/lado) sobre |Delta peso| (turnover real) + cenario 2x.
VEREDITO usa o cenario ESTRESSADO (honesto).

n_TRIALS HONESTO: variantes {BAB beta-neutral L/S, low-beta L/S simples, low-beta LO}
  x janelas de beta {6,12,24m} x cortes {tercil, quartil}. Tudo contado no DSR;
  PBO sobre a matriz de TODAS as configs.

BARRA: PASSA so se DSR>=0.95 E Sharpe_liq robusto E (bate SPY OU corr baixa ao SPY
melhorando o conjunto) E robusto OOS (PBO baixo + split temporal).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE_DIR = Path("data/bab_cache")
REPORT_PATH = Path("data/bab_verdict.txt")

# Universo: ETFs setoriais US + regionais + bonds/real assets. Mistura deliberada
# de high-beta (XLK, XLF, EEM, IWM) e low-beta (XLU, XLP, XLV, TLT) p/ a anomalia
# ter dispersao de beta p/ explorar. Historico longo (maioria desde ~2007).
UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ",
    "XLE", "XLF", "XLK", "XLU", "XLP", "XLV", "XLI", "XLB", "XLY",
    "TLT", "IEF", "LQD", "HYG", "GLD", "VNQ",
]
SPY = "SPY"
MIN_NAMES = 10  # cross-section honesto exige nomes suficientes


# --------------------------------------------------------------------------- #
# DADOS — yfinance, cacheado por ticker (Adj Close = total return).
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.csv"


def _fetch_one(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf

    h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if h is None or len(h) == 0:
        return None
    df = pd.DataFrame({"adj_close": h["Adj Close"].astype(float)})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["adj_close"] > 0]
    return df


def load_panel(force: bool = False) -> pd.DataFrame:
    """Retorna painel (datas x tickers) de Adj Close (total return)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cols = {}
    failed = []
    for tk in UNIVERSE:
        p = _cache_path(tk)
        d = None
        if p.exists() and not force:
            try:
                d = pd.read_csv(p, index_col=0, parse_dates=True)
            except Exception:
                d = None
        if d is None or len(d) == 0:
            d = _fetch_one(tk)
            if d is not None and len(d) > 0:
                d.to_csv(p)
                time.sleep(0.4)
        if d is None or len(d) == 0:
            failed.append(tk)
            continue
        cols[tk] = d["adj_close"]
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    panel.attrs["failed"] = failed
    return panel


# --------------------------------------------------------------------------- #
# SINAL — beta rolante vs SPY, ranking cross-seccional mensal.
# --------------------------------------------------------------------------- #
def rolling_beta(rets: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    """Beta OLS rolante de cada coluna vs mkt (mercado). Cov/Var em janela `window`."""
    mkt = mkt.reindex(rets.index)
    var_m = mkt.rolling(window, min_periods=int(window * 0.6)).var()
    betas = {}
    for col in rets.columns:
        cov = rets[col].rolling(window, min_periods=int(window * 0.6)).cov(mkt)
        betas[col] = cov / var_m
    return pd.DataFrame(betas, index=rets.index)


def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ultimo dia de pregao de cada mes presente no indice."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


@dataclass
class SimResult:
    name: str
    daily_returns: pd.Series  # liquido de custo (cenario base)
    daily_returns_stress: pd.Series  # liquido de custo (cenario 2x)
    turnover_ann: float


def build_weights(
    betas_me: pd.DataFrame,
    *,
    variant: str,
    cut: str,
) -> pd.DataFrame:
    """Pesos mensais (datas-fim-de-mes x tickers) a partir do beta no fim de cada mes.

    variant:
      - "bab"      : BAB beta-neutral. Long low-beta scaled to beta 1, short high-beta
                     scaled to beta 1 (Frazzini-Pedersen). Dollar pode != 0; beta ~0.
      - "ls"       : long low-beta tercil/quartil (EW) - short high-beta (EW), dollar-neutral.
      - "long"     : long-only low-beta tercil/quartil (EW).
    cut: "tercil" (1/3) ou "quartil" (1/4).
    """
    frac = 1.0 / 3.0 if cut == "tercil" else 1.0 / 4.0
    weights = pd.DataFrame(0.0, index=betas_me.index, columns=betas_me.columns)

    for dt, row in betas_me.iterrows():
        b = row.dropna()
        # exclui o proprio benchmark do cross-section (beta=1 trivial) p/ honestidade
        if SPY in b.index:
            b = b.drop(SPY)
        if b.size < MIN_NAMES:
            continue
        n = b.size
        k = max(1, int(np.floor(n * frac)))
        order = b.sort_values()  # ascendente: low-beta primeiro
        low = order.index[:k]
        high = order.index[-k:]

        w = pd.Series(0.0, index=betas_me.columns)
        if variant == "long":
            w.loc[low] = 1.0 / k
        elif variant == "ls":
            w.loc[low] = 0.5 / k
            w.loc[high] = -0.5 / k
        elif variant == "bab":
            # beta de cada perna
            bl = b.loc[low].mean()
            bh = b.loc[high].mean()
            # proteje contra beta nao-positivo (raro): cai p/ ls simples
            if bl is None or bh is None or bl <= 0 or bh <= 0:
                w.loc[low] = 0.5 / k
                w.loc[high] = -0.5 / k
            else:
                # long leg EW, escalado p/ beta 1; short leg EW, escalado p/ beta 1
                w.loc[low] = (1.0 / bl) / k
                w.loc[high] = -(1.0 / bh) / k
        weights.loc[dt] = w
    return weights


def simulate(panel: pd.DataFrame, weights_me: pd.DataFrame, name: str) -> SimResult:
    """Aplica pesos mensais (definidos no fim do mes m) aos retornos diarios do mes m+1.

    weights_me: indexado por datas-fim-de-mes. Reindexa p/ diario com ffill e SHIFT(1)
    p/ o peso so valer a partir do dia SEGUINTE ao calculo (sem look-ahead).
    """
    rets = panel.pct_change().dropna(how="all")
    # pesos diarios: vigentes no dia seguinte ao fim do mes em que foram setados
    w_daily = weights_me.reindex(rets.index, method="ffill").shift(1)
    w_daily = w_daily.reindex(columns=rets.columns).fillna(0.0)

    # retorno bruto do portfolio
    port = (w_daily * rets.reindex(columns=w_daily.columns)).sum(axis=1)

    # turnover: variacao de peso de um dia p/ o outro (so muda nos rebalances)
    dw = w_daily.diff().abs().sum(axis=1).fillna(0.0)
    cost_base = dw * (EQUITY_BASE.per_side_bps / 1e4)
    cost_stress = dw * (EQUITY_BASE.stressed(2.0).per_side_bps / 1e4)

    net_base = (port - cost_base).dropna()
    net_stress = (port - cost_stress).dropna()
    turnover_ann = float(dw.sum() / max(len(rets) / EQUITY_PERIODS, 1e-9))
    return SimResult(name, net_base, net_stress, turnover_ann)


# --------------------------------------------------------------------------- #
# TRIBUNAL
# --------------------------------------------------------------------------- #
def correlation_to_spy(strat: pd.Series, spy_rets: pd.Series) -> float:
    j = pd.concat([strat, spy_rets], axis=1).dropna()
    if len(j) < 30:
        return float("nan")
    return float(j.iloc[:, 0].corr(j.iloc[:, 1]))


def main() -> dict:
    force = "--force" in sys.argv
    panel = load_panel(force=force)
    lines: list[str] = []

    def log(s: str = ""):
        lines.append(s)
        print(s)

    log("=" * 78)
    log("TRIBUNAL R&D — Betting-Against-Beta (BAB) / defensive (low-beta anomaly)")
    log("=" * 78)

    if panel.empty:
        REPORT_PATH.write_text("DATA BLOCKED: yfinance retornou vazio p/ todo o universo.\n")
        return {"data_status": "blocked", "reason": "yfinance vazio"}

    failed = panel.attrs.get("failed", [])
    panel = panel.dropna(how="all")
    # exige >= MIN_NAMES nomes vivos p/ rodar o cross-section
    live_cols = [c for c in panel.columns if panel[c].notna().sum() > 252 * 3]
    panel = panel[live_cols]
    if SPY not in panel.columns:
        REPORT_PATH.write_text("DATA BLOCKED: SPY (benchmark de beta) ausente.\n")
        return {"data_status": "blocked", "reason": "SPY ausente"}

    start = panel.index.min()
    end = panel.index.max()
    n_years = (end - start).days / 365.25
    log(f"\nUniverso vivo: {len(panel.columns)} tickers | falhas: {failed}")
    log(f"Periodo: {start.date()} -> {end.date()} ({n_years:.1f} anos)")
    log(f"Nomes: {list(panel.columns)}")

    if len(panel.columns) < MIN_NAMES + 1:  # +1 p/ SPY
        REPORT_PATH.write_text(
            f"DATA LIMITED: so {len(panel.columns)} tickers vivos (< {MIN_NAMES+1}).\n"
        )
        return {"data_status": "limited", "reason": "poucos nomes"}

    rets = panel.pct_change().dropna(how="all")
    spy_rets = rets[SPY]
    me_idx = month_end_index(panel.index)

    # ---- grid de configs (n_trials honesto) ----
    windows = {"6m": 126, "12m": 252, "24m": 504}
    variants = ["bab", "ls", "long"]
    cuts = ["tercil", "quartil"]

    sims: dict[str, SimResult] = {}
    for wname, wlen in windows.items():
        betas = rolling_beta(rets.drop(columns=[]), spy_rets, wlen)
        # beta no fim de cada mes (usa info <= fim do mes -> sem look-ahead)
        betas_me = betas.reindex(me_idx).dropna(how="all")
        for variant in variants:
            for cut in cuts:
                w = build_weights(betas_me, variant=variant, cut=cut)
                if (w.abs().sum(axis=1) == 0).all():
                    continue
                name = f"{variant}_{wname}_{cut}"
                sims[name] = simulate(panel, w, name)

    n_trials = len(sims)
    log(f"\nConfigs testadas (n_trials honesto): {n_trials}")

    if n_trials == 0:
        REPORT_PATH.write_text("DATA LIMITED: nenhuma config produziu pesos validos.\n")
        return {"data_status": "limited", "reason": "sem pesos"}

    # ---- alinha series p/ matriz PBO (mesmo indice) ----
    aligned = pd.DataFrame({k: v.daily_returns_stress for k, v in sims.items()}).dropna()
    pbo = probability_of_backtest_overfitting(aligned.values, n_splits=16)

    # SPY buy&hold no mesmo periodo (benchmark)
    spy_bh = spy_rets.reindex(aligned.index).dropna()
    spy_sharpe = observed_sharpe(spy_bh.values, EQUITY_PERIODS)
    spy_eq = (1 + spy_bh).cumprod().values
    spy_cagr = float(spy_eq[-1] ** (EQUITY_PERIODS / len(spy_eq)) - 1)
    spy_mdd = max_drawdown(spy_eq)
    log(f"\nBenchmark SPY (mesmo periodo): Sharpe={spy_sharpe:.2f} "
        f"CAGR={spy_cagr:.2%} MaxDD={spy_mdd:.2%}")

    # ---- avalia cada config; escolhe a melhor por Sharpe ESTRESSADO ----
    log("\n" + "-" * 78)
    log(f"{'config':<22}{'Shrp_str':>9}{'CAGR':>8}{'MaxDD':>8}{'corrSPY':>9}{'turn':>7}")
    log("-" * 78)
    table = []
    for name, sr in sorted(sims.items()):
        r = sr.daily_returns_stress.reindex(aligned.index).dropna()
        if len(r) < 252:
            continue
        sh = observed_sharpe(r.values, EQUITY_PERIODS)
        eq = (1 + r).cumprod().values
        cagr = float(eq[-1] ** (EQUITY_PERIODS / len(eq)) - 1) if eq[-1] > 0 else -1.0
        mdd = max_drawdown(eq)
        corr = correlation_to_spy(r, spy_bh)
        table.append((name, sh, cagr, mdd, corr, sr.turnover_ann))
        log(f"{name:<22}{sh:>9.2f}{cagr:>8.1%}{mdd:>8.1%}{corr:>9.2f}{sr.turnover_ann:>7.1f}")

    if not table:
        REPORT_PATH.write_text("DATA LIMITED: series curtas demais p/ avaliar.\n")
        return {"data_status": "limited", "reason": "series curtas"}

    best = max(table, key=lambda t: t[1])
    best_name = best[0]
    best_sim = sims[best_name]
    log("-" * 78)
    log(f"\nMELHOR (por Sharpe estressado): {best_name}")

    best_r = best_sim.daily_returns_stress.reindex(aligned.index).dropna()
    verdict = evaluate_edge(
        best_r.values,
        n_trials=n_trials,
        periods_per_year=EQUITY_PERIODS,
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )
    best_eq = (1 + best_r).cumprod().values
    best_cagr = float(best_eq[-1] ** (EQUITY_PERIODS / len(best_eq)) - 1) if best_eq[-1] > 0 else -1.0
    best_mdd = max_drawdown(best_eq)
    best_corr = correlation_to_spy(best_r, spy_bh)

    # ---- robustez OOS: split temporal 50/50 ----
    half = len(best_r) // 2
    sh_is = observed_sharpe(best_r.values[:half], EQUITY_PERIODS)
    sh_oos = observed_sharpe(best_r.values[half:], EQUITY_PERIODS)

    log("\n" + verdict.summary())
    log(f"CAGR={best_cagr:.2%}  MaxDD={best_mdd:.2%}  corr_SPY={best_corr:.2f}  "
        f"turnover_ann={best_sim.turnover_ann:.1f}x")
    log(f"PBO (todas as configs, estressado) = {pbo:.3f}")
    log(f"Split temporal: Sharpe IS(1a metade)={sh_is:.2f}  OOS(2a metade)={sh_oos:.2f}")

    # ---- BARRA de graduacao ----
    beats_spy = verdict.sharpe_annual > spy_sharpe
    diversifies = (abs(best_corr) < 0.3) and (verdict.sharpe_annual > 0.5)
    robust_oos = (pbo < 0.5) and (sh_oos > 0) and (sh_is > 0)
    passed = bool(
        verdict.passes_dsr
        and verdict.passes_sharpe
        and (beats_spy or diversifies)
        and robust_oos
    )

    log("\n" + "=" * 78)
    log("BARRA DE GRADUACAO")
    log("=" * 78)
    log(f"  DSR>=0.95 ............... {verdict.dsr:.3f}  -> {verdict.passes_dsr}")
    log(f"  Sharpe_liq>=0.8 ........ {verdict.sharpe_annual:.2f}  -> {verdict.passes_sharpe}")
    log(f"  bate SPY (Sharpe) ...... {verdict.sharpe_annual:.2f} vs {spy_sharpe:.2f} -> {beats_spy}")
    log(f"  diversifica (|corr|<0.3 & Sh>0.5) -> {diversifies}")
    log(f"  robusto OOS (PBO<0.5 & OOS>0) ... PBO={pbo:.3f} OOS={sh_oos:.2f} -> {robust_oos}")
    log(f"\n  VEREDITO: {'PASSA' if passed else 'REPROVADO'}")
    log("=" * 78)

    REPORT_PATH.write_text("\n".join(lines) + "\n")

    return {
        "data_status": "real",
        "strategy_best": best_name,
        "n_trials": n_trials,
        "n_obs": verdict.n_obs,
        "sharpe": round(verdict.sharpe_annual, 3),
        "dsr": round(verdict.dsr, 3),
        "psr": round(verdict.psr, 3),
        "pbo": round(pbo, 3),
        "cagr": round(best_cagr, 4),
        "maxdd": round(best_mdd, 4),
        "corr_spy": round(best_corr, 3),
        "spy_sharpe": round(spy_sharpe, 3),
        "sh_is": round(sh_is, 3),
        "sh_oos": round(sh_oos, 3),
        "beats_spy": beats_spy,
        "diversifies": diversifies,
        "robust_oos": robust_oos,
        "passed": passed,
    }


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
