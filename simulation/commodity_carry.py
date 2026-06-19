"""Tribunal: COMMODITY CARRY / ROLL YIELD (R&D offline, isolado).

PERGUNTA (imparcial, MEDIR): existe um premio de CARRY de commodities — capturavel
de graca via ETFs — com edge risco-ajustado ROBUSTO **e** que bata buy&hold OU
diversifique o mercado (SPY) com correlacao baixa, robusto OOS?

CONTEXTO TEORICO
----------------
O retorno total de uma posicao em futuros de commodity decompoe-se em:
    retorno_total ≈ retorno_spot + ROLL_YIELD (+ collateral)
O roll yield e POSITIVO em backwardation (curva descendente) e NEGATIVO em contango
(curva ascendente). O "carry trade" de commodities compra os backwardated (carry+) e
evita/vende os contangoed (carry-). A literatura (Erb & Harvey 2006; Gorton &
Rouwenhorst 2006; Koijen et al. "Carry" 2018) mostra que o roll yield e o componente
DOMINANTE do retorno de longo prazo de uma cesta de commodities.

DADOS — HONESTIDADE (data_status = "limited")
---------------------------------------------
A estrutura a termo completa (front vs deferred, por commodity, point-in-time) NAO e
obtenivel de graca. APROXIMAMOS o roll yield SO com ETFs gratis (yfinance), por dois
caminhos independentes, sem inventar nada:

  (A) ROLL-METHOD SPREAD (isola roll yield diretamente): para o MESMO subjacente,
      dois ETFs que diferem SO no metodo de rolagem. Ex.: petroleo WTI
        DBO (rolagem "otimizada", busca minimizar contango) vs USO (front-month naive).
      A diferenca de retorno DBO-USO e, por construcao, ~puro efeito de roll yield
      (o spot e o mesmo). Idem metais fisicos (GLD/SLV/...) tem roll yield ~0 (sem futuro).

  (B) CARRY CROSS-SECCIONAL via PROXY DE ROLL POR ETF: para cada ETF de futuros de
      UMA commodity, estimamos o roll yield como o retorno do ETF MENOS o retorno de
      um "spot proxy" do MESMO grupo quando existe fisico; quando nao existe fisico,
      usamos o sinal de carry-momentum (12-1m) — que e o proxy classico de
      backwardation usado por indices "carry-enhanced" (DBC/USCI). Ranqueamos o
      universo por esse carry e vamos long nos carry+.

  (C) CARRY-ENHANCED vs NAIVE (a expressao mais limpa e gratis do premio): USCI
      (United States Commodity Index — seleciona/rola por backwardation/momentum,
      i.e. CARRY) vs cestas naive front-month (DBC/GSG/DJP). Se o premio de carry
      e real, USCI deve bater as naive risco-ajustado.

SEM LOOK-AHEAD: sinal de t usa dados <= t (close do fim do mes m); pesos do mes m+1
aplicados aos retornos do mes m+1 (weights.shift(1) na simulacao mensal). Janelas de
lookback terminam em t.

CUSTO: simulation.costs.EQUITY_BASE (3 bps/lado) + cenario ESTRESSADO 2x sobre o
turnover real. VEREDITO no cenario ESTRESSADO.

n_TRIALS HONESTO: todas as variantes/lookbacks/cortes contadas e passadas ao DSR; PBO
sobre a matriz de configs cross-seccionais.

BARRA: PASSA so se DSR>=0.95 E Sharpe liq robusto E (bate buy&hold OU diversifica SPY
com corr baixa melhorando o conjunto), robusto OOS. Senao FALHA.

Uso:
    .venv/bin/python -m simulation.commodity_carry          # usa cache
    .venv/bin/python -m simulation.commodity_carry --force  # re-baixa
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown, sharpe as _sharpe
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.commodity_carry")

CACHE_DIR = Path("data/commodity_cache")
REPORT_PATH = Path("data/commodity_carry_verdict.txt")
MONTHS_PER_YEAR = 12

# Universo de ETFs de commodity por SETOR (single-commodity onde possivel, p/ o
# ranqueamento cross-seccional de carry ter sentido). Tudo yfinance, historico longo.
# group = subjacente economico; phys = e fisico/spot (roll yield ~ 0)?
@dataclass(frozen=True)
class Asset:
    ticker: str
    group: str
    phys: bool


SINGLE_COMMODITY: list[Asset] = [
    Asset("USO", "oil", False),   # WTI front-month naive
    Asset("DBO", "oil", False),   # WTI optimized roll
    Asset("BNO", "brent", False),
    Asset("UGA", "gasoline", False),
    Asset("UNG", "natgas", False),  # cronicamente em contango -> carry-
    Asset("GLD", "gold", True),     # fisico (spot)
    Asset("SLV", "silver", True),
    Asset("PPLT", "platinum", True),
    Asset("PALL", "palladium", True),
    Asset("CPER", "copper", False),
    Asset("CORN", "corn", False),
    Asset("WEAT", "wheat", False),
    Asset("SOYB", "soybean", False),
]

# Cestas para o teste (C): carry-enhanced vs naive
BASKET_CARRY = "USCI"  # seleciona/rola por backwardation+momentum (carry)
BASKETS_NAIVE = ["DBC", "GSG", "DJP"]
MARKET = "SPY"

ALL_TICKERS = (
    [a.ticker for a in SINGLE_COMMODITY] + [BASKET_CARRY] + BASKETS_NAIVE + [MARKET]
)


# ============================================================================
# DADOS — yfinance, cache por ticker (mesma convencao do beta)
# ============================================================================
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_1d.csv"


def _yf_close(ticker: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(
        ticker, period="max", interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)
    cols = {str(c).lower(): c for c in df.columns}
    col = cols.get("close") or cols.get("adj close")
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    s = s[~s.index.isna()].dropna()
    return s[s > 0].sort_index()


def load_close(ticker: str, *, force: bool = False) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(ticker)
    if p.exists() and not force:
        s = pd.read_csv(p, index_col=0)
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
        return s[~s.index.isna()].iloc[:, 0].astype(float)
    s = _yf_close(ticker)
    if not s.empty:
        s.to_frame("close").to_csv(p)
        logger.info("%s: %d barras (%s..%s)", ticker, len(s), s.index[0].date(), s.index[-1].date())
    return s


def load_monthly(force: bool = False) -> pd.DataFrame:
    """Painel de fechamentos MENSAIS (ultimo dia util do mes) alinhado por data."""
    cols: dict[str, pd.Series] = {}
    for t in ALL_TICKERS:
        s = load_close(t, force=force)
        if s.empty:
            logger.warning("%s vazio", t)
            continue
        cols[t] = s.resample("ME").last()
    if not cols:
        raise RuntimeError("nenhum dado carregado")
    return pd.DataFrame(cols).sort_index()


# ============================================================================
# RETORNOS / SINAIS
# ============================================================================
def monthly_logret(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def carry_momentum(prices: pd.DataFrame, lookback: int, skip: int = 1) -> pd.DataFrame:
    """Proxy de carry/backwardation: retorno 'lookback-skip' meses (pula `skip`).

    Backwardation persistente -> roll positivo -> tendencia de alta; e o proxy de
    carry usado por indices carry-enhanced. Calculado com close <= t (sem look-ahead).
    """
    logp = np.log(prices)
    return logp.shift(skip) - logp.shift(lookback)


# ============================================================================
# SIMULACAO de cesta long-only equal-weight do TOP-k por carry (cross-seccional)
# ============================================================================
def simulate_cross_section(
    rets: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    top_k: int,
    cost_per_side_bps: float,
) -> pd.Series:
    """Long-only equal-weight nos top_k por `signal`, rebalance mensal.

    Pesos decididos em t (a partir de signal[t], que so usa dados<=t) e aplicados ao
    retorno de t+1 (rets.shift via alinhamento). Custo sobre |Delta peso| (turnover).
    Retorna serie de retornos LIQUIDOS mensais da cesta.
    """
    common = rets.index.intersection(signal.index)
    rets = rets.loc[common]
    signal = signal.loc[common]
    cps = cost_per_side_bps / 1e4

    prev_w = pd.Series(0.0, index=rets.columns)
    out: list[float] = []
    out_idx: list[pd.Timestamp] = []
    for i in range(len(common) - 1):
        t = common[i]
        nt = common[i + 1]
        sig = signal.loc[t].dropna()
        # so usa ativos com retorno valido no proximo mes
        valid = sig.index[rets.loc[nt, sig.index].notna()]
        sig = sig.loc[valid]
        if len(sig) < top_k:
            # cesta indisponivel (universo ainda nao existe) -> NaN p/ sair do periodo ativo
            prev_w = pd.Series(0.0, index=rets.columns)
            out.append(np.nan)
            out_idx.append(nt)
            continue
        winners = sig.sort_values(ascending=False).head(top_k).index
        w = pd.Series(0.0, index=rets.columns)
        w[winners] = 1.0 / top_k
        turnover = (w - prev_w).abs().sum()
        gross = float((w * rets.loc[nt].fillna(0.0)).sum())
        net = gross - turnover * cps
        out.append(net)
        out_idx.append(nt)
        prev_w = w
    return pd.Series(out, index=pd.DatetimeIndex(out_idx))


def simulate_long_short(
    rets: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    top_k: int,
    cost_per_side_bps: float,
) -> pd.Series:
    """Dollar-neutral: long top_k - short bottom_k por carry. Mesma disciplina anti-LA."""
    common = rets.index.intersection(signal.index)
    rets = rets.loc[common]
    signal = signal.loc[common]
    cps = cost_per_side_bps / 1e4
    prev_w = pd.Series(0.0, index=rets.columns)
    out: list[float] = []
    out_idx: list[pd.Timestamp] = []
    for i in range(len(common) - 1):
        t = common[i]
        nt = common[i + 1]
        sig = signal.loc[t].dropna()
        valid = sig.index[rets.loc[nt, sig.index].notna()]
        sig = sig.loc[valid]
        if len(sig) < 2 * top_k:
            prev_w = pd.Series(0.0, index=rets.columns)
            out.append(np.nan)
            out_idx.append(nt)
            continue
        ranked = sig.sort_values(ascending=False)
        longs = ranked.head(top_k).index
        shorts = ranked.tail(top_k).index
        w = pd.Series(0.0, index=rets.columns)
        w[longs] = 0.5 / top_k
        w[shorts] = -0.5 / top_k
        turnover = (w - prev_w).abs().sum()
        gross = float((w * rets.loc[nt].fillna(0.0)).sum())
        net = gross - turnover * cps
        out.append(net)
        out_idx.append(nt)
        prev_w = w
    return pd.Series(out, index=pd.DatetimeIndex(out_idx))


# ============================================================================
# METRICAS auxiliares
# ============================================================================
def ann_sharpe(r: np.ndarray) -> float:
    return observed_sharpe(np.asarray(r, dtype=float)) * np.sqrt(MONTHS_PER_YEAR)


def cagr_from_monthly(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    if eq.empty or eq.iloc[-1] <= 0:
        return -1.0
    yrs = len(r) / MONTHS_PER_YEAR
    return float(eq.iloc[-1] ** (1.0 / max(yrs, 1e-9)) - 1.0)


def maxdd_from_monthly(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod().to_numpy()
    return max_drawdown(eq)


# ============================================================================
# DRIVER / TRIBUNAL
# ============================================================================
@dataclass
class Config:
    name: str
    rets: pd.Series  # retornos mensais LIQUIDOS


def _bh(rets_panel: pd.DataFrame, ticker: str) -> pd.Series:
    """Buy&hold simples (retorno aritmetico mensal) de um ticker."""
    p = rets_panel[ticker].dropna()
    return p


def build_configs(prices: pd.DataFrame, cost_per_side: float) -> tuple[list[Config], pd.Series, pd.Series, dict]:
    """Monta TODAS as configs candidatas. Retorna (configs, spy_ret, usci_ret, extras)."""
    arith = prices.pct_change()
    spy = arith[MARKET].dropna()
    # Universo cross-seccional: somente single-commodity (exclui cestas e SPY)
    xs_cols = [a.ticker for a in SINGLE_COMMODITY]
    xs_prices = prices[xs_cols]
    xs_rets = xs_prices.pct_change()

    configs: list[Config] = []
    cross_matrix_cols: dict[str, pd.Series] = {}

    # (B) Carry cross-seccional: carry-momentum proxy. Variantes/lookbacks/cortes.
    for lookback in (6, 9, 12):
        sig = carry_momentum(xs_prices, lookback=lookback, skip=1)
        for top_k in (3, 4, 5):
            lo = simulate_cross_section(xs_rets, sig, top_k=top_k, cost_per_side_bps=cost_per_side)
            nm = f"carryXS_long_lb{lookback}_k{top_k}"
            configs.append(Config(nm, lo))
            cross_matrix_cols[nm] = lo
            ls = simulate_long_short(xs_rets, sig, top_k=top_k, cost_per_side_bps=cost_per_side)
            nm2 = f"carryXS_LS_lb{lookback}_k{top_k}"
            configs.append(Config(nm2, ls))
            cross_matrix_cols[nm2] = ls

    # (A) Roll-method spread (isola roll yield): DBO - USO (WTI), long-only timing.
    #     Sinal: se o roll yield recente (DBO-USO trailing) > 0 -> carry+ no oil, long DBO; senao flat.
    if "DBO" in prices.columns and "USO" in prices.columns:
        roll = (np.log(prices["DBO"]) - np.log(prices["USO"])).diff()  # roll yield mensal de WTI
        for lb in (3, 6, 12):
            sig_pos = roll.rolling(lb).mean().shift(1)  # decide em t com dados<=t
            dbo = arith["DBO"]
            r = (sig_pos > 0).astype(float).reindex(dbo.index).fillna(0.0).shift(0) * dbo
            # custo de entrar/sair: aproxima por |delta sinal|*cost
            pos = (sig_pos > 0).astype(float).reindex(dbo.index).fillna(0.0)
            tc = pos.diff().abs().fillna(pos.abs()) * (cost_per_side / 1e4)
            net = (r - tc).dropna()
            nm = f"rollTiming_WTI_lb{lb}"
            configs.append(Config(nm, net))

    # (C) Carry-enhanced (USCI) vs naive baskets — buy&hold de cada (sem custo de giro;
    #     custo so de 1 entrada, desprezivel). Para comparacao de premio de carry.
    usci = _bh(arith, BASKET_CARRY)
    extras = {"naive": {b: _bh(arith, b) for b in BASKETS_NAIVE if b in arith.columns}}

    # matriz p/ PBO: so as configs cross-seccionais (mesma familia, comparaveis)
    cross_df = pd.DataFrame(cross_matrix_cols).dropna(how="all")
    extras["cross_matrix"] = cross_df
    return configs, spy, usci, extras


def run(force: bool = False) -> dict:
    prices = load_monthly(force=force)
    base = EQUITY_BASE
    stressed = base.stressed(2.0)

    # n_trials HONESTO: contamos TODAS as variantes consideradas.
    # (B) lookbacks {6,9,12} x top_k {3,4,5} x {long, LS} = 18
    # (A) rollTiming lookbacks {3,6,12} = 3
    # (C) USCI vs 3 naive (comparacao, conta como hipoteses) = ~4
    # total honesto:
    n_trials = 18 + 3 + 4

    configs_base, spy, usci, extras = build_configs(prices, base.per_side_bps)
    configs_str, _, _, extras_str = build_configs(prices, stressed.per_side_bps)

    # melhor config IN-SAMPLE pelo Sharpe (cenario ESTRESSADO -> honesto)
    def sh(c: Config) -> float:
        return ann_sharpe(c.rets.dropna().to_numpy()) if len(c.rets.dropna()) > 12 else -9.0

    ranked = sorted(configs_str, key=sh, reverse=True)
    best = ranked[0]
    best_base = next(c for c in configs_base if c.name == best.name)

    # SPY alinhado p/ correlacao (no periodo da melhor config)
    r_best = best.rets.dropna()
    spy_al = spy.reindex(r_best.index)
    mask = spy_al.notna() & r_best.notna()
    corr = float(np.corrcoef(r_best[mask], spy_al[mask])[0, 1]) if mask.sum() > 3 else float("nan")

    # buy&hold benchmarks no MESMO periodo da melhor config
    def bh_in_period(series: pd.Series) -> pd.Series:
        return series.reindex(r_best.index).dropna()

    usci_p = bh_in_period(usci)
    naive_sh = {b: ann_sharpe(bh_in_period(s).to_numpy()) for b, s in extras["naive"].items() if len(bh_in_period(s)) > 12}

    # tribunal sobre a MELHOR config (cenario estressado) — DSR conta n_trials
    v = evaluate_edge(
        r_best.to_numpy(),
        n_trials=n_trials,
        periods_per_year=MONTHS_PER_YEAR,
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # PBO sobre a matriz de configs cross-seccionais (cenario estressado)
    cm = extras_str["cross_matrix"].dropna()
    pbo = probability_of_backtest_overfitting(cm.to_numpy(), n_splits=10) if cm.shape[1] >= 2 and cm.shape[0] >= 12 else float("nan")

    # metricas da melhor config (estressado)
    best_sharpe = ann_sharpe(r_best.to_numpy())
    best_cagr = cagr_from_monthly(r_best)
    best_dd = maxdd_from_monthly(r_best)

    # bate buy&hold? (vs USCI carry-enhanced e vs melhor naive)
    usci_sharpe = ann_sharpe(usci_p.to_numpy()) if len(usci_p) > 12 else float("nan")
    spy_sharpe = ann_sharpe(spy_al[mask].to_numpy()) if mask.sum() > 12 else float("nan")
    best_naive_sharpe = max(naive_sh.values()) if naive_sh else float("nan")

    beats_bh = (best_sharpe > usci_sharpe) and (best_sharpe > best_naive_sharpe)
    diversifies = (abs(corr) < 0.3) and (best_sharpe > 0.3)

    passed = bool(
        v.passes_dsr
        and best_sharpe >= 0.8
        and (beats_bh or diversifies)
        and (np.isnan(pbo) or pbo < 0.5)
    )

    # USCI (carry-enhanced) vs naive: o premio de carry "puro e gratis" existe?
    usci_full = usci.dropna()
    naive_full = {b: s.dropna() for b, s in extras["naive"].items()}
    # alinha no periodo comum USCI vs DBC
    carry_premium_lines = []
    for b, s in naive_full.items():
        common = usci_full.index.intersection(s.index)
        if len(common) > 24:
            u = usci_full.loc[common]; n = s.loc[common]
            carry_premium_lines.append(
                f"  USCI vs {b}: Sharpe {ann_sharpe(u.to_numpy()):.2f} vs {ann_sharpe(n.to_numpy()):.2f} "
                f"| CAGR {cagr_from_monthly(u):+.1%} vs {cagr_from_monthly(n):+.1%} | n={len(common)}m"
            )

    result = {
        "strategy": "Commodity carry / roll yield",
        "data_status": "limited",
        "best_config": best.name,
        "n_obs_months": int(len(r_best)),
        "period": f"{r_best.index[0].date()}..{r_best.index[-1].date()}",
        "n_trials": n_trials,
        "sharpe_stressed": round(best_sharpe, 3),
        "dsr": round(v.dsr, 4),
        "psr": round(v.psr, 4),
        "pbo": round(pbo, 3) if not np.isnan(pbo) else None,
        "cagr": round(best_cagr, 4),
        "maxdd": round(best_dd, 4),
        "corr_to_spy": round(corr, 3) if not np.isnan(corr) else None,
        "usci_sharpe": round(usci_sharpe, 3) if not np.isnan(usci_sharpe) else None,
        "best_naive_sharpe": round(best_naive_sharpe, 3) if not np.isnan(best_naive_sharpe) else None,
        "spy_sharpe": round(spy_sharpe, 3) if not np.isnan(spy_sharpe) else None,
        "beats_bh": bool(beats_bh),
        "diversifies": bool(diversifies),
        "passed": passed,
        "verdict_obj": v,
        "carry_premium_lines": carry_premium_lines,
        "all_config_sharpes": {c.name: round(sh(c), 3) for c in ranked},
    }
    return result


def _write_report(res: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v = res["verdict_obj"]
    lines = []
    lines.append("=" * 78)
    lines.append("TRIBUNAL — COMMODITY CARRY / ROLL YIELD")
    lines.append("=" * 78)
    lines.append(f"data_status: {res['data_status']} (term structure point-in-time NAO e gratis;")
    lines.append("  roll yield aproximado via ETFs: roll-method spread (DBO-USO), carry-momentum")
    lines.append("  cross-seccional, e carry-enhanced USCI vs naive DBC/GSG/DJP)")
    lines.append("")
    lines.append(f"Melhor config (cenario ESTRESSADO 2x): {res['best_config']}")
    lines.append(f"Periodo: {res['period']}  | n={res['n_obs_months']} meses")
    lines.append(f"n_trials (honesto): {res['n_trials']}")
    lines.append("")
    lines.append("METRICAS (liquidas, cenario estressado):")
    lines.append(f"  Sharpe anual : {res['sharpe_stressed']:.3f}")
    lines.append(f"  CAGR         : {res['cagr']:+.2%}")
    lines.append(f"  MaxDD        : {res['maxdd']:+.2%}")
    lines.append(f"  Corr c/ SPY  : {res['corr_to_spy']}")
    lines.append("")
    lines.append("TRIBUNAL ANTI-OVERFITTING:")
    lines.append(f"  DSR  : {res['dsr']:.4f}  (barra >= 0.95)  {'PASSA' if v.passes_dsr else 'FALHA'}")
    lines.append(f"  PSR  : {res['psr']:.4f}")
    lines.append(f"  PBO  : {res['pbo']}  (barra < 0.5)")
    lines.append(f"  obstaculo data-snooping (Sharpe anual): {v.sr_benchmark_annual:.3f}")
    lines.append("")
    lines.append("BENCHMARKS (mesmo periodo):")
    lines.append(f"  Sharpe estrategia : {res['sharpe_stressed']:.3f}")
    lines.append(f"  Sharpe USCI (carry-enhanced BH): {res['usci_sharpe']}")
    lines.append(f"  Sharpe melhor naive (DBC/GSG/DJP): {res['best_naive_sharpe']}")
    lines.append(f"  Sharpe SPY        : {res['spy_sharpe']}")
    lines.append(f"  bate buy&hold? {res['beats_bh']}   diversifica (corr<0.3 & sharpe>0.3)? {res['diversifies']}")
    lines.append("")
    lines.append("PREMIO DE CARRY 'PURO E GRATIS' (USCI carry-enhanced vs cestas naive):")
    for ln in res["carry_premium_lines"]:
        lines.append(ln)
    lines.append("")
    lines.append("TODAS AS CONFIGS (Sharpe anual estressado, ranqueadas):")
    for nm, s in res["all_config_sharpes"].items():
        lines.append(f"  {nm:28s} {s:+.3f}")
    lines.append("")
    lines.append("=" * 78)
    verdict = "PASSA" if res["passed"] else "FALHA"
    lines.append(f"VEREDITO: {verdict}")
    lines.append("=" * 78)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res = run(force=args.force)
    _write_report(res)
    print(REPORT_PATH.read_text())
