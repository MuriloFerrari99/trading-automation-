"""TSMOM cross-asset (Trend-following / Managed-futures) — TRIBUNAL standalone.

R&D ISOLADO. NAO toca beta_*, main.py, nav_history nem config de producao. Cria
seu PROPRIO cache (data/tsmom_cache) e seu PROPRIO relatorio (data/tsmom_verdict.txt).
IMPORTA o tribunal compartilhado (simulation.statistics, .metrics, .costs).

TESE (Moskowitz, Ooi, Pedersen 2012 — "Time Series Momentum"): o retorno passado de
um ativo preve o seu proprio retorno futuro (12m pra cima, ~1m pra baixo). Um book
LONG/SHORT cross-asset, dimensionado por inverso-da-vol e escalado a uma vol-alvo de
portfolio, captura um premio persistente e — crucial aqui — DIVERSIFICANTE: skew
POSITIVO e correlacao BAIXA/NEGATIVA ao SPY nos crashes ("crisis alpha"), porque
des-arrisca/shorta em bear sustentado.

ESTE e o candidato a DIVERSIFICADOR (vs os alphas price-based ja reprovados, que eram
beta de bull). A barra reflete isso: PASSA se DSR>=0.95 E Sharpe_liq robusto E (bate
buy&hold do SPY OU diversifica — corr BAIXA ao SPY + melhora o conjunto 60/40).

UNIVERSO (~18 ETFs liquidos, total-return via auto_adjust, period=max -> 2006+,
cobre 2008/2020/2022): equities US/intl, bonds (curva), commodities, ouro/prata, REIT, FX.

HONESTIDADE (nao-negociavel):
  - SEM LOOK-AHEAD: sinal e vol no dia t usam dados ATE t-1 (.shift(1)); o retorno
    aplicado e o de t. Sinal de t com dado<=t-1, retorno t.
  - Custo real (simulation.costs.EQUITY_BASE e o cenario 2x estressado) sobre
    |Delta peso| (turnover), incluindo virar long->short. Sem custo de financiamento
    porque o gross e mantido <= 1.0 (vol-target de portfolio escala o book inteiro
    p/ baixo, nao alavanca acima de 1 — managed-futures conservador, sem margem).
  - n_trials HONESTO: conta TODAS as variantes de lookback/combinacao testadas no DSR
    (anti data-snooping). PBO via CSCV sobre a matriz de variantes.
  - Vereditos reportam EXPLICITAMENTE: corr ao SPY, performance nos piores decis de
    meses do SPY (e 2008/2020/2022), skew, vs SPY e vs 60/40.

Uso:
    uv run python -m simulation.tsmom_tribunal
    uv run python -m simulation.tsmom_tribunal --force   # re-baixa o cache yfinance
"""

from __future__ import annotations

import argparse
import logging
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
    skew,
)

logger = logging.getLogger("simulation.tsmom_tribunal")

CACHE_DIR = Path("data/tsmom_cache")
REPORT_PATH = Path("data/tsmom_verdict.txt")

# Universo managed-futures-like via ETFs liquidos. Total-return (auto_adjust).
# Diversificado por classe: o TSMOM precisa de classes DESCORRELACIONADAS p/ o
# vol-target de portfolio funcionar como diversificador.
UNIVERSE: dict[str, str] = {
    # Equities
    "SPY": "equity",   # US large cap
    "EFA": "equity",   # intl developed
    "EEM": "equity",   # emerging
    "IWM": "equity",   # US small cap
    # Bonds (curva)
    "TLT": "bond",     # US 20y+
    "IEF": "bond",     # US 7-10y
    "LQD": "bond",     # IG credit
    "HYG": "bond",     # high yield
    # Commodities
    "DBC": "commodity",  # broad commodity
    "USO": "commodity",  # oil
    "GLD": "metal",      # gold
    "SLV": "metal",      # silver
    # Real estate
    "VNQ": "reit",
    # FX (dollar e moedas via ETF)
    "UUP": "fx",   # USD bull
    "FXE": "fx",   # euro
    "FXY": "fx",   # yen
}
SPY_TICKER = "SPY"

TRADING_DAYS = EQUITY_PERIODS  # 252

# Lookbacks de momentum em dias de pregao (1/3/6/12 meses). Combinacao = media dos
# sinais. Cada item destes e uma VARIANTE testada -> entra no n_trials do DSR.
LOOKBACKS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}
# Variantes de combinacao de sinal testadas (cada uma e um trial honesto):
COMBO_VARIANTS = ["1m", "3m", "6m", "12m", "3-6-12", "1-3-6-12"]

VOL_LOOKBACK = 63          # janela de vol p/ inverso-da-vol e vol-target (3m)
PER_ASSET_VOL_TARGET = 0.40 / np.sqrt(TRADING_DAYS)  # 40% a.a. por nome (pre-cap de gross)
PORT_VOL_TARGET_ANNUAL = 0.10   # 10% a.a. de vol de portfolio (managed-futures tipico)
MAX_GROSS = 1.0            # SEM alavancagem acima de 1 -> sem custo de financiamento
REBAL = 5                  # rebalance semanal (reduz turnover/custo; 5 dias de pregao)
# Inicio HONESTO do book cross-asset: so a partir de quando o universo esta
# razoavelmente populado. Antes disso (1993-2006) so existe SPY/poucos ETFs e o
# "cross-asset" degenera p/ momentum de 1 ativo -> nao representa a estrategia e
# distorce as estatisticas de diversificacao. Todos os 16 ETFs existem em 2007.
START_DATE = "2007-06-01"


# ============================================================================
# DADOS
# ============================================================================
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_1d.csv"


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
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(ticker)
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]
    raw = yf.download(
        ticker, period="max", interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )
    df = _normalize(raw, ticker)
    if not df.empty:
        df.to_csv(p)
        logger.info("%s: %d barras (%s..%s)", ticker, len(df),
                    df.index[0].date(), df.index[-1].date())
    return df


def load_panel(force: bool = False) -> pd.DataFrame:
    """Painel de fechamentos total-return alinhado por dia de pregao (intersecao
    relaxada: union de datas, ffill curto). Retorna DataFrame (datas x tickers)."""
    cols = {}
    missing = []
    for tk in UNIVERSE:
        df = load_close(tk, force=force)
        if df.empty:
            missing.append(tk)
            continue
        cols[tk] = df["close"]
    if missing:
        raise RuntimeError(f"tickers sem dados: {missing}")
    panel = pd.DataFrame(cols).sort_index()
    # calendario de pregao = dias em que SPY existe; ffill curto p/ os demais
    panel = panel.reindex(panel[SPY_TICKER].dropna().index)
    panel = panel.ffill(limit=5)
    # corta p/ o periodo cross-asset HONESTO (universo populado) — ver START_DATE
    panel = panel[panel.index >= pd.Timestamp(START_DATE, tz="UTC")]
    return panel


# ============================================================================
# SINAL E PESOS — sem look-ahead (.shift(1) em tudo que vira peso)
# ============================================================================
def _momentum_signal(rets: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Sinal TSMOM em {-1,0,+1} por ativo. variant escolhe lookback(s).

    Sinal no dia t usa retorno cumulativo ATE t-1 (.shift(1)) -> sem look-ahead.
    Combos = media dos sinais dos lookbacks componentes (depois sign).
    """
    def sig_for(lb_key: str) -> pd.DataFrame:
        lb = LOOKBACKS[lb_key]
        # retorno cumulativo do lookback, conhecido em t-1
        cum = (1.0 + rets).rolling(lb).apply(lambda x: np.prod(x) - 1.0, raw=True)
        return np.sign(cum.shift(1))

    if "-" in variant:
        keys = {"3-6-12": ["3m", "6m", "12m"], "1-3-6-12": ["1m", "3m", "6m", "12m"]}[variant]
        combined = sum(sig_for(k) for k in keys) / len(keys)
        return np.sign(combined)
    return sig_for(variant)


def _inv_vol(rets: pd.DataFrame) -> pd.DataFrame:
    """Peso bruto por inverso da vol (vol conhecida em t-1). Escala cada nome p/
    PER_ASSET_VOL_TARGET antes do vol-target de portfolio."""
    vol = rets.rolling(VOL_LOOKBACK).std().shift(1)
    scale = PER_ASSET_VOL_TARGET / vol.replace(0.0, np.nan)
    return scale.clip(upper=5.0)  # teto sanidade p/ vol minuscula


def build_strategy_returns(
    panel: pd.DataFrame, variant: str, cost_per_side_bps: float
) -> tuple[pd.Series, pd.Series]:
    """Retorna (ret_liquido, ret_bruto) diarios do book TSMOM p/ uma variante.

    Pipeline (tudo causal):
      1. sinal {-1,0,+1} por ativo (lookback ate t-1)
      2. dimensao por inverso-da-vol a vol-alvo por ativo (vol ate t-1)
      3. peso bruto = sinal * dimensao; rebalance a cada REBAL dias (segura entre)
      4. vol-target de PORTFOLIO: escala o book inteiro p/ PORT_VOL_TARGET (vol
         realizada do book ate t-1), cap de gross em MAX_GROSS (sem alavancar > 1)
      5. retorno bruto = sum(peso_t * ret_t); custo = per_side * sum|Delta peso|
    """
    rets = panel.pct_change()
    sig = _momentum_signal(rets, variant)
    dim = _inv_vol(rets)
    raw_w = (sig * dim).fillna(0.0)

    # rebalance: segura o peso entre rebalances (passo causal, sem peek)
    hold = raw_w.copy()
    hold[(np.arange(len(hold)) % REBAL) != 0] = np.nan
    hold = hold.ffill().fillna(0.0)

    # vol-target de portfolio sobre o book pre-escala (vol realizada ate t-1)
    pre_ret = (hold * rets).sum(axis=1)
    pre_vol = pre_ret.rolling(VOL_LOOKBACK).std().shift(1)
    target_daily = PORT_VOL_TARGET_ANNUAL / np.sqrt(TRADING_DAYS)
    pscale = (target_daily / pre_vol.replace(0.0, np.nan)).clip(upper=10.0)

    weights = hold.mul(pscale, axis=0)
    # cap de gross (L1) em MAX_GROSS -> sem financiamento
    gross = weights.abs().sum(axis=1)
    gcap = (MAX_GROSS / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(gcap, axis=0).fillna(0.0)

    gross_ret = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_per_side_bps / 1e4)
    net_ret = gross_ret - cost

    # corta o aquecimento (precisa de max lookback + vol windows)
    warmup = max(LOOKBACKS.values()) + VOL_LOOKBACK + 5
    return net_ret.iloc[warmup:], gross_ret.iloc[warmup:]


# ============================================================================
# DIAGNOSTICOS DE DIVERSIFICACAO (a tese inteira mora aqui)
# ============================================================================
def _ann_sharpe(r: np.ndarray) -> float:
    return observed_sharpe(r) * np.sqrt(TRADING_DAYS)


def _cagr(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0


def crash_analysis(strat: pd.Series, spy: pd.Series) -> dict:
    """Performance do TSMOM nos piores meses do SPY + janelas de crise nomeadas."""
    df = pd.DataFrame({"strat": strat, "spy": spy}).dropna()
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    worst_decile = monthly[monthly["spy"] <= monthly["spy"].quantile(0.10)]
    out = {
        "corr_daily": float(df["strat"].corr(df["spy"])),
        "corr_monthly": float(monthly["strat"].corr(monthly["spy"])),
        "worst_decile_spy_mean": float(worst_decile["spy"].mean()),
        "worst_decile_strat_mean": float(worst_decile["strat"].mean()),
        "worst_decile_n": int(len(worst_decile)),
    }
    # janelas de crise nomeadas
    crises = {
        "2008_GFC": ("2008-01-01", "2009-03-31"),
        "2020_COVID": ("2020-02-15", "2020-04-30"),
        "2022_bear": ("2022-01-01", "2022-10-31"),
    }
    for name, (a, b) in crises.items():
        a_ts, b_ts = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        w = df[(df.index >= a_ts) & (df.index <= b_ts)]
        if len(w) > 5:
            out[name] = {
                "strat": float((1.0 + w["strat"]).prod() - 1.0),
                "spy": float((1.0 + w["spy"]).prod() - 1.0),
            }
        else:
            out[name] = {"strat": float("nan"), "spy": float("nan")}
    return out


def combine_6040(strat: pd.Series, spy: pd.Series, tlt: pd.Series) -> dict:
    """60/40 (SPY/TLT) puro vs 60/40 com sleeve TSMOM (80% 60/40 + 20% TSMOM,
    re-normalizado). Compara Sharpe/MaxDD do CONJUNTO — e o teste do diversificador."""
    df = pd.DataFrame({"strat": strat, "spy": spy, "tlt": tlt}).dropna()
    base = 0.6 * df["spy"] + 0.4 * df["tlt"]
    blended = 0.8 * base + 0.2 * df["strat"]

    def stats(r: pd.Series) -> dict:
        eq = (1.0 + r).cumprod().to_numpy()
        return {
            "sharpe": _ann_sharpe(r.to_numpy()),
            "cagr": _cagr(r),
            "maxdd": max_drawdown(eq),
            "skew": skew(r.to_numpy()),
        }

    return {"6040": stats(base), "6040_plus_tsmom": stats(blended)}


# ============================================================================
# RUNNER
# ============================================================================
def run(force: bool = False) -> dict:
    panel = load_panel(force=force)
    spy_ret = panel[SPY_TICKER].pct_change()
    tlt_ret = panel["TLT"].pct_change()

    cost = EQUITY_BASE  # 3 bps/lado base
    cost_stress = EQUITY_BASE.stressed(2.0)  # 6 bps/lado

    # Constroi TODAS as variantes (base e estressado). n_trials = nº de variantes.
    series_net, series_gross, series_net_stress = {}, {}, {}
    for v in COMBO_VARIANTS:
        net, gross = build_strategy_returns(panel, v, cost.per_side_bps)
        net_s, _ = build_strategy_returns(panel, v, cost_stress.per_side_bps)
        series_net[v] = net
        series_gross[v] = gross
        series_net_stress[v] = net_s

    # Matriz alinhada p/ PBO (datas comuns x variantes), cenario ESTRESSADO (honesto)
    mat_df = pd.DataFrame(series_net_stress).dropna()
    pbo = probability_of_backtest_overfitting(mat_df.to_numpy(), n_splits=16)

    # Sharpes por periodo de TODAS as variantes -> dispersao p/ DSR (anti-snooping)
    trial_sharpes = [observed_sharpe(series_net_stress[v].to_numpy()) for v in COMBO_VARIANTS]
    n_trials = len(COMBO_VARIANTS) * len(["base", "stress"])  # honesto: base+stress contam

    # Variante CAMPEA = melhor Sharpe liquido ESTRESSADO (a barra usa o cenario pessimista)
    champ = max(COMBO_VARIANTS, key=lambda v: _ann_sharpe(series_net_stress[v].to_numpy()))
    champ_net = series_net[champ]
    champ_net_stress = series_net_stress[champ]

    verdict = evaluate_edge(
        champ_net_stress.to_numpy(),
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        periods_per_year=TRADING_DAYS,
        min_sharpe_annual=0.5,   # diversificador: barra de Sharpe modesta, mas exige diversificacao
        dsr_threshold=0.95,
    )

    # Diagnosticos de diversificacao na variante campea (cenario base p/ leitura)
    crash = crash_analysis(champ_net, spy_ret)
    blend = combine_6040(champ_net, spy_ret, tlt_ret)

    # SPY buy&hold no mesmo periodo da estrategia campea
    spy_aligned = spy_ret.reindex(champ_net.index).dropna()
    spy_eq = (1.0 + spy_aligned).cumprod().to_numpy()
    spy_stats = {
        "sharpe": _ann_sharpe(spy_aligned.to_numpy()),
        "cagr": _cagr(spy_aligned),
        "maxdd": max_drawdown(spy_eq),
    }

    eq = (1.0 + champ_net).cumprod().to_numpy()
    champ_stats = {
        "sharpe_net": _ann_sharpe(champ_net.to_numpy()),
        "sharpe_net_stress": _ann_sharpe(champ_net_stress.to_numpy()),
        "sharpe_gross": _ann_sharpe(series_gross[champ].to_numpy()),
        "cagr": _cagr(champ_net),
        "maxdd": max_drawdown(eq),
        "skew": skew(champ_net.to_numpy()),
        "n_obs": int(len(champ_net)),
    }

    return {
        "champ": champ,
        "verdict": verdict,
        "pbo": pbo,
        "champ_stats": champ_stats,
        "spy_stats": spy_stats,
        "crash": crash,
        "blend": blend,
        "all_sharpes": {v: _ann_sharpe(series_net_stress[v].to_numpy()) for v in COMBO_VARIANTS},
        "period": (str(champ_net.index[0].date()), str(champ_net.index[-1].date())),
        "n_trials": n_trials,
    }


def _build_report(res: dict) -> str:
    v = res["verdict"]
    cs = res["champ_stats"]
    ss = res["spy_stats"]
    cr = res["crash"]
    bl = res["blend"]

    corr = cr["corr_daily"]
    # Barra de graduacao
    beats_spy = cs["sharpe_net_stress"] >= ss["sharpe"]
    diversifies = (
        abs(corr) <= 0.3
        and bl["6040_plus_tsmom"]["sharpe"] > bl["6040"]["sharpe"]
        and bl["6040_plus_tsmom"]["maxdd"] >= bl["6040"]["maxdd"]  # menos negativo = melhor
    )
    passed = bool(v.passes_dsr and cs["sharpe_net_stress"] >= 0.5 and (beats_spy or diversifies))

    L = []
    L.append("=" * 78)
    L.append("TRIBUNAL — TSMOM cross-asset (Trend-following / Managed-futures)")
    L.append("=" * 78)
    L.append(f"Periodo: {res['period'][0]} .. {res['period'][1]}  |  n_obs={cs['n_obs']}")
    L.append(f"Universo: {len(UNIVERSE)} ETFs total-return  |  variante campea: {res['champ']}")
    L.append(f"n_trials (honesto, base+stress): {res['n_trials']}")
    L.append("")
    L.append("--- TRIBUNAL ESTATISTICO (cenario ESTRESSADO 2x custo) ---")
    L.append(v.summary())
    L.append(f"PBO (CSCV) = {res['pbo']:.3f}   (< 0.5 aceitavel; menor = mais robusto)")
    L.append("")
    L.append("--- PERFORMANCE (variante campea) ---")
    L.append(f"Sharpe gross           : {cs['sharpe_gross']:.2f}")
    L.append(f"Sharpe liq (base)      : {cs['sharpe_net']:.2f}")
    L.append(f"Sharpe liq (estress 2x): {cs['sharpe_net_stress']:.2f}")
    L.append(f"CAGR liq               : {cs['cagr']*100:.2f}%")
    L.append(f"MaxDD                  : {cs['maxdd']*100:.2f}%")
    L.append(f"Skew (diario)          : {cs['skew']:.2f}   (esperado POSITIVO p/ trend)")
    L.append("")
    L.append("--- vs SPY buy&hold (mesmo periodo) ---")
    L.append(f"SPY Sharpe={ss['sharpe']:.2f}  CAGR={ss['cagr']*100:.2f}%  MaxDD={ss['maxdd']*100:.2f}%")
    L.append(f"TSMOM bate SPY no Sharpe (estress)? {'SIM' if beats_spy else 'NAO'}")
    L.append("")
    L.append("--- DIVERSIFICACAO (a tese) ---")
    L.append(f"Corr ao SPY (diaria)   : {cr['corr_daily']:.3f}")
    L.append(f"Corr ao SPY (mensal)   : {cr['corr_monthly']:.3f}")
    L.append(f"Piores 10% meses do SPY (n={cr['worst_decile_n']}):")
    L.append(f"    SPY medio   = {cr['worst_decile_spy_mean']*100:.2f}%/mes")
    L.append(f"    TSMOM medio = {cr['worst_decile_strat_mean']*100:.2f}%/mes  (positivo = crisis alpha)")
    for name in ("2008_GFC", "2020_COVID", "2022_bear"):
        c = cr[name]
        L.append(f"  {name:12s}: TSMOM {c['strat']*100:+7.2f}%   SPY {c['spy']*100:+7.2f}%")
    L.append("")
    L.append("--- vs 60/40 (SPY/TLT) — teste do CONJUNTO ---")
    b0, b1 = bl["6040"], bl["6040_plus_tsmom"]
    L.append(f"60/40 puro           : Sharpe={b0['sharpe']:.2f} CAGR={b0['cagr']*100:.2f}% MaxDD={b0['maxdd']*100:.2f}% skew={b0['skew']:.2f}")
    L.append(f"60/40 +20% TSMOM     : Sharpe={b1['sharpe']:.2f} CAGR={b1['cagr']*100:.2f}% MaxDD={b1['maxdd']*100:.2f}% skew={b1['skew']:.2f}")
    L.append(f"Sleeve TSMOM melhora o 60/40? Sharpe {'SIM' if b1['sharpe']>b0['sharpe'] else 'NAO'}  "
             f"MaxDD {'SIM' if b1['maxdd']>=b0['maxdd'] else 'NAO'}")
    L.append("")
    L.append("--- robustez por variante (Sharpe liq estressado) ---")
    for vk, sh in res["all_sharpes"].items():
        L.append(f"    {vk:10s}: {sh:.2f}")
    L.append("")
    L.append("--- BARRA DE GRADUACAO ---")
    L.append(f"DSR >= 0.95 ?            {'SIM' if v.passes_dsr else 'NAO'} (DSR={v.dsr:.3f})")
    L.append(f"Sharpe liq estress>=0.5? {'SIM' if cs['sharpe_net_stress']>=0.5 else 'NAO'}")
    L.append(f"Bate SPY OU diversifica? {'SIM' if (beats_spy or diversifies) else 'NAO'} "
             f"(bate_spy={beats_spy}, diversifica={diversifies})")
    L.append("")
    L.append(f"VEREDITO: {'PASSA' if passed else 'REPROVADO'}")
    L.append("=" * 78)
    return "\n".join(L), passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    args = ap.parse_args()

    res = run(force=args.force)
    report, passed = _build_report(res)
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    logger.info("\nrelatorio salvo em %s", REPORT_PATH)


if __name__ == "__main__":
    main()
