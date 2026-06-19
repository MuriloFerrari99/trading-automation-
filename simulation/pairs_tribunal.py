"""Tribunal de PAIRS / COINTEGRACAO (stat-arb).

Testa pairs trading classico no MESMO tribunal estatistico do projeto (DSR/PSR/PBO,
max_drawdown, custos reais). Estrategia: num universo liquido de equities, achar pares
COINTEGRADOS (Engle-Granger / ADF sobre o residuo) numa janela de FORMACAO, operar o
SPREAD fora-da-amostra (z-score entra no desvio, sai na reversao), market-neutral
(dolar-neutro via hedge ratio beta).

ANTI-OVERFIT — os tres pecados classicos do pairs trading sao tratados:
  1. SELECAO DE PARES: a formacao (cointegracao + hedge ratio + media/desvio do spread)
     usa SO a janela de formacao; o trading roda na janela seguinte (walk-forward OOS).
     NENHUM par e selecionado usando dados do periodo em que e operado.
  2. n_trials HONESTO: o DSR conta TODOS os pares escaneados x TODAS as variantes de
     parametro (z_entry, z_exit, lookback) — nao um espantalho.
  3. PBO via CSCV sobre a matriz de configs (cada coluna = serie de retorno de uma config).
  4. DECAIMENTO POS-2000s: roda em sub-amostras por decada para ver se o edge morreu.

SEM LOOK-AHEAD: hedge ratio e mu/sigma do spread vem da formacao (passado). No trading,
z_t usa spread ate t (preco de fechamento t, conhecido no fim do dia t); a posicao
decidida em t rende o retorno de t->t+1. Custo deduzido a cada MUDANCA de posicao em
AMBAS as pernas (equities Alpaca: 0 comissao + slippage; round-trip do par = 4 lados).

Dados GRATIS: data/beta_cache/*.csv (close ajustado, ate 1962). Se faltar, baixa via
yfinance. Se a rede bloquear e nao houver cache -> reporta data_status, nao inventa.

Uso:
    uv run python -m simulation.pairs_tribunal --report-file data/pairs_verdict.txt
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import EQUITY_BASE, CostModel
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE = Path("data/beta_cache")

# Universo liquido. Agrupado por setor/tema p/ candidatos economicamente plausiveis a
# cointegracao (mesma exposicao -> spread estacionario faz sentido economico, nao so
# estatistico). So pares DENTRO do mesmo grupo sao testados -> reduz busca espuria.
GROUPS: dict[str, list[str]] = {
    "energy": ["XOM", "CVX", "COP"],
    "banks": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
    "staples": ["KO", "PEP", "PG", "WMT", "COST", "MCD", "SBUX"],
    "tech": ["MSFT", "AAPL", "ORCL", "IBM", "CSCO", "INTC", "TXN", "QCOM"],
    "health": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH"],
    "telecom": ["T", "VZ"],
    "metals": ["GLD", "SLV"],
    "indices": ["SPY", "QQQ"],
    "industrial": ["CAT", "GE", "HON", "BA"],
}

# Grade de parametros (TUDO conta no n_trials honesto).
Z_ENTRY = [1.5, 2.0, 2.5]
Z_EXIT = [0.0, 0.5]
LOOKBACK = [20, 40, 60]  # janela do rolling z-score no trading (dias)
FORMATION_DAYS = 252  # 1 ano de formacao
TRADE_DAYS = 126       # 6 meses de trading OOS antes de re-formar
ADF_PVALUE_MAX = 0.05  # so opera pares cuja cointegracao IS passa a 5%
MAX_HOLD = 60          # forca saida do trade apos N dias (anti-spread-quebrado)


def _load_close(ticker: str) -> pd.Series | None:
    f = CACHE / f"{ticker}_1d.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # beta_cache: colunas Date,close
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    close_col = "close" if "close" in df.columns else df.columns[-1]
    s = pd.Series(
        df[close_col].astype(float).values,
        index=pd.to_datetime(df[date_col], utc=True),
        name=ticker,
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def load_panel(tickers: list[str]) -> pd.DataFrame:
    series = {}
    missing = []
    for t in tickers:
        s = _load_close(t)
        if s is None or s.size < FORMATION_DAYS + TRADE_DAYS:
            missing.append(t)
            continue
        series[t] = s
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series)
    panel.attrs["missing"] = missing
    return panel


# --------------------------------------------------------------------------------------
# ADF test (self-contained; sem statsmodels). Regressao de Dickey-Fuller aumentada com
# constante. p-valor por interpolacao das tabelas de MacKinnon (1994/2010) para o caso
# "com constante". Usado p/ testar estacionariedade do RESIDUO (Engle-Granger 2o passo).
# --------------------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS: retorna (coefs, residuos). X ja deve incluir constante se desejada."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def adf_tstat(x: np.ndarray, max_lag: int = 1) -> float:
    """t-stat do ADF (com constante) sobre a serie x. Mais negativo = mais estacionario."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    n = dx.size
    if n <= max_lag + 5:
        return 0.0
    # y = dx[t]; regressores: x[t-1], const, lags de dx
    y = dx[max_lag:]
    lvl = x[max_lag:-1]  # x_{t-1} alinhado a y
    cols = [lvl, np.ones_like(lvl)]
    for L in range(1, max_lag + 1):
        cols.append(dx[max_lag - L : -L])
    X = np.column_stack(cols)
    beta, resid = _ols(y, X)
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return 0.0
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_gamma = float(np.sqrt(s2 * XtX_inv[0, 0]))
    if se_gamma == 0:
        return 0.0
    return float(beta[0] / se_gamma)


# Tabela de p-valor de MacKinnon para ADF com constante (modelo "c"), aproximada.
# Mapeia t-stat -> p-valor por interpolacao linear em pontos criticos publicados.
_ADF_C = [
    (-3.43, 0.01), (-3.12, 0.025), (-2.86, 0.05), (-2.57, 0.10),
    (-2.20, 0.25), (-1.62, 0.50), (-1.00, 0.75), (-0.44, 0.90),
]


def adf_pvalue(x: np.ndarray, max_lag: int = 1) -> float:
    t = adf_tstat(x, max_lag=max_lag)
    pts = _ADF_C
    if t <= pts[0][0]:
        return 0.005
    if t >= pts[-1][0]:
        return 0.99
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0)
            return float(p0 + w * (p1 - p0))
    return 0.99


# --------------------------------------------------------------------------------------
# Engle-Granger: formacao
# --------------------------------------------------------------------------------------
@dataclass
class Formation:
    a: str
    b: str
    beta: float       # hedge ratio: spread = log(a) - beta*log(b) - alpha
    alpha: float
    mu: float         # media do spread na formacao
    sigma: float      # desvio do spread na formacao
    adf_p: float      # p-valor da cointegracao IS


def form_pair(la: np.ndarray, lb: np.ndarray, a: str, b: str) -> Formation | None:
    """Engle-Granger sobre logs. Regride log(a)~log(b)+const, testa ADF do residuo."""
    X = np.column_stack([lb, np.ones_like(lb)])
    coef, resid = _ols(la, X)
    beta, alpha = float(coef[0]), float(coef[1])
    if beta <= 0:  # hedge negativo nao faz sentido p/ duas acoes do mesmo setor
        return None
    p = adf_pvalue(resid)
    return Formation(
        a=a, b=b, beta=beta, alpha=alpha,
        mu=float(resid.mean()), sigma=float(resid.std(ddof=1)), adf_p=p,
    )


# --------------------------------------------------------------------------------------
# Backtest de UM par numa janela de trading, dado o spread realizado.
# Retorna a serie de retorno DIARIO do par (liquido) sobre a janela de trading.
# --------------------------------------------------------------------------------------
def trade_pair(
    la: np.ndarray, lb: np.ndarray, ra: np.ndarray, rb: np.ndarray,
    form: Formation, z_entry: float, z_exit: float, lookback: int, cost: CostModel,
) -> np.ndarray:
    """la/lb: log-precos na janela trading (incl. 1 dia anterior p/ retorno). ra/rb:
    retornos simples diarios alinhados (t->t+1). Tudo mesmo comprimento T.

    Posicao em t (decidida com info <= t) rende o retorno t->t+1. Spread = la - beta*lb.
    Sinal: z = (spread - mu_roll)/sigma_roll com janela `lookback` (rolling, causal);
    se nao houver historico suficiente, usa mu/sigma da formacao.

    Pesos dolar-neutro: long spread => +1 em A, -beta em B (normalizado p/ bruto=1).
    Custo por LADO em bps deduzido a cada mudanca de |peso| em cada perna.
    """
    T = la.size
    spread = la - form.beta * lb
    pos = np.zeros(T)  # -1 short spread, 0 flat, +1 long spread
    state = 0
    hold = 0
    rs = pd.Series(spread)
    roll_mu = rs.rolling(lookback).mean().to_numpy()
    roll_sd = rs.rolling(lookback).std(ddof=1).to_numpy()
    for t in range(T):
        mu = roll_mu[t] if not np.isnan(roll_mu[t]) else form.mu
        sd = roll_sd[t] if (not np.isnan(roll_sd[t]) and roll_sd[t] > 0) else form.sigma
        if sd <= 0:
            pos[t] = state
            continue
        z = (spread[t] - mu) / sd
        if state == 0:
            if z >= z_entry:
                state = -1; hold = 0      # spread alto -> short spread (short A, long B)
            elif z <= -z_entry:
                state = 1; hold = 0       # spread baixo -> long spread (long A, short B)
        else:
            hold += 1
            crossed = (state == 1 and z >= -z_exit) or (state == -1 and z <= z_exit)
            if crossed or hold >= MAX_HOLD:
                state = 0
        pos[t] = state

    # peso dolar-neutro normalizado: bruto (|wa|+|wb|) = 1
    gross = 1.0 + form.beta
    wa = pos / gross
    wb = -pos * form.beta / gross
    # retorno do par em t->t+1: w_t . r_{t+1}. ra/rb sao retornos t->t+1 (ja shiftados).
    pnl = wa * ra + wb * rb
    # custo: mudanca de peso em cada perna * per_side_bps
    dwa = np.abs(np.diff(np.concatenate([[0.0], wa])))
    dwb = np.abs(np.diff(np.concatenate([[0.0], wb])))
    turn_cost = (dwa + dwb) * (cost.per_side_bps / 1e4)
    return pnl - turn_cost


# --------------------------------------------------------------------------------------
# Walk-forward sobre o panel inteiro, para UMA config (z_entry, z_exit, lookback).
# Retorna serie de retorno diario do PORTFOLIO (equal-weight entre pares ativos) + stats.
# --------------------------------------------------------------------------------------
@dataclass
class ConfigResult:
    z_entry: float
    z_exit: float
    lookback: int
    daily: pd.Series           # retorno diario do portfolio de pares
    n_pairs_traded: int


def candidate_pairs() -> list[tuple[str, str]]:
    pairs = []
    for grp in GROUPS.values():
        pairs.extend(itertools.combinations(grp, 2))
    return pairs


def run_config(
    panel: pd.DataFrame, z_entry: float, z_exit: float, lookback: int, cost: CostModel,
) -> ConfigResult:
    log_px = np.log(panel)
    rets = panel.pct_change()
    dates = panel.index
    N = len(dates)
    # acumula retorno por par em cada data; depois normaliza por #pares ativos no dia
    pair_pnl = pd.DataFrame(0.0, index=dates, columns=["_dummy"])
    pair_active = pd.Series(0, index=dates)
    pairs = candidate_pairs()
    traded = set()

    start = FORMATION_DAYS
    while start + TRADE_DAYS <= N:
        f0, f1 = start - FORMATION_DAYS, start          # formacao [f0,f1)
        t0, t1 = start, start + TRADE_DAYS              # trading [t0,t1)
        for a, b in pairs:
            la_f = log_px[a].iloc[f0:f1].to_numpy()
            lb_f = log_px[b].iloc[f0:f1].to_numpy()
            if np.isnan(la_f).any() or np.isnan(lb_f).any():
                continue
            form = form_pair(la_f, lb_f, a, b)
            if form is None or form.adf_p > ADF_PVALUE_MAX:
                continue
            # janela trading: precisa de log px e retornos t->t+1 alinhados
            la_t = log_px[a].iloc[t0:t1].to_numpy()
            lb_t = log_px[b].iloc[t0:t1].to_numpy()
            ra_t = rets[a].iloc[t0:t1].shift(-1).to_numpy()  # retorno t->t+1
            rb_t = rets[b].iloc[t0:t1].shift(-1).to_numpy()
            if np.isnan(la_t).any() or np.isnan(lb_t).any():
                continue
            ra_t = np.nan_to_num(ra_t)
            rb_t = np.nan_to_num(rb_t)
            pnl = trade_pair(la_t, lb_t, ra_t, rb_t, form, z_entry, z_exit, lookback, cost)
            idx = dates[t0:t1]
            key = f"{a}-{b}"
            if key not in pair_pnl.columns:
                pair_pnl[key] = 0.0
            pair_pnl.loc[idx, key] = pair_pnl.loc[idx, key].to_numpy() + pnl
            active_mask = np.abs(pnl) > 0
            # marca dias em que o par estava posicionado (aprox: pnl != 0) p/ normalizar
            pair_active.loc[idx] = pair_active.loc[idx].to_numpy() + active_mask.astype(int)
            traded.add(key)
        start += TRADE_DAYS

    if "_dummy" in pair_pnl.columns:
        pair_pnl = pair_pnl.drop(columns=["_dummy"])
    if pair_pnl.shape[1] == 0:
        return ConfigResult(z_entry, z_exit, lookback, pd.Series(dtype=float), 0)
    # portfolio: media dos pnls dos pares (equal-weight entre TODOS os pares formados).
    # divide pelo numero de pares-config para alocar capital (capital nao alavancado).
    n_pair_slots = max(pair_pnl.shape[1], 1)
    daily = pair_pnl.sum(axis=1) / n_pair_slots
    daily = daily[daily.index >= dates[FORMATION_DAYS]]
    return ConfigResult(z_entry, z_exit, lookback, daily, len(traded))


@dataclass
class Verdict:
    data_status: str
    passed: bool
    verdict: str
    sharpe: float = 0.0
    cagr: float = 0.0
    maxdd: float = 0.0
    dsr: float = 0.0
    pbo: float = 0.0
    corr_to_market: float = 0.0
    n_trials: int = 0
    detail: list[str] = field(default_factory=list)


def main(report_file: str | None = None) -> Verdict:
    all_tickers = sorted({t for g in GROUPS.values() for t in g})
    panel = load_panel(all_tickers)
    lines: list[str] = []

    def log(s: str = "") -> None:
        lines.append(s)

    log("=" * 78)
    log("TRIBUNAL — PAIRS / COINTEGRACAO (stat-arb)")
    log("=" * 78)

    if panel.empty:
        v = Verdict(
            data_status="blocked", passed=False,
            verdict="Sem dados: data/beta_cache vazio e sem rede. Rode o fetch de precos.",
        )
        log(v.verdict)
        if report_file:
            Path(report_file).write_text("\n".join(lines))
        return v

    missing = panel.attrs.get("missing", [])
    panel = panel.dropna(how="all")
    # alinha no overlap comum (intersecao das datas com dado em todos? nao — pairs sao
    # intra-grupo; mantemos union e tratamos NaN por par na formacao). Usamos panel cru.
    log(f"Universo: {panel.shape[1]} tickers | barras={panel.shape[0]} "
        f"| {panel.index[0].date()} -> {panel.index[-1].date()}")
    if missing:
        log(f"Faltando (cache curto/ausente): {missing}")

    pairs = candidate_pairs()
    n_pairs = len(pairs)
    cost = EQUITY_BASE.stressed(2.0)  # cenario ESTRESSADO p/ honestidade (2x slippage)
    log(f"Pares candidatos (intra-grupo): {n_pairs} | custo={cost.name} "
        f"({cost.per_side_bps:.1f} bps/lado)")

    # n_trials HONESTO: pares x grade de params (cointegracao filtra, mas a busca existiu)
    n_param = len(Z_ENTRY) * len(Z_EXIT) * len(LOOKBACK)
    n_trials = n_pairs * n_param
    log(f"n_trials honesto = {n_pairs} pares x {n_param} params = {n_trials}")
    log("")

    # roda todas as configs (grade de params) -> matriz p/ PBO + escolhe melhor p/ DSR
    configs: list[ConfigResult] = []
    log("Rodando grade de configs (walk-forward OOS, formacao 252d / trade 126d)...")
    for ze, zx, lb in itertools.product(Z_ENTRY, Z_EXIT, LOOKBACK):
        cr = run_config(panel, ze, zx, lb, cost)
        if cr.daily.size > 10:
            configs.append(cr)

    if not configs:
        v = Verdict(
            data_status="limited", passed=False,
            verdict="Nenhuma config gerou trades suficientes (poucos pares cointegrados OOS).",
            n_trials=n_trials,
        )
        log(v.verdict)
        if report_file:
            Path(report_file).write_text("\n".join(lines))
        return v

    # matriz alinhada p/ PBO (datas comuns a todas as configs)
    common = configs[0].daily.index
    for cr in configs[1:]:
        common = common.intersection(cr.daily.index)
    mat = np.column_stack([cr.daily.reindex(common).fillna(0.0).to_numpy() for cr in configs])
    pbo = probability_of_backtest_overfitting(mat, n_splits=16)

    # escolhe a melhor config por Sharpe IS (consciente: e isso que o overfitter faria;
    # o DSR penaliza pela busca e o PBO mede o overfit dessa escolha)
    sharpes = [observed_sharpe(cr.daily.to_numpy(), EQUITY_PERIODS) for cr in configs]
    best_i = int(np.argmax(sharpes))
    best = configs[best_i]
    daily = best.daily

    log(f"Configs com trades: {len(configs)} | melhor: z_entry={best.z_entry} "
        f"z_exit={best.z_exit} lookback={best.lookback} | pares operados={best.n_pairs_traded}")
    log("")

    # tribunal estatistico (n_trials honesto)
    trial_sharpes_period = [observed_sharpe(cr.daily.to_numpy()) for cr in configs]
    edge = evaluate_edge(
        daily.to_numpy(),
        n_trials=n_trials,
        trial_sharpes=trial_sharpes_period,
        periods_per_year=EQUITY_PERIODS,
        min_sharpe_annual=0.8,
    )

    equity = (1.0 + daily).cumprod().to_numpy()
    mdd = max_drawdown(equity)
    years = daily.size / EQUITY_PERIODS
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 and years > 0 else -1.0

    # correlacao com o mercado (SPY). Espera-se ~0 (market-neutral).
    spy = _load_close("SPY")
    corr_mkt = 0.0
    if spy is not None:
        spy_ret = spy.pct_change().reindex(daily.index).fillna(0.0)
        if spy_ret.std() > 0 and daily.std() > 0:
            corr_mkt = float(np.corrcoef(daily.to_numpy(), spy_ret.to_numpy())[0, 1])

    # SPY buy&hold no mesmo periodo p/ comparar
    spy_sharpe = 0.0
    if spy is not None:
        spy_r = spy.pct_change().reindex(daily.index).dropna()
        spy_sharpe = observed_sharpe(spy_r.to_numpy(), EQUITY_PERIODS)

    log("-" * 78)
    log("RESULTADO (config vencedora, cenario de custo ESTRESSADO 2x):")
    log(edge.summary())
    log(f"  Sharpe_anual_liq = {edge.sharpe_annual:.2f}  (SPY buy&hold mesmo periodo = {spy_sharpe:.2f})")
    log(f"  CAGR = {cagr*100:.2f}%   MaxDD = {mdd*100:.2f}%")
    log(f"  DSR = {edge.dsr:.3f}   PSR = {edge.psr:.3f}   PBO = {pbo:.3f}")
    log(f"  corr vs SPY = {corr_mkt:.3f}  (esperado ~0, market-neutral)")
    log(f"  n_obs={edge.n_obs}  n_trials={edge.n_trials}")
    log("")

    # decaimento por decada (sub-amostras)
    log("DECAIMENTO POR PERIODO (Sharpe anual liq da config vencedora):")
    for lo, hi in [("1990", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2099")]:
        sub = daily[(daily.index.year >= int(lo)) & (daily.index.year <= int(hi))]
        if sub.size > 60:
            log(f"  {lo[:4]}s: Sharpe={observed_sharpe(sub.to_numpy(), EQUITY_PERIODS):6.2f}  n={sub.size}")
    log("")

    # BARRA: DSR>=0.95 E Sharpe robusto E (bate buy&hold OU diversifica c/ corr baixa)
    diversifies = abs(corr_mkt) < 0.3 and edge.sharpe_annual > 0.5
    beats_bh = edge.sharpe_annual > spy_sharpe
    passes_bar = (
        edge.dsr >= 0.95
        and edge.sharpe_annual >= 0.8
        and pbo < 0.5
        and (beats_bh or diversifies)
    )

    if passes_bar:
        verdict_txt = (
            f"PASSA: DSR={edge.dsr:.3f}, Sharpe_liq={edge.sharpe_annual:.2f}, "
            f"PBO={pbo:.3f}, corr_SPY={corr_mkt:.2f}. Edge stat-arb robusto OOS."
        )
    else:
        reasons = []
        if edge.dsr < 0.95:
            reasons.append(f"DSR={edge.dsr:.3f}<0.95 (Sharpe nao sobrevive ao data-snooping de {n_trials} trials)")
        if edge.sharpe_annual < 0.8:
            reasons.append(f"Sharpe_liq={edge.sharpe_annual:.2f}<0.8")
        if pbo >= 0.5:
            reasons.append(f"PBO={pbo:.3f}>=0.5 (selecao escolhe sorte)")
        if not (beats_bh or diversifies):
            reasons.append(f"nao bate SPY ({edge.sharpe_annual:.2f} vs {spy_sharpe:.2f}) nem diversifica (corr={corr_mkt:.2f})")
        verdict_txt = "REPROVA: " + "; ".join(reasons)

    log("=" * 78)
    log(("VEREDITO: PASSA" if passes_bar else "VEREDITO: REPROVA"))
    log(verdict_txt)
    log("=" * 78)

    if report_file:
        Path(report_file).write_text("\n".join(lines))

    print("\n".join(lines))
    return Verdict(
        data_status="real",
        passed=passes_bar,
        verdict=verdict_txt,
        sharpe=round(edge.sharpe_annual, 4),
        cagr=round(cagr, 4),
        maxdd=round(mdd, 4),
        dsr=round(edge.dsr, 4),
        pbo=round(pbo, 4),
        corr_to_market=round(corr_mkt, 4),
        n_trials=n_trials,
        detail=lines,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-file", default="data/pairs_verdict.txt")
    args = ap.parse_args()
    main(report_file=args.report_file)
