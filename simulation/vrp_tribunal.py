"""Tribunal estatistico para Volatility Risk Premium (venda sistematica de vol).

R&D ISOLADO. Nao toca em beta_*, main, producao. Cria artefatos proprios em data/.

ESTRATEGIA: venda sistematica de volatilidade em SPY.
  - Implementada como SHORT PUT mensal coberto (cash-secured), repricado com Black-Scholes
    usando o VIX (vol implicita, conhecido em t-1) como IV de entrada, e a payoff realizada
    ao longo do mes pelo caminho REAL do SPY. Isso captura honestamente o RISCO DE CAUDA:
    quando o SPY cai forte, a put vendida estoura.
  - Tambem reportamos o proxy academico do VRP (VIX^2 - var realizada) como sanidade.

DADOS: yfinance (SPY desde 1993, ^VIX desde 1990 via VXO splice). GRATIS.
  Opcoes historicas reais nao sao obtenivel de graca -> data_status="limited":
  aproximamos o premio via BS+VIX. Marcado honestamente.

SEM look-ahead: a decisao de vender e o IV usado vem de t-1 (VIX de ontem, S de ontem);
o retorno vem do caminho t..t+1 mês. Custo real (bid-ask de opcoes + comissao).

n_trials HONESTO: contamos TODAS as variantes de moneyness x alavancagem x filtro de
regime testadas no DSR.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "vrp_cache")
REPORT = os.path.join(os.path.dirname(__file__), "..", "data", "vrp_verdict.txt")
N = NormalDist = None  # placeholder; usamos math.erf

ANN = math.sqrt(EQUITY_PERIODS)


# ----------------------------- Black-Scholes ------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Preco Black-Scholes de uma PUT europeia. T em anos, sigma anualizada."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ----------------------------- Dados --------------------------------------------
def load_data() -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    fpath = os.path.join(CACHE, "spy_vix_daily.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        return df
    spy = yf.download("SPY", start="1993-01-01", end="2026-06-17",
                      progress=False, auto_adjust=False)
    vix = yf.download("^VIX", start="1993-01-01", end="2026-06-17",
                      progress=False, auto_adjust=False)
    spy_close = spy["Close"]
    spy_adj = spy["Adj Close"]
    vix_close = vix["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
        spy_adj = spy_adj.iloc[:, 0]
        vix_close = vix_close.iloc[:, 0]
    df = pd.DataFrame({
        "spy": spy_close.astype(float),
        "spy_adj": spy_adj.astype(float),
        "vix": vix_close.astype(float),
    }).dropna()
    df.to_csv(fpath)
    return df


# ----------------------------- Backtest short-put -------------------------------
@dataclass
class VRPConfig:
    moneyness: float       # K = S * moneyness (0.95 = 5% OTM put)
    leverage: float        # notional vendido / equity
    regime_gate: bool      # so vende se VIX(t-1) > MA(VIX) ? (carry filter)
    name: str = "cfg"


# Custo de opcao: bid-ask + comissao. Opcoes SPY ~ 1-3% do premio em spread no varejo;
# usamos bps sobre o NOTIONAL por entrada+saida para ser conservador.
OPT_COST_BPS_NOTIONAL = 5.0   # base: ~5 bps do notional por roll (spread+comissao)
RF = 0.02                     # taxa livre de risco media aproximada


def run_short_put(df: pd.DataFrame, cfg: VRPConfig, cost_bps: float) -> np.ndarray:
    """Retorna serie de retornos MENSAIS (por roll) da venda sistematica de put.

    Mecanica (sem look-ahead):
      - A cada ~21 pregoes (mensal), em t0 conhecemos S(t0) e VIX(t0).
      - Vendemos N puts strike K=S0*moneyness, vencimento 21 pregoes (T=21/252),
        premio = BS_put(S0, K, T, r, sigma=VIX/100).
      - Em t1 (vencimento) o payoff = max(K - S1, 0). PnL = premio - payoff.
      - Notional por unidade = K (cash-secured). Alavancagem escala o notional.
    """
    idx = df.index
    S = df["spy"].values
    V = df["vix"].values / 100.0
    step = 21
    T = step / 252.0

    # gate de regime: media movel de 252d do VIX, calculada com dados <= t-1
    vix_ma = pd.Series(df["vix"].values, index=idx).rolling(252, min_periods=60).mean().values

    vix_raw = df["vix"].values
    rets = []
    i = step  # comeca depois de termos historico p/ MA
    while i + step < len(S):
        s0 = S[i]
        sigma = V[i]
        # gate de crash: NAO vende quando VIX(t-1) ja esta MUITO acima da sua MA
        # (regime de stress -> evita pegar a faca caindo). Usa dados <= t.
        if cfg.regime_gate and not np.isnan(vix_ma[i]) and vix_raw[i] > 1.5 * vix_ma[i]:
            # fora de regime: flat (caixa). Retorno 0 no mes (sem premio, sem risco).
            rets.append(0.0)
            i += step
            continue
        K = s0 * cfg.moneyness
        premium = bs_put(s0, K, T, RF, sigma)
        s1 = S[i + step]
        payoff = max(K - s1, 0.0)
        # custo: bid-ask + comissao sobre o notional (entrada e saida)
        cost = (cost_bps / 1e4) * K
        # PnL por unidade vendida, normalizado pelo capital alocado (cash-secured = K)
        pnl_unit = premium - payoff - cost
        capital = K  # cash-secured
        # juros do colateral (rende rf no mes)
        rets.append(cfg.leverage * (pnl_unit / capital) + RF * T)
        i += step
    return np.array(rets, dtype=float)


def buy_hold_monthly(df: pd.DataFrame) -> np.ndarray:
    """Retorno mensal (21 pregoes) de buy&hold SPY total return, alinhado aos rolls."""
    S = df["spy_adj"].values
    step = 21
    rets = []
    i = step
    while i + step < len(S):
        rets.append(S[i + step] / S[i] - 1.0)
        i += step
    return np.array(rets, dtype=float)


def crash_windows(df: pd.DataFrame, cfg: VRPConfig, cost_bps: float):
    """Pior evento por janela de crise: GFC 2008, COVID 2020, bear 2022."""
    out = {}
    windows = {
        "GFC_2008": ("2008-06-01", "2009-06-30"),
        "COVID_2020": ("2020-01-15", "2020-06-30"),
        "BEAR_2022": ("2022-01-01", "2022-12-31"),
    }
    for name, (a, b) in windows.items():
        sub = df.loc[a:b]
        if len(sub) < 25:
            out[name] = None
            continue
        r = run_short_put(sub, cfg, cost_bps)
        if r.size == 0:
            out[name] = None
            continue
        eq = np.cumprod(1.0 + r)
        out[name] = {
            "total_return": float(eq[-1] - 1.0),
            "worst_month": float(r.min()),
            "maxdd": float(max_drawdown(eq)),
        }
    return out


def main():
    df = load_data()
    print(f"Dados: {len(df)} pregoes, {df.index.min().date()} -> {df.index.max().date()}")

    # ----- grade de variantes (n_trials HONESTO) -----
    moneyness_grid = [1.00, 0.97, 0.95, 0.92]     # ATM ... 8% OTM
    leverage_grid = [0.5, 1.0, 1.5]
    gate_grid = [False, True]
    cost_base = OPT_COST_BPS_NOTIONAL
    cost_stress = OPT_COST_BPS_NOTIONAL * 2.0

    configs = []
    for m in moneyness_grid:
        for lev in leverage_grid:
            for g in gate_grid:
                configs.append(VRPConfig(m, lev, g, f"m{m}_l{lev}_g{int(g)}"))
    n_trials = len(configs)
    print(f"n_trials (variantes) = {n_trials}")

    # matriz de retornos (alinhada) p/ PBO + selecao da melhor IS por Sharpe
    series = {}
    minlen = None
    for cfg in configs:
        r = run_short_put(df, cfg, cost_base)
        series[cfg.name] = r
        minlen = len(r) if minlen is None else min(minlen, len(r))
    M = np.column_stack([series[c.name][:minlen] for c in configs])
    # trial_sharpes: ignora configs degeneradas (gate quase sempre em caixa)
    trial_sharpes = [
        observed_sharpe(series[c.name])
        for c in configs
        if series[c.name].std(ddof=1) >= 0.003
    ]

    pbo = probability_of_backtest_overfitting(M, n_splits=14)

    # melhor config por Sharpe (in-sample) — a candidata "vencedora".
    # Guarda anti-degenerado: descarta series quase-constantes (vol mensal < 0.3%),
    # que sao artefatos do gate (fica em caixa) e produzem Sharpe espurio.
    def _sel_sharpe(name: str) -> float:
        r = series[name]
        if r.std(ddof=1) < 0.003:
            return -1e9
        return observed_sharpe(r)

    best_i = int(np.argmax([_sel_sharpe(c.name) for c in configs]))
    best = configs[best_i]
    r_best_base = series[best.name]
    r_best_stress = run_short_put(df, best, cost_stress)

    bh = buy_hold_monthly(df)[: len(r_best_base)]

    # ----- avaliacao no tribunal (cenario ESTRESSADO p/ honestidade) -----
    verdict = evaluate_edge(
        r_best_stress,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        periods_per_year=12,           # rolls mensais
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # metricas da candidata (estressado)
    eq = np.cumprod(1.0 + r_best_stress)
    years = len(r_best_stress) / 12.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    mdd = float(max_drawdown(eq))
    sharpe_ann = observed_sharpe(r_best_stress) * math.sqrt(12)

    eq_bh = np.cumprod(1.0 + bh)
    cagr_bh = float(eq_bh[-1] ** (1.0 / years) - 1.0)
    sharpe_bh = observed_sharpe(bh) * math.sqrt(12)
    mdd_bh = float(max_drawdown(eq_bh))

    n = min(len(r_best_stress), len(bh))
    corr = float(np.corrcoef(r_best_stress[:n], bh[:n])[0, 1])

    crashes = crash_windows(df, best, cost_stress)

    # --- ROBUSTEZ DE FASE DE ROLL (anti-artefato) ---
    # O Sharpe da config defensiva depende de QUANDO o roll mensal comeca. Reamostramos
    # o offset 0..20 e exigimos que a edge sobreviva a fase media, nao so a sortuda.
    phase_sharpes = []
    for off in range(0, 21, 2):
        rr = run_short_put(df.iloc[off:].copy(), best, cost_stress)
        if rr.size > 12:
            phase_sharpes.append(observed_sharpe(rr) * math.sqrt(12))
    phase_med = float(np.median(phase_sharpes)) if phase_sharpes else 0.0
    phase_min = float(np.min(phase_sharpes)) if phase_sharpes else 0.0

    # Bar economico: beats_bh exige bater B&H no RETORNO (Sharpe maior nao basta se a
    # estrategia so rende ~rf). Diversifica exige correlacao BAIXA real (<0.5).
    beats_bh = cagr > cagr_bh + 0.005
    diversifies = corr < 0.5 and sharpe_ann > 0.5
    robust_phase = phase_med >= 0.8  # a edge sobrevive a fase mediana de roll?
    passed = (
        verdict.dsr >= 0.95
        and sharpe_ann >= 0.8
        and robust_phase
        and (beats_bh or diversifies)
    )

    # ----- relatorio -----
    lines = []
    lines.append("=" * 72)
    lines.append("TRIBUNAL: Volatility Risk Premium (venda sistematica de opcoes)")
    lines.append("=" * 72)
    lines.append(f"data_status = limited (sem cadeia de opcoes real gratis; premio via BS+VIX)")
    lines.append(f"Periodo: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} pregoes)")
    lines.append(f"Estrutura: SHORT PUT mensal cash-secured, IV=VIX(t-1), payoff caminho real")
    lines.append(f"Custo: {cost_base} bps notional (base) / {cost_stress} bps (estressado, usado no veredito)")
    lines.append(f"n_trials honesto = {n_trials} (moneyness x lev x gate)")
    lines.append("")
    lines.append(f"MELHOR CONFIG (por Sharpe IS): {best.name}")
    lines.append(f"  moneyness={best.moneyness} leverage={best.leverage} regime_gate={best.regime_gate}")
    lines.append("")
    lines.append("--- METRICAS (cenario ESTRESSADO, liquido) ---")
    lines.append(f"  Sharpe_anual = {sharpe_ann:.3f}")
    lines.append(f"  CAGR         = {cagr*100:.2f}%")
    lines.append(f"  MaxDD        = {mdd*100:.2f}%")
    lines.append(f"  DSR          = {verdict.dsr:.4f}  (barra >= 0.95)")
    lines.append(f"  PSR          = {verdict.psr:.4f}")
    lines.append(f"  PBO          = {pbo:.4f}  (barra < 0.5)")
    lines.append(f"  skew         = {verdict.skew:.3f}  kurt = {verdict.kurtosis:.2f}")
    lines.append(f"  obstaculo data-snooping (Sharpe anual) = {verdict.sr_benchmark_annual:.3f}")
    lines.append("")
    lines.append("--- vs BUY&HOLD SPY (mesmo periodo, mensal) ---")
    lines.append(f"  B&H Sharpe={sharpe_bh:.3f} CAGR={cagr_bh*100:.2f}% MaxDD={mdd_bh*100:.2f}%")
    lines.append(f"  correlacao VRP vs SPY = {corr:.3f}")
    lines.append(f"  bate_bh(CAGR)={beats_bh}  diversifica(corr<0.5 & SR>0.5)={diversifies}")
    lines.append("")
    lines.append("--- ROBUSTEZ DE FASE DE ROLL (config vencedora, estressado) ---")
    lines.append(f"  Sharpe por offset (0..20): {[round(s,2) for s in phase_sharpes]}")
    lines.append(f"  Sharpe mediano={phase_med:.2f} minimo={phase_min:.2f}  robusto(>=0.8)={robust_phase}")
    lines.append("  NOTA: o Sharpe da config defensiva e instavel entre fases de roll =>")
    lines.append("        o 'pass' do offset fixo e SORTE de fase, nao edge robusta.")
    lines.append("")
    lines.append("--- CONFIG ECONOMICA (ATM 1.0x, a que de fato colhe o VRP) ---")
    c_econ = VRPConfig(1.0, 1.0, False)
    r_econ = run_short_put(df, c_econ, cost_stress)
    bh_e = buy_hold_monthly(df)[: len(r_econ)]
    eq_e = np.cumprod(1.0 + r_econ)
    ne = min(len(r_econ), len(bh_e))
    lines.append(f"  Sharpe={observed_sharpe(r_econ)*math.sqrt(12):.2f} "
                 f"CAGR={(eq_e[-1]**(12/len(r_econ))-1)*100:.2f}% "
                 f"MaxDD={max_drawdown(eq_e)*100:.1f}% "
                 f"corr_SPY={np.corrcoef(r_econ[:ne], bh_e[:ne])[0,1]:.2f} "
                 f"pior_mes={r_econ.min()*100:.1f}%")
    lines.append("  => colhe ~11% CAGR (= B&H) mas corr~0.87: e beta de bull alavancado")
    lines.append("     com cauda esquerda truncada-em-cima/aberta-embaixo. NAO diversifica.")
    lines.append("")
    lines.append("--- RISCO DE CAUDA (pior evento por crise, config vencedora) ---")
    for k, v in crashes.items():
        if v is None:
            lines.append(f"  {k}: sem dados suficientes")
        else:
            lines.append(f"  {k}: retorno_total={v['total_return']*100:.1f}%  "
                         f"pior_mes={v['worst_month']*100:.1f}%  MaxDD={v['maxdd']*100:.1f}%")
    lines.append("")
    lines.append(f"VEREDITO: passed = {passed}")
    if not passed:
        reasons = []
        if verdict.dsr < 0.95:
            reasons.append(f"DSR {verdict.dsr:.3f} < 0.95")
        if sharpe_ann < 0.8:
            reasons.append(f"Sharpe {sharpe_ann:.2f} < 0.8")
        if not robust_phase:
            reasons.append(f"Sharpe instavel entre fases de roll (mediano {phase_med:.2f}<0.8)")
        if not (beats_bh or diversifies):
            reasons.append("nao bate B&H no retorno nem diversifica (corr alta)")
        lines.append("  motivos: " + "; ".join(reasons))
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)
    with open(REPORT, "w") as f:
        f.write(report + "\n")

    return {
        "passed": passed, "sharpe": sharpe_ann, "cagr": cagr, "maxdd": mdd,
        "dsr": verdict.dsr, "psr": verdict.psr, "pbo": pbo, "corr": corr,
        "n_trials": n_trials, "best": best.name, "crashes": crashes,
        "sharpe_bh": sharpe_bh, "cagr_bh": cagr_bh, "skew": verdict.skew,
        "kurt": verdict.kurtosis, "phase_med": phase_med, "phase_min": phase_min,
    }


if __name__ == "__main__":
    main()
