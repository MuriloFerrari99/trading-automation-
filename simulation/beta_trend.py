"""Beta trend — um overlay de TREND-FOLLOWING (managed futures / TSMOM) AGREGA
valor risco-ajustado ao beta de producao, de forma ROBUSTA? Cabeca-a-cabeca.

CONTEXTO: o beta de producao (auditado e ao vivo) e o VOL-TARGET multi-ativo
(SPY,QQQ,TLT,IEF,GLD,SLV,BTC,ETH; peso ~1/vol por ativo, escala p/ vol-alvo,
teto de cripto) ALAVANCADO ~2.0x com financiamento real, LONG-only. Ponto fraco
medido: drawdown (~-36% ao vivo). Tese do trend-following (TSMOM, Moskowitz-Ooi-
Pedersen 2012): premio documentado e persistente; e DIVERSIFICANTE ao long-bias
porque des-arrisca/shorta em bear sustentado -> "crisis alpha" -> melhora o tombo.

ESTE MODULO (NOVO; offline, em PARALELO ao beta vivo) NAO edita as funcoes de
peso auditadas (beta_portfolio.weights_disciplined, beta_v2 leverage/financing) —
IMPORTA delas. NAO toca main.py / beta_rebalancer / config de producao.

CONTENDORES (mesmo periodo, custo, financiamento, vol-alvo, leverage 2.0x):
  BASE = beta puro (vol-target 1/vol, 2.0x, long-only) — exatamente o de producao.
  V1   = TREND-FILTER do beta long: des-arrisca POR-ATIVO quando o ativo esta
         ABAIXO da sua tendencia (retorno do lookback < 0); SEM shorts. So muda o
         vetor de pesos do MESMO book vol-target/alavancado (zera o nome em
         downtrend, redistribui via re-normalizacao do vol-target).
  V2   = SLEEVE TSMOM long/short combinado com o beta: 50% beta puro + 50% sleeve
         TSMOM (sinal = sign(retorno do lookback) por ativo, dimensionado a
         1/vol, escalado a vol-alvo, PODE SHORTAR em downtrend; diversificante).
         Combinado dolar-a-dolar e re-escalado p/ o MESMO gross-alvo do BASE.

PARAMS PADRAO (NAO otimizados): lookbacks de tendencia/TSMOM = 3, 6, 12 meses
(63, 126, 252 dias de pregao). Sensibilidade RODA os 3 lookbacks p/ TODAS as
variantes (a barra exige robustez em TODOS). Vol-alvo, lookback de vol, teto de
cripto, leverage e financiamento sao HERDADOS do beta de producao (nao mexemos).

HONESTIDADE (igual aos modulos auditados, nao-negociavel):
  - SEM LOOK-AHEAD: o sinal de trend no dia t usa retorno do lookback ate t-1 (o
    .shift(1) herdado dos construtores de beta_portfolio garante peso_t <- dado<=t-1;
    o retorno aplicado e o de t). Sinal de t com dado <= t-1, retorno t+1.
  - Custo de transacao real (simulation.costs) sobre |Delta peso|, INCLUI o
    turnover extra que o trend introduz (entrar/sair, virar long->short).
  - Custo de FINANCIAMENTO real sobre o capital emprestado (gross>1), inclusive o
    gross do lado short do TSMOM (shortar tambem consome margem).
  - DSR/PBO sobre a serie de variantes x lookbacks (n_trials = todas as
    combinacoes testadas) — anti data-snooping.

BARRA HONESTA (topo do relatorio): GRADUA p/ uma variante SO se ela BATE o beta
puro no risco-ajustado (Sharpe E/OU Calmar) LIQUIDO de custo, de forma ROBUSTA
(em TODOS os lookbacks 3/6/12), idealmente melhorando o MaxDD sem matar o
retorno. Senao, o BETA PURO fica. Se graduar, precisa de AUDITORIA DO CODER
antes de ir p/ config viva.

Uso:
    uv run python -m simulation.beta_trend
    uv run python -m simulation.beta_trend --force        # re-baixa o cache
    uv run python -m simulation.beta_trend --no-sensitivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# REUTILIZA os modulos auditados — NAO os edita. Importa pesos/custo/financiamento/metricas.
from simulation.beta_portfolio import (
    CRYPTO_MAX_WEIGHT,
    TRADING_DAYS,
    VOL_LOOKBACK,
    VOL_TARGET_ANNUAL,
    Stats,
    _hold_between_rebalances,
    _rebalance_mask,
    compute_stats,
    diversified_start,
    load_panel,
    weights_disciplined,
)
from simulation.beta_v2 import (
    FINANCING_ANNUAL,
    leverage_weights,
    run_portfolio_levered,
)
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.beta_trend")

# ----------------------------------------------------------------------------
# PARAMS PADRAO deste estudo (NAO otimizados; declarados no relatorio).
# ----------------------------------------------------------------------------
# Lookbacks de tendencia / TSMOM, em meses -> dias de pregao. A barra exige que a
# variante BATA o beta puro em TODOS estes (robustez aos lookbacks).
LOOKBACKS_MONTHS = (3, 6, 12)
MONTH_DAYS = 21  # ~dias de pregao por mes
DEFAULT_LOOKBACK_M = 6  # o lookback "headline" exibido nas tabelas principais

# Leverage de PRODUCAO (o beta vivo roda ~2.0x). Aplicado igualmente a BASE/V1/V2
# (head-to-head) sobre o vetor de pesos de cada variante, com financiamento real.
PROD_LEVERAGE = 2.0

# V2: peso do sleeve TSMOM na combinacao (50/50 beta puro + sleeve diversificante).
TSMOM_SLEEVE_WEIGHT = 0.50


def _lb_days(months: int) -> int:
    return int(round(months * MONTH_DAYS))


# ============================================================================
# CONSTRUTORES DE PESO (todos SEM look-ahead — .shift(1) no fim, como os auditados)
#
# BASE: o beta de producao = vol-target puro (use_gate=False) ALAVANCADO 2.0x.
#   Reusa weights_disciplined (auditado) + leverage_weights (auditado).
# ============================================================================
def weights_base(closes: pd.DataFrame, classes: dict[str, str]) -> pd.DataFrame:
    """BASE = beta puro de producao: vol-target 1/vol (sem portao), long-only,
    escalado p/ vol-alvo, gross<=1, depois ALAVANCADO 2.0x."""
    vt = weights_disciplined(closes, classes, use_gate=False, use_vol_target=True)
    return leverage_weights(vt, PROD_LEVERAGE)


def _trend_up_flags(close: pd.Series, lookback: int) -> pd.Series:
    """True nos dias em que o ativo esta em UPTREND: retorno do lookback >= 0,
    decidido com dado ate o PROPRIO dia t (o .shift(1) global cuida do look-ahead).

    Sinal TSMOM canonico (Moskowitz-Ooi-Pedersen): sinal de t = sign do retorno
    passado de `lookback` dias. retorno_passado_t = close_t / close_{t-lookback} - 1.
    Warmup (sem lookback completo) -> trata como UP (nao des-arrisca por falta de
    historico; o ativo so nasce no painel quando ja ha preco).
    """
    c = close.astype(float)
    past_ret = c / c.shift(lookback) - 1.0
    up = (past_ret >= 0.0)
    # warmup (NaN) -> True (nao penaliza por falta de dado; conservador p/ V1)
    return up.fillna(True)


def weights_trend_filter(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    lookback_months: int = DEFAULT_LOOKBACK_M,
    leverage: float = PROD_LEVERAGE,
) -> pd.DataFrame:
    """V1 = TREND-FILTER do beta long, SEM shorts.

    Reconstroi o vol-target EXATAMENTE como o auditado (1/vol, teto de cripto,
    escala p/ vol-alvo, gross<=1), mas ZERA o peso bruto de qualquer ativo em
    DOWNTREND (retorno do lookback < 0) ANTES de normalizar. O capital liberado e
    redistribuido pela propria re-normalizacao do vol-target (vai p/ os ativos em
    uptrend e/ou p/ caixa via a escala de vol-alvo). Depois alavanca igual ao BASE.

    Espelha a logica de beta_portfolio.weights_disciplined (vol-target puro) com o
    mask de tendencia no lugar do portao de regime — p/ a comparacao ser limpa
    (mesmo .shift(1), mesmo _hold_between_rebalances, mesmo teto de cripto).
    """
    lookback = _lb_days(lookback_months)
    rets = closes.pct_change()
    vol = rets.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK // 2).std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)

    raw = inv_vol.copy()
    raw[closes.isna()] = np.nan

    # TREND-FILTER: zera o peso bruto do ativo em DOWNTREND (sem shorts).
    for tkr in raw.columns:
        up = _trend_up_flags(closes[tkr], lookback)
        down = ~up
        raw.loc[down[down].index, tkr] = 0.0

    raw = raw.fillna(0.0)
    gross = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0)

    # teto de cripto (identico ao auditado).
    crypto_cols = [t for t in w.columns if classes.get(t) == "crypto"]
    if crypto_cols:
        for t in crypto_cols:
            over = (w[t] - CRYPTO_MAX_WEIGHT).clip(lower=0.0)
            w[t] = w[t] - over
        noncrypto = [t for t in w.columns if t not in crypto_cols]
        cut_total = (1.0 - w.sum(axis=1)).clip(lower=0.0)
        base = w[noncrypto].sum(axis=1).replace(0, np.nan)
        share = w[noncrypto].div(base, axis=0).fillna(0.0)
        w[noncrypto] = w[noncrypto] + share.mul(cut_total, axis=0)

    # vol-target ex-ante (ignora correlacao -> conservador), scale em [0,1].
    port_vol = (w * vol.fillna(0.0)).sum(axis=1)
    scale = (VOL_TARGET_ANNUAL / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    w = w.mul(scale, axis=0)

    # gross<=1 (resto caixa) ANTES de alavancar.
    gross_final = w.sum(axis=1)
    over = (gross_final - 1.0).clip(lower=0.0)
    w = w.sub(w.div(gross_final.replace(0, np.nan), axis=0).mul(over, axis=0), fill_value=0.0)

    held = _hold_between_rebalances(w, _rebalance_mask(closes.index))
    held = held.shift(1).fillna(0.0)
    return leverage_weights(held, leverage)


def _tsmom_sleeve_weights(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    lookback_months: int,
) -> pd.DataFrame:
    """Sleeve TSMOM LONG/SHORT puro (pre-shift, pre-leverage): sinal = sign(retorno
    do lookback) por ativo, dimensionado a 1/vol e escalado p/ a MESMA vol-alvo.

    LONG em uptrend (+1), SHORT em downtrend (-1). Gross (|long|+|short|)<=1 antes
    de alavancar. Diversificante: em bear sustentado o sleeve fica liquido SHORT.
    Sem teto de cripto explicito aqui (a vol-target ja amortece cripto via 1/vol;
    o teto e do book long-only). NAO faz .shift — o caller compoe e depois shifta.
    """
    lookback = _lb_days(lookback_months)
    rets = closes.pct_change()
    vol = rets.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK // 2).std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)

    past_ret = closes / closes.shift(lookback) - 1.0
    sign = np.sign(past_ret)  # +1 uptrend, -1 downtrend, 0 flat
    sign = sign.where(past_ret.notna(), 0.0).fillna(0.0)

    raw = (inv_vol * sign)
    raw[closes.isna()] = 0.0
    raw = raw.fillna(0.0)

    # normaliza pelo GROSS (soma dos |pesos|) -> long/short com gross<=1.
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0)

    # escala p/ vol-alvo (vol ex-ante do sleeve, ignorando correlacao -> conservador).
    sleeve_vol = (w.abs() * vol.fillna(0.0)).sum(axis=1)
    scale = (VOL_TARGET_ANNUAL / sleeve_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    w = w.mul(scale, axis=0)
    return w  # pre-shift, gross<=1


def weights_tsmom_combo(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    lookback_months: int = DEFAULT_LOOKBACK_M,
    sleeve_weight: float = TSMOM_SLEEVE_WEIGHT,
    leverage: float = PROD_LEVERAGE,
) -> pd.DataFrame:
    """V2 = beta puro + sleeve TSMOM long/short, combinados e re-escalados p/ o
    MESMO gross-alvo do BASE (head-to-head justo: mesma alavancagem total).

    combo_pre = (1-sleeve_weight)*beta_voltarget_puro + sleeve_weight*sleeve_tsmom
    (ambos com gross<=1 antes de combinar). Re-escala o combo p/ que seu gross
    medio iguale o do beta puro alavancado, depois aplica a MESMA leverage. Assim a
    UNICA diferenca vs BASE e a COMPOSICAO (o sleeve diversificante), nao o tamanho.
    """
    # beta puro vol-target (pre-shift, gross<=1) — reusa o auditado SEM leverage/shift.
    # weights_disciplined ja shifta; aqui precisamos do pre-shift p/ combinar limpo.
    # Reconstruimos o vol-target puro pre-shift identico ao auditado:
    rets = closes.pct_change()
    vol = rets.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK // 2).std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    raw = inv_vol.copy()
    raw[closes.isna()] = np.nan
    raw = raw.fillna(0.0)
    g = raw.sum(axis=1).replace(0, np.nan)
    beta = raw.div(g, axis=0).fillna(0.0)
    crypto_cols = [t for t in beta.columns if classes.get(t) == "crypto"]
    if crypto_cols:
        for t in crypto_cols:
            over = (beta[t] - CRYPTO_MAX_WEIGHT).clip(lower=0.0)
            beta[t] = beta[t] - over
        noncrypto = [t for t in beta.columns if t not in crypto_cols]
        cut_total = (1.0 - beta.sum(axis=1)).clip(lower=0.0)
        base = beta[noncrypto].sum(axis=1).replace(0, np.nan)
        share = beta[noncrypto].div(base, axis=0).fillna(0.0)
        beta[noncrypto] = beta[noncrypto] + share.mul(cut_total, axis=0)
    port_vol = (beta * vol.fillna(0.0)).sum(axis=1)
    bscale = (VOL_TARGET_ANNUAL / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    beta = beta.mul(bscale, axis=0)
    bgross = beta.sum(axis=1)
    bover = (bgross - 1.0).clip(lower=0.0)
    beta = beta.sub(beta.div(bgross.replace(0, np.nan), axis=0).mul(bover, axis=0), fill_value=0.0)

    sleeve = _tsmom_sleeve_weights(closes, classes, lookback_months=lookback_months)

    combo = (1.0 - sleeve_weight) * beta + sleeve_weight * sleeve

    # re-escala o combo p/ o MESMO gross medio do beta puro (head-to-head justo no
    # tamanho); a diferenca vs BASE passa a ser SO a composicao diversificante.
    beta_gross = beta.abs().sum(axis=1)
    combo_gross = combo.abs().sum(axis=1).replace(0, np.nan)
    match = (beta_gross / combo_gross).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combo = combo.mul(match, axis=0)

    held = _hold_between_rebalances(combo, _rebalance_mask(closes.index))
    held = held.shift(1).fillna(0.0)
    return leverage_weights(held, leverage)


# ============================================================================
# BACKTEST — usa o motor ALAVANCADO auditado (transacao + financiamento real).
# run_portfolio_levered cobra financiamento sobre (gross-1)+, inclusive o gross
# do lado SHORT do TSMOM (shortar consome margem). |peso| na turnover/gross.
# ============================================================================
def _run_window_lev(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    name: str, start: pd.Timestamp, financing: float = FINANCING_ANNUAL,
) -> tuple[Stats, pd.Series]:
    p = panel.loc[panel.index >= start]
    w = weights.loc[weights.index >= start]
    eq, net = run_portfolio_levered(p, w, classes, financing_annual=financing)
    return compute_stats(name, eq, net, w), net


def crisis_returns_lev(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    financing: float = FINANCING_ANNUAL,
) -> dict[str, float]:
    """Retorno acumulado em janelas de crise (historico INTEIRO, book alavancado)."""
    _, net = run_portfolio_levered(panel, weights, classes, financing_annual=financing)
    windows = {
        "2008 (GFC)": ("2008-01-01", "2009-03-31"),
        "2020 (COVID)": ("2020-02-01", "2020-04-30"),
        "2022 (Bear)": ("2022-01-01", "2022-12-31"),
    }
    out: dict[str, float] = {}
    for label, (a, b) in windows.items():
        seg = net.loc[(net.index >= a) & (net.index <= b)]
        if seg.size >= 5:
            out[label] = float((1.0 + seg).prod() - 1.0)
    return out


# ============================================================================
# AVALIACAO — barra honesta mecanica + robustez aos lookbacks.
# ============================================================================
@dataclass
class VariantRun:
    name: str
    lookback_m: int
    stats: Stats
    net: pd.Series


def _beats_base(v: Stats, base: Stats) -> tuple[bool, bool, bool, bool]:
    """A BARRA HONESTA, mecanica. GRADUA se a variante BATE o beta puro no
    risco-ajustado (Sharpe E/OU Calmar) liquido de custo, idealmente melhorando o
    MaxDD sem matar o retorno.

    Retorna (passa, sharpe_melhor, calmar_melhor, dd_melhor):
      - sharpe_melhor : Sharpe da variante >= Sharpe base + folga (0.05)
      - calmar_melhor : Calmar da variante >= Calmar base * 1.10
      - dd_melhor     : |MaxDD| da variante < |MaxDD| base (tombo menor)
      - retorno nao 'morto': CAGR da variante >= 70% do CAGR base (nao mata o retorno)
      - passa = (sharpe_melhor OU calmar_melhor) E retorno nao morto.
    A robustez (em TODOS os lookbacks) e checada FORA, no agregador.
    """
    sharpe_melhor = v.sharpe >= base.sharpe + 0.05
    calmar_melhor = base.calmar > 0 and v.calmar >= base.calmar * 1.10
    dd_melhor = abs(v.max_dd) < abs(base.max_dd)
    ret_vivo = v.cagr >= base.cagr * 0.70 if base.cagr > 0 else v.cagr >= base.cagr
    passa = (sharpe_melhor or calmar_melhor) and ret_vivo
    return passa, sharpe_melhor, calmar_melhor, dd_melhor


# ============================================================================
# RELATORIO
# ============================================================================
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _stats_header(label: str = "estrategia") -> list[str]:
    head = (
        f"  {label:<34} | {'CAGR':>7} | {'Sharpe':>6} | {'Sortino':>7} | "
        f"{'MaxDD':>7} | {'Calmar':>6} | {'pior ano':>14} | {'gross':>5}"
    )
    return [head, "  " + "-" * (len(head) - 2)]


def _stats_line(s: Stats) -> str:
    return (
        f"  {s.name:<34} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {s.sortino:>7.2f} | "
        f"{_fmt_pct(s.max_dd):>7} | {s.calmar:>6.2f} | {_fmt_pct(s.worst_year):>8} "
        f"({s.worst_year_label}) | {s.avg_exposure*100:>4.0f}%"
    )


def _robustness_table(
    base_by_lb: dict[int, Stats],
    v1_by_lb: dict[int, Stats],
    v2_by_lb: dict[int, Stats],
) -> tuple[str, dict[str, bool]]:
    """Tabela variante x lookback (3/6/12). GRADUA so se a variante bate o beta
    puro em TODOS os lookbacks. Retorna (texto, {variante: robusto?})."""
    head = (
        f"  {'variante / lookback':<22} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>7} | "
        f"{'Calmar':>6} | {'Sh>base?':>8} | {'Cal>base?':>9} | {'DD<base?':>8}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    robust = {"V1": True, "V2": True}
    for vname, by_lb in (("V1", v1_by_lb), ("V2", v2_by_lb)):
        all_pass = True
        for lb in LOOKBACKS_MONTHS:
            base = base_by_lb[lb]
            s = by_lb[lb]
            passa, sh_ok, cal_ok, dd_ok = _beats_base(s, base)
            all_pass = all_pass and passa
            lines.append(
                f"  {vname + ' ' + str(lb) + 'm vs base':<22} | {_fmt_pct(s.cagr):>7} | "
                f"{s.sharpe:>6.2f} | {_fmt_pct(s.max_dd):>7} | {s.calmar:>6.2f} | "
                f"{('SIM' if sh_ok else 'nao'):>8} | {('SIM' if cal_ok else 'nao'):>9} | "
                f"{('SIM' if dd_ok else 'nao'):>8}"
            )
        robust[vname] = all_pass
        lines.append("  " + "." * (len(head) - 2))
    # linha de referencia do base por lookback (o base muda de leverage? nao — muda
    # nada com o lookback; mostramos uma vez, mas o base e identico p/ os 3).
    b = base_by_lb[LOOKBACKS_MONTHS[0]]
    lines.append(
        f"  {'BASE (beta puro 2.0x)':<22} | {_fmt_pct(b.cagr):>7} | {b.sharpe:>6.2f} | "
        f"{_fmt_pct(b.max_dd):>7} | {b.calmar:>6.2f} | {'—':>8} | {'—':>9} | {'—':>8}"
    )
    lines.append("")
    lines.append("  ROBUSTO = bate o beta puro (Sharpe E/OU Calmar) em TODOS os lookbacks 3/6/12,")
    lines.append("  com o retorno NAO morto (CAGR >= 70% do base). Senao, o beta puro fica.")
    return "\n".join(lines), robust


def _stat_tribunal(
    runs: list[VariantRun], base_net: pd.Series
) -> tuple[str, float]:
    """DSR/PBO sobre a serie de variantes x lookbacks (n_trials = nº de combinacoes
    testadas) — anti data-snooping. Para o DSR pegamos a MELHOR variante por Sharpe
    e a deflacionamos pelo nº de trials; o PBO usa a matriz de retornos (alinhada)
    de TODAS as variantes + o base como colunas-config.

    Pergunta que o tribunal responde: o melhor Sharpe que achamos entre as
    variantes de trend e sinal ou ruido de ter testado varias? E o processo de
    'escolher a melhor variante' esta selecionando sorte (PBO alto)?
    """
    # n_trials = nº de variantes de trend testadas (V1 e V2 x 3 lookbacks = 6).
    n_trials = len(runs)
    # melhor variante por Sharpe.
    best = max(runs, key=lambda r: r.stats.sharpe)
    trial_sharpes_period = [observed_sharpe(r.net.to_numpy()) for r in runs]

    verdict = evaluate_edge(
        best.net.to_numpy(),
        n_trials=n_trials,
        trial_sharpes=trial_sharpes_period,
        periods_per_year=EQUITY_PERIODS,
        min_sharpe_annual=0.0,  # aqui o interesse e o DSR (significancia), nao a barra de Sharpe
        dsr_threshold=0.95,
    )

    # PBO: matriz (T, N) com as variantes + o base como colunas (alinhadas no indice comum).
    cols: dict[str, pd.Series] = {r.name + f"_{r.lookback_m}m": r.net for r in runs}
    cols["BASE"] = base_net
    mat = pd.DataFrame(cols).dropna()
    pbo = probability_of_backtest_overfitting(mat.to_numpy(), n_splits=10) if mat.shape[0] > 20 else float("nan")

    L: list[str] = []
    L.append(f"  Melhor variante por Sharpe: {best.name} {best.lookback_m}m "
             f"(Sharpe anual {best.stats.sharpe:.2f}).")
    L.append(f"  n_trials (variantes de trend testadas) = {n_trials}.")
    L.append(f"  DSR (deflacionado por {n_trials} trials) = {verdict.dsr:.3f}  "
             f"[obstaculo data-snooping Sharpe anual = {verdict.sr_benchmark_annual:.2f}].")
    L.append(f"  PSR (P[Sharpe verdadeiro > 0]) = {verdict.psr:.3f}.")
    if not np.isnan(pbo):
        L.append(f"  PBO (prob. de overfitting do processo de selecao) = {pbo:.3f}  "
                 f"[< 0.50 = aceitavel].")
    else:
        L.append("  PBO: amostra curta demais p/ CSCV.")
    L.append("")
    L.append("  LEITURA: DSR alto (>=0.95) => o melhor Sharpe das variantes SUPERA o que se")
    L.append("  esperaria do melhor de N tentativas por acaso (sinal). DSR baixo => o ganho")
    L.append("  aparente do trend e indistinguivel de sorte de ter testado varias configs.")
    L.append("  PBO alto (>=0.50) => escolher 'a melhor variante de trend' seleciona sorte.")
    return "\n".join(L), verdict.dsr


def _build_verdict(
    base: Stats,
    v1_head: Stats,
    v2_head: Stats,
    robust: dict[str, bool],
    dsr_best: float,
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("(4) VEREDITO — gradua (qual variante) ou o beta puro fica?")
    L.append("=" * 96)

    v1_ok = robust["V1"]
    v2_ok = robust["V2"]
    dsr_ok = dsr_best >= 0.95

    L.append(f"  BASE  (beta puro 2.0x) : CAGR {_fmt_pct(base.cagr)} | Sharpe {base.sharpe:.2f} | "
             f"MaxDD {_fmt_pct(base.max_dd)} | Calmar {base.calmar:.2f}")
    L.append(f"  V1 ({DEFAULT_LOOKBACK_M}m trend-filter) : CAGR {_fmt_pct(v1_head.cagr)} | Sharpe {v1_head.sharpe:.2f} | "
             f"MaxDD {_fmt_pct(v1_head.max_dd)} | Calmar {v1_head.calmar:.2f}")
    L.append(f"  V2 ({DEFAULT_LOOKBACK_M}m TSMOM combo) : CAGR {_fmt_pct(v2_head.cagr)} | Sharpe {v2_head.sharpe:.2f} | "
             f"MaxDD {_fmt_pct(v2_head.max_dd)} | Calmar {v2_head.calmar:.2f}")
    L.append("")
    L.append(f"  Robustez (bate o beta puro em TODOS os lookbacks 3/6/12?): "
             f"V1={'SIM' if v1_ok else 'NAO'} · V2={'SIM' if v2_ok else 'NAO'}.")
    L.append(f"  Tribunal estatistico: DSR da melhor variante = {dsr_best:.3f} "
             f"({'>=0.95 (sinal)' if dsr_ok else '<0.95 (indistinguivel de sorte)'}).")
    L.append("")

    graduates = None
    if v2_ok and dsr_ok:
        graduates = "V2"
    elif v1_ok and dsr_ok:
        graduates = "V1"

    if graduates == "V2":
        L.append("  >>> GRADUA: V2 (sleeve TSMOM long/short combinado com o beta).")
        L.append("      Bate o beta puro no risco-ajustado em TODOS os lookbacks E o DSR confirma")
        L.append("      que nao e sorte de ter testado varias configs. O sleeve diversificante")
        L.append("      (shorta em bear sustentado) entrega o 'crisis alpha' esperado da tese.")
        L.append("      *** PRECISA DE AUDITORIA DO CODER ANTES DA CONFIG VIVA. ***")
    elif graduates == "V1":
        L.append("  >>> GRADUA: V1 (trend-filter long-only).")
        L.append("      Bate o beta puro no risco-ajustado em TODOS os lookbacks E o DSR confirma.")
        L.append("      *** PRECISA DE AUDITORIA DO CODER ANTES DA CONFIG VIVA. ***")
    else:
        L.append("  >>> O BETA PURO FICA. Nenhuma variante de trend BATE o beta puro de forma")
        L.append("      ROBUSTA (em TODOS os lookbacks 3/6/12) no risco-ajustado com retorno vivo,")
        if not dsr_ok:
            L.append("      e o DSR da melhor variante nao chega a 0.95 — o ganho aparente do trend e")
            L.append("      indistinguivel de sorte de ter testado varias configuracoes.")
        else:
            L.append("      mesmo com DSR ok, a vantagem nao se sustenta em todos os lookbacks (fragil).")
        L.append("      O overlay de trend-following, nesta cesta multi-ativo ja vol-targeted e")
        L.append("      alavancada, NAO agrega valor risco-ajustado robusto. NAO vou vender o sonho.")
        # nuance: o trend pode ajudar SO no drawdown mesmo sem passar a barra cheia.
        if abs(v1_head.max_dd) < abs(base.max_dd) or abs(v2_head.max_dd) < abs(base.max_dd):
            which = "V1" if abs(v1_head.max_dd) < abs(base.max_dd) else "V2"
            L.append("")
            L.append(f"      NUANCE HONESTA: {which} REDUZ o MaxDD vs o beta puro, mas ao custo de")
            L.append("      Sharpe/CAGR — nao passa a barra de risco-ajustado. Se o objetivo fosse")
            L.append("      SO cortar tombo (a custo de retorno), seria um hedge opcional — mas a")
            L.append("      barra deste estudo e risco-AJUSTADO, e ai o trend nao domina.")
    return "\n".join(L)


def _assemble_report(
    panel: pd.DataFrame,
    div_start: pd.Timestamp,
    base_head: Stats,
    v1_head: Stats,
    v2_head: Stats,
    robustness_txt: str,
    dd_block: str,
    crisis: dict[str, dict[str, float]],
    tribunal_txt: str,
    verdict_txt: str,
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("BETA TREND — um overlay de TREND-FOLLOWING (TSMOM / managed futures) AGREGA valor")
    L.append("risco-ajustado ao beta de producao, de forma ROBUSTA? Cabeca-a-cabeca vs beta puro.")
    L.append("=" * 96)
    L.append("BARRA HONESTA: GRADUA p/ uma variante SO se ela BATE o beta puro no risco-ajustado")
    L.append("  (Sharpe E/OU Calmar) liquido de custo, de forma ROBUSTA (em TODOS os lookbacks")
    L.append("  3/6/12), idealmente melhorando o MaxDD sem matar o retorno. Senao, o beta puro fica.")
    L.append("")
    L.append("=" * 96)
    L.append("PREMISSAS / HONESTIDADE")
    L.append("=" * 96)
    L.append(
        "  BASE = beta de PRODUCAO (importado, NAO editado): vol-target 1/vol (sem portao),\n"
        f"  long-only, vol-alvo {VOL_TARGET_ANNUAL*100:.0f}% a.a., lookback vol {VOL_LOOKBACK}d, teto cripto "
        f"{CRYPTO_MAX_WEIGHT*100:.0f}%/nome,\n  alavancado {PROD_LEVERAGE:.1f}x com financiamento real."
    )
    L.append(
        "  V1 = trend-filter long-only: zera o peso do ativo em DOWNTREND (retorno do lookback < 0),\n"
        "  redistribui via re-normalizacao do vol-target; MESMA leverage; SEM shorts."
    )
    L.append(
        f"  V2 = {(1-TSMOM_SLEEVE_WEIGHT)*100:.0f}% beta puro + {TSMOM_SLEEVE_WEIGHT*100:.0f}% sleeve TSMOM "
        "long/short (sinal = sign do retorno do lookback,\n  1/vol, escalado a vol-alvo, PODE SHORTAR); "
        "combo re-escalado p/ o MESMO gross do BASE, MESMA leverage."
    )
    L.append(
        f"  LOOKBACKS PADRAO (NAO otimizados): {', '.join(str(m)+'m' for m in LOOKBACKS_MONTHS)} "
        f"(~{', '.join(str(_lb_days(m))+'d' for m in LOOKBACKS_MONTHS)}). Headline = {DEFAULT_LOOKBACK_M}m."
    )
    L.append(
        "  SEM LOOK-AHEAD: sinal de trend de t usa retorno do lookback ate t-1 (.shift(1) herdado);\n"
        "  retorno aplicado em t. Custo de transacao real sobre |Delta peso| (inclui o turnover do\n"
        "  trend e o virar long->short). Financiamento real sobre o capital emprestado, INCLUSIVE o\n"
        f"  gross do lado SHORT do TSMOM. Custo equities 3bps/lado, cripto 35bps/lado; financ. {FINANCING_ANNUAL*100:.1f}% a.a."
    )
    L.append("")
    spans = []
    for c in panel.columns:
        s = panel[c].dropna()
        if not s.empty:
            spans.append(f"{c}:{s.index[0].date()}→{s.index[-1].date()}({len(s)})")
    L.append(
        f"  PAINEL: {len(panel.columns)} ativos, {panel.index[0].date()}..{panel.index[-1].date()} "
        f"({len(panel)} dias). JANELA-CABECA a partir de {div_start.date()} (cesta diversificada)."
    )
    L.append("  " + "  ".join(spans))
    L.append("")
    L.append("=" * 96)
    L.append(f"(1) TABELA CABECA-A-CABECA — BASE vs V1 vs V2 (lookback headline = {DEFAULT_LOOKBACK_M}m)")
    L.append("=" * 96)
    L.extend(_stats_header())
    for s in (base_head, v1_head, v2_head):
        L.append(_stats_line(s))
    L.append("")
    L.append("=" * 96)
    L.append("(2) GANHO ROBUSTO AOS LOOKBACKS? (a barra exige bater o beta puro em TODOS: 3/6/12)")
    L.append("=" * 96)
    L.append(robustness_txt)
    L.append("")
    L.append("=" * 96)
    L.append("(3) MELHORA O DRAWDOWN? (MaxDD por lookback + comportamento em crise)")
    L.append("=" * 96)
    L.append(dd_block)
    L.append("")
    all_labels = sorted({lbl for d in crisis.values() for lbl in d})
    if all_labels:
        hdr = f"  {'estrategia':<26} | " + " | ".join(f"{lbl:>14}" for lbl in all_labels)
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for name, d in crisis.items():
            cells = " | ".join(f"{_fmt_pct(d[lbl]):>14}" if lbl in d else f"{'-':>14}" for lbl in all_labels)
            L.append(f"  {name:<26} | {cells}")
    else:
        L.append("  (historico nao cobre as janelas de crise)")
    L.append("")
    L.append("=" * 96)
    L.append("(3b) TRIBUNAL ESTATISTICO (DSR/PBO sobre variantes x lookbacks — anti data-snooping)")
    L.append("=" * 96)
    L.append(tribunal_txt)
    L.append("")
    L.append(verdict_txt)
    L.append("")
    L.append("=" * 96)
    L.append("(5) ARQUIVOS")
    L.append("=" * 96)
    L.append("  - simulation/beta_trend.py  (este modulo; importa beta_portfolio + beta_v2, nao edita)")
    L.append("  - data/beta_trend_report.txt  (este relatorio)")
    L.append("  Se graduar, a variante PRECISA de auditoria do Coder antes de qualquer config viva.")
    L.append("")
    return "\n".join(L)


# ============================================================================
# RUNNER
# ============================================================================
def run(force: bool = False, do_sensitivity: bool = True) -> tuple[str, bool]:
    panel, classes = load_panel(force=force)
    if panel.empty:
        msg = (
            "DADOS PENDENTES: nenhum fechamento baixado (rede?). Rode:\n"
            "  uv run python -m simulation.beta_trend --force\n"
            "Universo: SPY,QQQ,TLT,IEF,GLD,SLV,BTC-USD,ETH-USD"
        )
        return msg, False

    div_start = diversified_start(panel, classes)

    # --- BASE (beta puro 2.0x) — pesos no painel inteiro, warmup correto ---
    w_base = weights_base(panel, classes)
    base_head, base_net = _run_window_lev(panel, w_base, classes, "BASE: beta puro (2.0x)", div_start)

    # --- V1 / V2 por lookback (3/6/12) ---
    base_by_lb: dict[int, Stats] = {}
    v1_by_lb: dict[int, Stats] = {}
    v2_by_lb: dict[int, Stats] = {}
    runs: list[VariantRun] = []
    v1_nets: dict[int, pd.Series] = {}
    v2_nets: dict[int, pd.Series] = {}
    for lb in LOOKBACKS_MONTHS:
        base_by_lb[lb] = base_head  # base nao depende do lookback
        w_v1 = weights_trend_filter(panel, classes, lookback_months=lb)
        w_v2 = weights_tsmom_combo(panel, classes, lookback_months=lb)
        s_v1, net_v1 = _run_window_lev(panel, w_v1, classes, f"V1 {lb}m trend-filter", div_start)
        s_v2, net_v2 = _run_window_lev(panel, w_v2, classes, f"V2 {lb}m TSMOM combo", div_start)
        v1_by_lb[lb] = s_v1
        v2_by_lb[lb] = s_v2
        v1_nets[lb] = net_v1
        v2_nets[lb] = net_v2
        runs.append(VariantRun("V1", lb, s_v1, net_v1))
        runs.append(VariantRun("V2", lb, s_v2, net_v2))

    v1_head = v1_by_lb[DEFAULT_LOOKBACK_M]
    v2_head = v2_by_lb[DEFAULT_LOOKBACK_M]

    # --- (2) robustez aos lookbacks ---
    robustness_txt, robust = _robustness_table(base_by_lb, v1_by_lb, v2_by_lb)

    # --- (3) drawdown por lookback ---
    dd_lines: list[str] = []
    dd_lines.append(f"  {'variante':<22} | " + " | ".join(f"{str(lb)+'m':>9}" for lb in LOOKBACKS_MONTHS))
    dd_lines.append("  " + "-" * (24 + 12 * len(LOOKBACKS_MONTHS)))
    dd_lines.append(
        f"  {'BASE (beta puro 2.0x)':<22} | " + " | ".join(f"{_fmt_pct(base_head.max_dd):>9}" for _ in LOOKBACKS_MONTHS)
    )
    dd_lines.append(
        f"  {'V1 trend-filter':<22} | " + " | ".join(f"{_fmt_pct(v1_by_lb[lb].max_dd):>9}" for lb in LOOKBACKS_MONTHS)
    )
    dd_lines.append(
        f"  {'V2 TSMOM combo':<22} | " + " | ".join(f"{_fmt_pct(v2_by_lb[lb].max_dd):>9}" for lb in LOOKBACKS_MONTHS)
    )
    dd_lines.append("  (MaxDD; mais perto de 0 = melhor. A tese do trend e melhorar AQUI.)")
    dd_block = "\n".join(dd_lines)

    # --- crise (historico inteiro) ---
    crisis = {
        "BASE: beta puro 2.0x": crisis_returns_lev(panel, w_base, classes),
        f"V1 {DEFAULT_LOOKBACK_M}m trend-filter": crisis_returns_lev(
            panel, weights_trend_filter(panel, classes, lookback_months=DEFAULT_LOOKBACK_M), classes
        ),
        f"V2 {DEFAULT_LOOKBACK_M}m TSMOM combo": crisis_returns_lev(
            panel, weights_tsmom_combo(panel, classes, lookback_months=DEFAULT_LOOKBACK_M), classes
        ),
    }
    crisis = {k: v for k, v in crisis.items() if v}

    # --- (3b) tribunal estatistico ---
    tribunal_txt, dsr_best = _stat_tribunal(runs, base_net)

    # --- (4) veredito ---
    verdict_txt = _build_verdict(base_head, v1_head, v2_head, robust, dsr_best)

    report = _assemble_report(
        panel, div_start, base_head, v1_head, v2_head,
        robustness_txt, dd_block, crisis, tribunal_txt, verdict_txt,
    )
    graduated = (robust["V1"] or robust["V2"]) and dsr_best >= 0.95
    return report, graduated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beta trend: o overlay TSMOM agrega valor risco-ajustado robusto ao beta puro?"
    )
    parser.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    parser.add_argument("--no-sensitivity", action="store_true", help="(no-op: a sensibilidade SAO os lookbacks)")
    parser.add_argument("--report-file", default="data/beta_trend_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report, _graduated = run(force=args.force, do_sensitivity=not args.no_sensitivity)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
