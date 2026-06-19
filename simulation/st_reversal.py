"""Tribunal de SHORT-TERM REVERSAL (mean-reversion 1-5d em acoes) — R&D offline.

PERGUNTA (imparcial, MEDIR — nao assumir): reversao de curto prazo cross-seccional
(long perdedores recentes / short ganhadores recentes, hold 1-5d, dollar-neutral)
tem edge risco-ajustado ROBUSTO **depois do custo real** e/ou diversifica nosso beta?

Este modulo e R&D PURO, em paralelo ao beta vivo. NAO toca em beta_*, main.py,
nav_history nem config de producao. IMPORTA (read-only) de:
  - simulation.factors_equity : load_equity_panel/common_window (loader/cache de precos)
  - simulation.beta_portfolio : beta vol-target (serie do nosso produto, p/ corr)
  - simulation.costs          : EQUITY_BASE (custo real de equity, por lado)
  - simulation.statistics     : DSR/PSR/PBO (tribunal anti-overfitting)
  - simulation.metrics        : sharpe/sortino/max_drawdown

SINAL (cross-seccional, diario, SEM look-ahead):
  Em cada dia t, score = -1 * retorno acumulado dos ultimos `lookback` dias usando
  closes ATE t-1 (ret de t-lookback-1 .. t-1). Score MAIOR = mais perdedor recente =
  long. Ranqueia o universo; long top fracao `cut` (perdedores), short bottom `cut`
  (ganhadores), dollar-neutral (long=+0.5, short=-0.5). Hold `hold` dias via media de
  `hold` carteiras sobrepostas (overlapping portfolios — reduz turnover e ruido de
  reentrada diaria, padrao Jegadeesh-Titman). Peso de t aplicado ao retorno de t+1
  (weights.shift(1)) -> sem look-ahead.

CRITICO — TURNOVER ALTISSIMO: o custo real (estressado 2x) provavelmente mata. Por isso
  este tribunal roda um SWEEP DE CUSTO (0..stress) e reporta o BREAK-EVEN de custo
  (bps/lado onde o Sharpe liquido cruza ~0 e onde deixa de bater b&h). Veredito no
  cenario ESTRESSADO (custo equity 2x).

n_TRIALS HONESTO (anti data-snooping): lookback {1,2,3,5} x hold {1,2,3,5} x
  cut {decil 0.1, quintil 0.2, tercil 0.33} = 48 configs. Passado ao DSR; PBO sobre a
  matriz de todas as 48 series.

BARRA (regua do programa): PASSA so se:
  DSR >= 0.95  E  Sharpe liq robusto (estressado)  E
  (bate buy&hold/equal-weight  OU  diversifica: corr baixa com beta/SPY MELHORANDO o
   conjunto), robusto OOS. Senao: FALHA -> cemiterio.

Uso:
    uv run python -m simulation.st_reversal            # usa cache
    uv run python -m simulation.st_reversal --force    # re-baixa precos
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import EQUITY_BASE
from simulation.factors_equity import (
    SPY,
    beta_voltarget_returns,
    common_window,
    equal_weight_benchmark,
    load_equity_panel,
)
from simulation.metrics import max_drawdown, sharpe as _sharpe, sortino as _sortino
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.st_reversal")

REPORT_PATH = Path("data/st_reversal_verdict.txt")
TRADING_DAYS = EQUITY_PERIODS  # 252


# ============================================================================
# SINAL E SIMULACAO (cross-seccional diario, SEM look-ahead)
# ============================================================================
def reversal_weights(
    closes: pd.DataFrame, *, lookback: int, hold: int, cut: float
) -> pd.DataFrame:
    """Pesos diarios de reversao de curto prazo (dollar-neutral), SEM look-ahead.

    Para cada dia t:
      score_t = -1 * (close[t-1] / close[t-1-lookback] - 1)   (usa SO dados <= t-1)
    Long top fracao `cut` (mais negativos recentes), short bottom `cut`. Dollar-neutral.
    Overlapping `hold` carteiras: peso final = media dos books dos ultimos `hold` dias.
    O peso de t e DECIDIDO com info ate t-1; a simulacao aplica .shift(1) (captura ret
    de t+1). Logo, a propria construcao ja usa close ate t-1 -> dupla garantia anti-lookahead.
    """
    # ret acumulado dos ultimos `lookback` dias, conhecido em t-1 (shift 1).
    past_ret = closes / closes.shift(lookback) - 1.0
    score = -past_ret.shift(1)  # score de t usa dados <= t-1

    cols = score.columns
    # book "instantaneo" por dia (antes do overlap)
    w_inst = pd.DataFrame(0.0, index=score.index, columns=cols)
    arr = score.to_numpy()
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.flatnonzero(~np.isnan(row))
        if valid.size < 6:  # precisa de universo minimo p/ ranquear
            continue
        v = row[valid]
        order = valid[np.argsort(v)]  # ascendente: fim = maior score = mais perdedor
        k = max(1, int(round(valid.size * cut)))
        longs = order[-k:]   # maior score -> long (perdedores)
        shorts = order[:k]   # menor score -> short (ganhadores)
        w_inst.iloc[i, longs] = 0.5 / k
        w_inst.iloc[i, shorts] = -0.5 / k

    if hold <= 1:
        return w_inst
    # overlapping: media dos `hold` books instantaneos mais recentes (inclui o atual).
    w_over = w_inst.rolling(window=hold, min_periods=1).mean()
    return w_over


def simulate(
    closes: pd.DataFrame, weights_daily: pd.DataFrame, cost_per_side_bps: float
) -> pd.Series:
    """Retornos diarios LIQUIDOS. SEM LOOK-AHEAD: weights.shift(1) (decide t-1, ret t).

    Custo = |Delta peso| * custo_lado, cobrado no dia da mudanca de peso (turnover real).
    """
    rets = closes.pct_change().fillna(0.0)
    w = weights_daily.reindex(closes.index).fillna(0.0)
    w_eff = w.shift(1).fillna(0.0)  # decide em t-1, captura retorno de t
    common = [c for c in closes.columns if c in w_eff.columns]
    r = rets[common].to_numpy()
    we = w_eff[common].to_numpy()

    gross = (we * r).sum(axis=1)
    turnover = np.abs(np.diff(we, axis=0, prepend=np.zeros((1, we.shape[1]))))
    cost = turnover.sum(axis=1) * (cost_per_side_bps / 1e4)
    net = gross - cost

    net_s = pd.Series(net, index=closes.index)
    nz = np.flatnonzero(np.abs(we).sum(axis=1) > 0)
    start = int(nz[0]) if nz.size else 0
    return net_s.iloc[start:]


def gross_and_turnover(
    closes: pd.DataFrame, weights_daily: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Retorno BRUTO diario e turnover diario (sum |Delta peso|), p/ sweep de custo."""
    rets = closes.pct_change().fillna(0.0)
    w = weights_daily.reindex(closes.index).fillna(0.0)
    w_eff = w.shift(1).fillna(0.0)
    common = [c for c in closes.columns if c in w_eff.columns]
    r = rets[common].to_numpy()
    we = w_eff[common].to_numpy()
    gross = (we * r).sum(axis=1)
    turnover = np.abs(np.diff(we, axis=0, prepend=np.zeros((1, we.shape[1])))).sum(axis=1)
    nz = np.flatnonzero(np.abs(we).sum(axis=1) > 0)
    start = int(nz[0]) if nz.size else 0
    idx = closes.index[start:]
    return pd.Series(gross[start:], index=idx), pd.Series(turnover[start:], index=idx)


def net_from_components(gross: pd.Series, turnover: pd.Series, cost_per_side_bps: float) -> pd.Series:
    return gross - turnover * (cost_per_side_bps / 1e4)


# ============================================================================
# CONFIGS (n_trials honesto)
# ============================================================================
@dataclass(frozen=True)
class Config:
    lookback: int
    hold: int
    cut: float
    label: str


def build_configs() -> list[Config]:
    cfgs: list[Config] = []
    looks = [1, 2, 3, 5]
    holds = [1, 2, 3, 5]
    cuts = [(0.10, "D"), (0.20, "Qn"), (0.33, "T")]
    for lb in looks:
        for h in holds:
            for cv, cname in cuts:
                cfgs.append(Config(lb, h, cv, f"L{lb}_H{h}_{cname}"))
    return cfgs


# ============================================================================
# METRICAS
# ============================================================================
@dataclass
class Stat:
    name: str
    n_days: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    avg_turnover: float
    corr_beta: float
    corr_spy: float


def _cagr(net: pd.Series) -> float:
    if net.size < 2:
        return 0.0
    eq = (1.0 + net).cumprod()
    years = net.size / TRADING_DAYS
    val = eq.iloc[-1]
    if val <= 0:
        return -1.0
    return float(val ** (1.0 / max(years, 1e-9)) - 1.0)


def stat_of(
    name: str,
    net: pd.Series,
    beta_net: pd.Series | None,
    spy_net: pd.Series | None,
    avg_turnover: float = float("nan"),
) -> Stat:
    r = net.to_numpy()
    eq = (1.0 + net).cumprod().to_numpy()

    def _corr(other: pd.Series | None) -> float:
        if other is None:
            return float("nan")
        a, b = net.align(other, join="inner")
        if a.size < 30 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])

    return Stat(
        name=name,
        n_days=int(net.size),
        cagr=_cagr(net),
        sharpe=_sharpe(r, periods=TRADING_DAYS),
        sortino=_sortino(r, periods=TRADING_DAYS),
        max_dd=max_drawdown(eq),
        avg_turnover=avg_turnover,
        corr_beta=_corr(beta_net),
        corr_spy=_corr(spy_net),
    )


def _fmt_pct(x: float) -> str:
    return "  n/a" if x != x else f"{x * 100:6.1f}%"


def _fmt(x: float) -> str:
    return " n/a" if x != x else f"{x:6.2f}"


# ============================================================================
# OOS (walk-forward simples: 1a metade IS p/ escolher, 2a metade OOS)
# ============================================================================
def oos_check(
    gross_to: dict[str, tuple[pd.Series, pd.Series]],
    configs: list[Config],
    cost_bps: float,
) -> tuple[str, float, float]:
    """Escolhe a melhor config por Sharpe na 1a metade (IS) e mede na 2a (OOS).

    Retorna (label_escolhido, sharpe_IS, sharpe_OOS) — todos no custo dado.
    """
    # alinha todas as series ao indice comum
    aligned = pd.DataFrame(
        {c.label: net_from_components(*gross_to[c.label], cost_bps) for c in configs}
    ).dropna()
    if len(aligned) < 100:
        return ("", float("nan"), float("nan"))
    half = len(aligned) // 2
    is_df = aligned.iloc[:half]
    oos_df = aligned.iloc[half:]
    is_sr = is_df.apply(lambda s: observed_sharpe(s.to_numpy()) * np.sqrt(TRADING_DAYS))
    best = str(is_sr.idxmax())
    oos_sr = observed_sharpe(oos_df[best].to_numpy()) * np.sqrt(TRADING_DAYS)
    return (best, float(is_sr.max()), float(oos_sr))


# ============================================================================
# RUN
# ============================================================================
def run(force: bool = False) -> dict:
    lines: list[str] = []
    panel = load_equity_panel(force=force)
    if panel.empty:
        report = (
            "DADOS PENDENTES — nenhum preco baixado.\n"
            "Rode: uv run python -m simulation.st_reversal --force\n"
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report)
        return {"data_status": "blocked", "report": report}

    panel = common_window(panel, min_names=30)
    spy = panel[SPY] if SPY in panel.columns else None
    closes = panel[[c for c in panel.columns if c != SPY]].copy()

    start = closes.index[0].date()
    end = closes.index[-1].date()

    configs = build_configs()
    n_trials = len(configs)

    cost_base = EQUITY_BASE.per_side_bps            # 3 bps/lado
    cost_stress = EQUITY_BASE.stressed(2.0).per_side_bps  # 6 bps/lado

    # roda cada config UMA vez (bruto + turnover) e reaproveita p/ todos os custos.
    gross_to: dict[str, tuple[pd.Series, pd.Series]] = {}
    avg_to: dict[str, float] = {}
    for cfg in configs:
        w = reversal_weights(closes, lookback=cfg.lookback, hold=cfg.hold, cut=cfg.cut)
        g, to = gross_and_turnover(closes, w)
        gross_to[cfg.label] = (g, to)
        avg_to[cfg.label] = float(to.mean())

    net_stress = {lb: net_from_components(*gross_to[lb], cost_stress) for lb in gross_to}
    net_base = {lb: net_from_components(*gross_to[lb], cost_base) for lb in gross_to}
    net_zero = {lb: gross_to[lb][0] for lb in gross_to}

    # benchmarks e beta
    ew = equal_weight_benchmark(panel)
    spy_net = spy.pct_change(fill_method=None).fillna(0.0) if spy is not None else None
    beta_net = beta_voltarget_returns(force=False)

    # PBO sobre TODAS as configs (estressado)
    aligned_stress = pd.DataFrame(net_stress).dropna()
    pbo = (
        probability_of_backtest_overfitting(aligned_stress.to_numpy(), n_splits=16)
        if len(aligned_stress) > 50
        else float("nan")
    )

    # headline = melhor Sharpe estressado
    head = max(net_stress, key=lambda lb: observed_sharpe(net_stress[lb].to_numpy()))
    head_cfg = next(c for c in configs if c.label == head)

    # trial sharpes (por periodo) p/ DSR honesto
    trial_sr = [observed_sharpe(net_stress[c.label].to_numpy()) for c in configs]

    def verdict(net: pd.Series):
        return evaluate_edge(
            net.to_numpy(),
            n_trials=n_trials,
            trial_sharpes=trial_sr,
            periods_per_year=TRADING_DAYS,
            min_sharpe_annual=0.8,
            dsr_threshold=0.95,
        )

    v_stress = verdict(net_stress[head])
    v_base = verdict(net_base[head])
    v_zero = verdict(net_zero[head])

    head_stat_stress = stat_of("REV_head_stress", net_stress[head], beta_net, spy_net, avg_to[head])
    head_stat_base = stat_of("REV_head_base", net_base[head], beta_net, spy_net, avg_to[head])
    head_stat_zero = stat_of("REV_head_grossNOcost", net_zero[head], beta_net, spy_net, avg_to[head])

    ew_stat = stat_of("BENCH_equal_weight", ew, beta_net, spy_net)
    spy_stat = stat_of("BENCH_SPY", spy_net, beta_net, None) if spy_net is not None else None
    beta_stat = stat_of("BETA_voltarget", beta_net, None, spy_net) if beta_net is not None else None

    # ---- SWEEP DE CUSTO -> break-even (na headline) ----
    g_head, to_head = gross_to[head]
    cost_grid = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    sweep: list[tuple[float, float, float]] = []  # (cost, sharpe_liq, cagr)
    for cbps in cost_grid:
        net_c = net_from_components(g_head, to_head, cbps)
        sweep.append((cbps, _sharpe(net_c.to_numpy(), periods=TRADING_DAYS), _cagr(net_c)))
    # break-even de Sharpe~0: interpola onde sharpe cruza 0
    be_cost = float("nan")
    for (c0, s0, _), (c1, s1, _) in zip(sweep, sweep[1:]):
        if s0 > 0 >= s1 and s0 != s1:
            be_cost = c0 + (c1 - c0) * (s0 - 0.0) / (s0 - s1)
            break
    if be_cost != be_cost and sweep[0][1] <= 0:
        be_cost = 0.0  # ja morto a custo zero

    # ---- OOS na headline (custo estressado) ----
    oos_label, oos_is_sr, oos_oos_sr = oos_check(gross_to, configs, cost_stress)
    # tambem mede a propria headline OOS especificamente
    head_aligned = net_stress[head]
    half = len(head_aligned) // 2
    head_oos_sr = observed_sharpe(head_aligned.iloc[half:].to_numpy()) * np.sqrt(TRADING_DAYS)
    head_is_sr = observed_sharpe(head_aligned.iloc[:half].to_numpy()) * np.sqrt(TRADING_DAYS)

    # ---- robustez: dispersao de Sharpe estressado entre TODAS as configs ----
    all_sr_ann = [observed_sharpe(net_stress[c.label].to_numpy()) * np.sqrt(TRADING_DAYS) for c in configs]
    sr_mean, sr_std = float(np.mean(all_sr_ann)), float(np.std(all_sr_ann))
    frac_pos = float(np.mean([s > 0 for s in all_sr_ann]))

    # =====================  RELATORIO  =====================
    lines.append("=" * 80)
    lines.append("TRIBUNAL — SHORT-TERM REVERSAL (mean-reversion 1-5d em acoes), dollar-neutral")
    lines.append("=" * 80)
    lines.append(f"Universo: {len(closes.columns)} nomes liquidos (S&P 100). Janela {start} .. {end}")
    lines.append(f"Dias de pregao avaliados (headline): {head_aligned.size}")
    lines.append(f"Custo (EQUITY_BASE): base {cost_base:.0f} bps/lado | "
                 f"VEREDITO no ESTRESSADO {cost_stress:.0f} bps/lado")
    lines.append(f"n_trials (honesto) = {n_trials}  (lookback{{1,2,3,5}} x hold{{1,2,3,5}} x cut{{0.10,0.20,0.33}})")
    lines.append(f"PBO (CSCV, todas as {n_trials} configs, estressado) = {_fmt(pbo)}  [<0.5 minimo; menor=melhor]")
    lines.append("")
    lines.append(f"HEADLINE = {head}  (lookback={head_cfg.lookback}d, hold={head_cfg.hold}d, cut={head_cfg.cut})")
    lines.append("")

    lines.append("(1) HEADLINE em 3 cenarios de custo vs benchmark vs beta")
    lines.append("-" * 80)
    hdr = f"{'serie':<24}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'turn/d':>8}{'cBeta':>7}{'cSPY':>7}"
    lines.append(hdr)
    rows = [head_stat_zero, head_stat_base, head_stat_stress, ew_stat]
    if spy_stat:
        rows.append(spy_stat)
    if beta_stat:
        rows.append(beta_stat)
    for s in rows:
        lines.append(
            f"{s.name:<24}{_fmt_pct(s.cagr):>8}{_fmt(s.sharpe):>8}{_fmt(s.sortino):>8}"
            f"{_fmt_pct(s.max_dd):>8}{_fmt(s.avg_turnover):>8}{_fmt(s.corr_beta):>7}{_fmt(s.corr_spy):>7}"
        )
    lines.append("")
    lines.append("    DSR/PSR da headline (n_trials e dispersao real entre trials):")
    lines.append(f"      grossNOcost : {v_zero.summary()}")
    lines.append(f"      base(3bps)  : {v_base.summary()}")
    lines.append(f"      stress(6bps): {v_stress.summary()}")
    lines.append("")

    lines.append("(2) SWEEP DE CUSTO (headline) — onde o edge morre")
    lines.append("-" * 80)
    lines.append(f"    {'custo bps/lado':>16}{'Sharpe_liq':>12}{'CAGR':>10}")
    for c, s, cg in sweep:
        flag = "  <== custo base/estress" if c in (cost_base, cost_stress) else ""
        lines.append(f"    {c:>16.1f}{s:>12.2f}{_fmt_pct(cg):>10}{flag}")
    lines.append(f"    BREAK-EVEN de custo (Sharpe_liq -> 0): "
                 f"{('%.1f bps/lado' % be_cost) if be_cost==be_cost else 'n/a'}")
    lines.append(f"    -> turnover medio diario da headline = {avg_to[head]:.2f} "
                 f"(sum|dw|/dia; cada 1.0 = ~1x book girado/dia)")
    lines.append("")

    lines.append("(3) OOS (walk-forward 50/50, custo estressado)")
    lines.append("-" * 80)
    lines.append(f"    Headline {head}: Sharpe IS(1a metade)={head_is_sr:.2f}  OOS(2a metade)={head_oos_sr:.2f}")
    lines.append(f"    Selecao-por-IS: melhor IS = {oos_label}  Sharpe_IS={oos_is_sr:.2f}  -> OOS={oos_oos_sr:.2f}")
    lines.append("")

    lines.append("(4) ROBUSTEZ entre as 48 configs (Sharpe anual estressado)")
    lines.append("-" * 80)
    lines.append(f"    media={sr_mean:.2f}  desvio={sr_std:.2f}  fracao c/ Sharpe>0 = {frac_pos*100:.0f}%")
    lines.append("")

    # =====================  VEREDITO  =====================
    ew_sr = _sharpe(ew.to_numpy(), periods=TRADING_DAYS)
    s = head_stat_stress
    beats_bh = s.sharpe > ew_sr + 0.1
    diversifies = (
        s.corr_beta == s.corr_beta and abs(s.corr_beta) < 0.3 and s.sharpe > 0.3
    ) or (
        s.corr_beta != s.corr_beta and s.corr_spy == s.corr_spy and abs(s.corr_spy) < 0.3 and s.sharpe > 0.3
    )
    oos_ok = head_oos_sr == head_oos_sr and head_oos_sr > 0.3
    passes = bool(v_stress.passes_dsr and v_stress.passes_sharpe and (beats_bh or diversifies) and oos_ok)

    corr_for_report = s.corr_beta if s.corr_beta == s.corr_beta else s.corr_spy

    lines.append("(5) VEREDITO (cenario ESTRESSADO — a regua do programa)")
    lines.append("-" * 80)
    lines.append(f"  DSR>=0.95 ? {v_stress.dsr:.3f}  -> {'OK' if v_stress.passes_dsr else 'FALHA'}")
    lines.append(f"  Sharpe_liq>=0.8 ? {s.sharpe:.2f}  -> {'OK' if v_stress.passes_sharpe else 'FALHA'}")
    lines.append(f"  Bate buy&hold (EW Sharpe={ew_sr:.2f}) ? {'sim' if beats_bh else 'nao'}")
    lines.append(f"  Diversifica (corr baixa, Sharpe>0.3) ? {'sim' if diversifies else 'nao'} "
                 f"(corr={_fmt(corr_for_report).strip()})")
    lines.append(f"  OOS robusto (Sharpe OOS>0.3) ? {'sim' if oos_ok else 'nao'} (OOS={head_oos_sr:.2f})")
    lines.append(f"  PBO aceitavel (<0.5) ? {'sim' if (pbo==pbo and pbo<0.5) else 'NAO/indef'} ({_fmt(pbo).strip()})")
    lines.append("")
    flag = "CANDIDATO A SLEEVE" if passes else "CEMITERIO (reprovado)"
    lines.append(f"  >>> VEREDITO FINAL: {flag}")
    lines.append("")
    if not passes:
        # diagnostico curto do que matou
        if not v_stress.passes_sharpe:
            lines.append("  DIAGNOSTICO: custo real consome o edge — Sharpe liquido abaixo da barra.")
        if be_cost == be_cost and be_cost < cost_stress:
            lines.append(f"  DIAGNOSTICO: break-even de custo ({be_cost:.1f} bps/lado) < custo estressado "
                         f"({cost_stress:.0f} bps/lado) -> turnover altissimo mata a estrategia.")
    lines.append("=" * 80)

    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)

    return {
        "data_status": "real",
        "report": report,
        "passed": passes,
        "strategy": "Short-term reversal (mean-reversion 1-5d em acoes)",
        "sharpe": float(s.sharpe),
        "cagr": float(s.cagr),
        "maxdd": float(s.max_dd),
        "dsr": float(v_stress.dsr),
        "pbo": float(pbo) if pbo == pbo else None,
        "corr_to_market": float(corr_for_report) if corr_for_report == corr_for_report else None,
        "be_cost_bps": float(be_cost) if be_cost == be_cost else None,
        "cost_stress_bps": float(cost_stress),
        "oos_sharpe": float(head_oos_sr) if head_oos_sr == head_oos_sr else None,
        "head": head,
        "n_trials": n_trials,
        "beats_bh": bool(beats_bh),
        "ew_sharpe": float(ew_sr),
        "gross_sharpe": float(head_stat_zero.sharpe),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tribunal de short-term reversal (1-5d)")
    parser.add_argument("--force", action="store_true", help="re-baixa precos (ignora cache)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    out = run(force=args.force)
    print(out["report"])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
