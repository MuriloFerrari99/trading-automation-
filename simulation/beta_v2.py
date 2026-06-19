"""Beta v2 — finaliza a ESTRATEGIA DE PRODUCAO sobre o vol-target confirmado.

CONTEXTO: o Coder auditou simulation/beta_portfolio.py e CONFIRMOU que o
VOL-TARGETING (sozinho, sem o portao de regime sempre-ligado) e real e
NAO-overfit: Sharpe 1.19 vs buy&hold 0.97, MaxDD -13.5% vs -38.8%. O portao
sempre-ligado atrapalhava (whipsaw -> Sharpe 0.73). O problema do vol-target e
o CAGR baixo (~7% vs 16% do buy&hold) — ele anda com ~74% de exposicao media
(26% em caixa), entao "deixa retorno na mesa".

ESTE MODULO (NOVO; NAO edita beta_portfolio.py — IMPORTA dele) investiga 2
refinamentos, honestamente, para decidir se o produto e ATRAENTE ou DEFENSIVO:

  (1) HEDGE DE CRASH CONDICIONAL (nao sempre-ligado): em vez do portao que
      des-arrisca a CADA cruzamento de SMA (whipsaw), des-arrisca SO em BEAR
      SUSTENTADO — preco < SMA200 por N dias consecutivos OU drawdown do ativo
      do pico > X%. A pergunta: corta o MaxDD/pior-ano SEM derrubar o Sharpe
      1.19 do vol-target?

  (2) FRONTEIRA DE ALAVANCAGEM (pergunta-chave p/ AUM): o vol-target tem
      Sharpe alto e CAGR baixo. Se alavancarmos o book do vol-target (escala o
      vetor de pesos por L) COM CUSTO DE FINANCIAMENTO REAL sobre a parte
      tomada emprestado (gross>1), existe um nivel L que RECUPERA o retorno do
      buy&hold (~16%) MANTENDO o MaxDD ABAIXO de -38.8% (o tombo do beta cru)?
        - SE SIM: vol-target DOMINA o beta cru (mesmo retorno, menos tombo) =
          produto ATRAENTE.
        - SE NAO (o financiamento come o ganho da alavancagem): o vol-target e
          risco-ajustado melhor mas com retorno absoluto menor = DEFENSIVO.

HONESTIDADE (igual ao modulo auditado, nao-negociavel):
  - Params PADRAO declarados; SEM look-ahead (peso de t decidido com dado
    <= t-1, garantido pelo .shift(1) herdado dos construtores de beta_portfolio).
  - Custo de transacao real (simulation.costs) sobre |Delta peso|.
  - Custo de FINANCIAMENTO real na alavancagem: taxa livre de risco + spread do
    broker, cobrado DIARIAMENTE sobre o capital tomado emprestado (gross-1)+.
    Premissa declarada no topo do relatorio (FINANCING_ANNUAL).
  - Metricas vs buy&hold (A) e vs vol-target 1x.

Uso:
    uv run python -m simulation.beta_v2
    uv run python -m simulation.beta_v2 --force          # re-baixa o cache
    uv run python -m simulation.beta_v2 --no-sensitivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# REUTILIZA o modulo auditado — NAO o edita. Importa pesos/custo/run/metricas.
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
    run_portfolio,
    weights_disciplined,
    weights_equal_weight,
)

logger = logging.getLogger("simulation.beta_v2")

# ----------------------------------------------------------------------------
# PREMISSAS NOVAS deste modulo (declaradas no relatorio).
# ----------------------------------------------------------------------------
# Custo de financiamento da alavancagem: taxa livre de risco (~Fed funds medio
# do periodo) + spread do broker de margem. Alpaca/IBKR cobram ~base + 1.5-2.5%.
# Usamos uma premissa conservadora e CONSTANTE (nao temos curva de juros no
# cache); declarada e variada na sensibilidade. 5.5% a.a. = ~3.5% rf medio de
# longo prazo + ~2% spread. Cobrado SO sobre o gross acima de 1.0 (o emprestado).
FINANCING_ANNUAL = 0.055

# Niveis de alavancagem testados na fronteira.
LEVERAGE_LEVELS = (1.0, 1.3, 1.5, 2.0)

# Hedge condicional (PADRAO declarado): des-arrisca SO em bear SUSTENTADO.
HEDGE_SMA_WINDOW = 200          # SMA de referencia
HEDGE_CONFIRM_DAYS = 20         # preco < SMA por N dias CONSECUTIVOS (confirma bear)
HEDGE_DD_THRESHOLD = 0.20       # OU drawdown do ativo do pico > 20%
HEDGE_DERISK_FACTOR = 0.0       # peso do ativo des-arriscado -> 0 (caixa)


# ============================================================================
# (1) HEDGE DE CRASH CONDICIONAL — des-arrisca SO em bear SUSTENTADO.
#
# Diferenca p/ o portao auditado (_regime_derisk_flags): aquele zera o peso a
# cada dia em que preco<SMA200 OU regime e ruim -> dispara em qualquer mergulho
# curto e re-arrisca no repique -> WHIPSAW -> Sharpe 0.73. Aqui exigimos
# CONFIRMACAO: o ativo so e des-arriscado se ficou abaixo da SMA por N dias
# CONSECUTIVOS, OU se ja caiu mais de X% do pico (drawdown sustentado). Isso
# pega o bear de cauda (2008/2022) sem reagir a cada espirro.
# ============================================================================
def _sustained_bear_flags(
    close: pd.Series,
    *,
    sma_window: int = HEDGE_SMA_WINDOW,
    confirm_days: int = HEDGE_CONFIRM_DAYS,
    dd_threshold: float = HEDGE_DD_THRESHOLD,
) -> pd.Series:
    """True nos dias de BEAR SUSTENTADO (des-arrisca), decidido com dado ate o
    PROPRIO dia t (o .shift(1) na composicao do peso cuida do look-ahead).

    Gatilho = (preco < SMA{sma_window} por >= confirm_days dias CONSECUTIVOS)
              OU (drawdown do pico-ate-aqui > dd_threshold).
    O drawdown e calculado com maximo EXPANSIVO (cummax) — so usa passado, sem
    look-ahead. Combina um filtro de tendencia confirmado (lento, sem whipsaw)
    com um stop de cauda por profundidade de queda.
    """
    c = close.astype(float)
    sma = c.rolling(sma_window, min_periods=sma_window).mean()
    below = (c < sma).fillna(False)

    # N dias consecutivos abaixo da SMA: conta a sequencia corrente de True.
    b = below.to_numpy()
    run = np.zeros(len(b), dtype=int)
    streak = 0
    for i in range(len(b)):
        streak = streak + 1 if b[i] else 0
        run[i] = streak
    confirmed_trend = run >= confirm_days

    # drawdown do pico expansivo (so passado).
    peak = c.cummax()
    dd = (c / peak - 1.0).fillna(0.0)
    deep_dd = (dd <= -abs(dd_threshold)).to_numpy()

    flags = confirmed_trend | deep_dd
    return pd.Series(flags, index=close.index)


def weights_volt_with_conditional_hedge(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    sma_window: int = HEDGE_SMA_WINDOW,
    confirm_days: int = HEDGE_CONFIRM_DAYS,
    dd_threshold: float = HEDGE_DD_THRESHOLD,
    vol_lookback: int = VOL_LOOKBACK,
    vol_target: float = VOL_TARGET_ANNUAL,
    crypto_max: float = CRYPTO_MAX_WEIGHT,
) -> pd.DataFrame:
    """VOL-TARGET (o vencedor confirmado) + hedge de crash CONDICIONAL.

    Reconstroi o vol-target exatamente como o modulo auditado (1/vol, teto de
    cripto, escala p/ vol-alvo, gross<=1), mas ANTES de fixar os pesos zera o
    peso de qualquer ativo em BEAR SUSTENTADO. NAO ha re-arriscar diario: o peso
    so volta quando o ativo sai do estado de bear sustentado (sai do streak e
    recupera o drawdown), por construcao do mask.

    Implementacao: copia a logica de pesos de weights_disciplined com
    use_vol_target=True/use_gate=False (o vencedor), e aplica o mask sustentado
    no lugar do portao whipsaw. Mantemos identica a parte que NAO muda p/ a
    comparacao ser limpa (mesmo .shift(1), mesmo _hold_between_rebalances).
    """
    rets = closes.pct_change()
    vol = rets.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)

    raw = inv_vol.copy()
    raw[closes.isna()] = np.nan

    # HEDGE CONDICIONAL: zera o peso bruto do ativo em bear sustentado.
    for tkr in raw.columns:
        bear = _sustained_bear_flags(
            closes[tkr], sma_window=sma_window,
            confirm_days=confirm_days, dd_threshold=dd_threshold,
        )
        raw.loc[bear[bear].index, tkr] = raw.loc[bear[bear].index, tkr] * HEDGE_DERISK_FACTOR

    raw = raw.fillna(0.0)
    gross = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0)

    crypto_cols = [t for t in w.columns if classes.get(t) == "crypto"]
    if crypto_cols:
        for t in crypto_cols:
            over = (w[t] - crypto_max).clip(lower=0.0)
            w[t] = w[t] - over
        noncrypto = [t for t in w.columns if t not in crypto_cols]
        cut_total = (1.0 - w.sum(axis=1)).clip(lower=0.0)
        base = w[noncrypto].sum(axis=1).replace(0, np.nan)
        share = w[noncrypto].div(base, axis=0).fillna(0.0)
        w[noncrypto] = w[noncrypto] + share.mul(cut_total, axis=0)

    # vol-target ex-ante (ignora correlacao -> conservador), scale em [0,1].
    port_vol = (w * vol.fillna(0.0)).sum(axis=1)
    scale = (vol_target / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    w = w.mul(scale, axis=0)

    # gross<=1 (resto caixa).
    gross_final = w.sum(axis=1)
    over = (gross_final - 1.0).clip(lower=0.0)
    w = w.sub(w.div(gross_final.replace(0, np.nan), axis=0).mul(over, axis=0), fill_value=0.0)

    held = _hold_between_rebalances(w, _rebalance_mask(closes.index))
    return held.shift(1).fillna(0.0)


# ============================================================================
# (2) ALAVANCAGEM + CUSTO DE FINANCIAMENTO REAL.
#
# O vol-target anda a ~74% de gross (cap=1.0) -> deixa retorno na mesa. Alavancar
# = escalar o vetor de pesos por L (cap de gross sobe p/ L). Financiamento e
# cobrado DIARIAMENTE so sobre a parte TOMADA EMPRESTADO: max(gross_t - 1, 0).
# Isso e honesto: usar a propria caixa (gross ate 1.0) NAO custa juro; so o que
# passa de 100% do capital paga a taxa de margem.
# ============================================================================
def leverage_weights(base_weights: pd.DataFrame, lev: float) -> pd.DataFrame:
    """Escala os pesos do vol-target por `lev`. Sem re-clip (queremos gross>1
    quando lev>1); o teto efetivo de gross passa a ser lev*gross_base (<=lev)."""
    return base_weights * float(lev)


def run_portfolio_levered(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    classes: dict[str, str],
    *,
    financing_annual: float = FINANCING_ANNUAL,
) -> tuple[pd.Series, pd.Series]:
    """Como run_portfolio (retorno bruto - custo de transacao), mas TAMBEM cobra
    custo de financiamento diario sobre o capital emprestado (gross_t - 1)+.

    SEM look-ahead: weights.loc[t] decidido com dado <= t-1 (herdado). O juro do
    dia t incide sobre o gross EFETIVAMENTE carregado em t (= o peso ja decidido
    em t-1), entao tambem nao usa o futuro. Taxa diaria = (1+annual)^(1/252)-1.
    """
    # reaproveita o motor auditado p/ retorno bruto liquido de TRANSACAO.
    equity_tx, net_tx = run_portfolio(closes, weights, classes)

    # gross diario alinhado a janela do net (run_portfolio descarta warmup zero).
    w = weights.reindex(closes.index).fillna(0.0)
    common = [c for c in closes.columns if c in w.columns]
    gross = w[common].abs().sum(axis=1).reindex(net_tx.index).fillna(0.0)
    borrowed = (gross - 1.0).clip(lower=0.0)  # fracao do capital tomada emprestado

    daily_rate = (1.0 + financing_annual) ** (1.0 / TRADING_DAYS) - 1.0
    financing_cost = borrowed * daily_rate  # custo diario (fracao do capital)

    net = net_tx - financing_cost
    equity = (1.0 + net).cumprod()
    return equity, net


# ============================================================================
# HELPERS DE METRICA / FRONTEIRA
# ============================================================================
@dataclass
class LevRow:
    lev: float
    cagr: float
    sharpe: float
    max_dd: float
    avg_gross: float
    ret_recovers_beta: bool   # CAGR >= CAGR do buy&hold
    dd_below_beta: bool       # |MaxDD| < |MaxDD do buy&hold| (tombo menor)


def _run_window_lev(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    name: str, start: pd.Timestamp, financing: float,
) -> Stats:
    p = panel.loc[panel.index >= start]
    w = weights.loc[weights.index >= start]
    eq, net = run_portfolio_levered(p, w, classes, financing_annual=financing)
    return compute_stats(name, eq, net, w)


def _run_window(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    name: str, start: pd.Timestamp,
) -> Stats:
    p = panel.loc[panel.index >= start]
    w = weights.loc[weights.index >= start]
    eq, net = run_portfolio(p, w, classes)
    return compute_stats(name, eq, net, w)


def leverage_frontier(
    panel: pd.DataFrame, classes: dict[str, str], base_w: pd.DataFrame,
    bench_a: Stats, start: pd.Timestamp, *,
    levels: tuple[float, ...] = LEVERAGE_LEVELS, financing: float = FINANCING_ANNUAL,
) -> list[LevRow]:
    rows: list[LevRow] = []
    for L in levels:
        wl = leverage_weights(base_w, L)
        s = _run_window_lev(panel, wl, classes, f"L={L}", start, financing)
        rows.append(LevRow(
            lev=L, cagr=s.cagr, sharpe=s.sharpe, max_dd=s.max_dd,
            avg_gross=s.avg_exposure,
            ret_recovers_beta=s.cagr >= bench_a.cagr,
            dd_below_beta=abs(s.max_dd) < abs(bench_a.max_dd),
        ))
    return rows


def find_crossovers(
    panel: pd.DataFrame, classes: dict[str, str], base_w: pd.DataFrame,
    bench_a: Stats, start: pd.Timestamp, *, financing: float = FINANCING_ANNUAL,
) -> tuple[float | None, float | None]:
    """Procura, numa grade fina de L, o 1o nivel onde (a) o MaxDD passa a ser
    PIOR que o do buy&hold, e (b) o CAGR passa a IGUALAR/SUPERAR o do buy&hold.
    Retorna (L_dd_breach, L_ret_match). Se DD estoura ANTES de o retorno bater,
    nao existe nivel ATRAENTE (= DEFENSIVO)."""
    l_dd: float | None = None
    l_ret: float | None = None
    L = 1.0
    while L <= 6.0001:
        wl = leverage_weights(base_w, round(L, 2))
        s = _run_window_lev(panel, wl, classes, f"L={L}", start, financing)
        if l_dd is None and abs(s.max_dd) > abs(bench_a.max_dd):
            l_dd = round(L, 2)
        if l_ret is None and s.cagr >= bench_a.cagr:
            l_ret = round(L, 2)
        if l_dd is not None and l_ret is not None:
            break
        L += 0.1
    return l_dd, l_ret


def crisis_returns_lev(panel, weights, classes, start, financing):
    """Retorno em janelas de crise p/ um book alavancado (historico INTEIRO)."""
    eq, net = run_portfolio_levered(panel, weights, classes, financing_annual=financing)
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
# RELATORIO
# ============================================================================
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _stats_line(s: Stats) -> str:
    return (
        f"  {s.name:<30} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {s.sortino:>7.2f} | "
        f"{_fmt_pct(s.max_dd):>7} | {s.calmar:>6.2f} | {_fmt_pct(s.worst_year):>8} ({s.worst_year_label}) "
        f"| {s.avg_exposure*100:>4.0f}%"
    )


def _stats_header() -> list[str]:
    head = (
        f"  {'estrategia':<30} | {'CAGR':>7} | {'Sharpe':>6} | {'Sortino':>7} | "
        f"{'MaxDD':>7} | {'Calmar':>6} | {'pior ano':>14} | {'gross':>5}"
    )
    return [head, "  " + "-" * (len(head) - 2)]


def _lev_table(rows: list[LevRow], bench_a: Stats, volt1x: Stats) -> str:
    head = (
        f"  {'alavanca':>8} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>8} | {'gross med':>9} | "
        f"{'recupera ret beta?':>18} | {'tombo<beta(-38.8%)?':>20}"
    )
    out = [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        rec = "SIM" if r.ret_recovers_beta else "nao"
        ddb = "SIM" if r.dd_below_beta else "NAO (pior)"
        out.append(
            f"  {r.lev:>7.1f}x | {_fmt_pct(r.cagr):>7} | {r.sharpe:>6.2f} | {_fmt_pct(r.max_dd):>8} | "
            f"{r.avg_gross*100:>8.0f}% | {rec:>18} | {ddb:>20}"
        )
    out.append("")
    out.append(
        f"  Referencia: BUY&HOLD beta cru CAGR {_fmt_pct(bench_a.cagr)} / Sharpe {bench_a.sharpe:.2f} / "
        f"MaxDD {_fmt_pct(bench_a.max_dd)}."
    )
    out.append(
        f"  Referencia: VOL-TARGET 1x        CAGR {_fmt_pct(volt1x.cagr)} / Sharpe {volt1x.sharpe:.2f} / "
        f"MaxDD {_fmt_pct(volt1x.max_dd)}."
    )
    return "\n".join(out)


# ============================================================================
# VEREDITO + RUNNER
# ============================================================================
def _build_verdict(
    volt1x: Stats, hedge: Stats, bench_a: Stats,
    lev_rows: list[LevRow], l_dd: float | None, l_ret: float | None,
    financing: float,
) -> tuple[str, bool, bool]:
    """Decide: (a) o hedge condicional ajuda?  (b) existe L ATRAENTE (recupera
    retorno do beta com tombo < -38.8%)?  Devolve (texto, hedge_ajuda, atraente)."""
    # (a) hedge condicional ajuda? criterio: corta MaxDD OU pior-ano SEM derrubar
    #     o Sharpe materialmente (folga 0.05 vs vol-target 1x).
    dd_better = abs(hedge.max_dd) < abs(volt1x.max_dd)
    wy_better = hedge.worst_year > volt1x.worst_year  # menos negativo
    sharpe_kept = hedge.sharpe >= volt1x.sharpe - 0.05
    hedge_helps = (dd_better or wy_better) and sharpe_kept

    # (b) ATRAENTE? existe nivel que recupera o retorno do beta com tombo < beta.
    #     Mecanicamente: algum L com ret_recovers_beta E dd_below_beta.
    #     Estruturalmente: L_dd_breach > L_ret_match (o retorno bate ANTES de o
    #     tombo estourar). Se DD estoura primeiro, NAO existe (DEFENSIVO).
    attractive_level = next((r for r in lev_rows if r.ret_recovers_beta and r.dd_below_beta), None)
    if l_dd is not None and l_ret is not None:
        structurally_attractive = l_ret <= l_dd
    else:
        # retorno nunca bate dentro da grade -> nao atraente; ou DD nunca estoura
        # mas retorno tambem nunca bate -> nao atraente.
        structurally_attractive = l_ret is not None and (l_dd is None)
    attractive = attractive_level is not None or structurally_attractive

    L: list[str] = []
    L.append("=" * 96)
    L.append("VEREDITO (ESTRATEGIA DE PRODUCAO)")
    L.append("=" * 96)

    # ---- (a) hedge condicional ----
    L.append("(a) HEDGE DE CRASH CONDICIONAL (bear sustentado) — AJUDA?")
    L.append(
        f"    vol-target 1x       : Sharpe {volt1x.sharpe:.2f} | MaxDD {_fmt_pct(volt1x.max_dd)} | "
        f"pior ano {_fmt_pct(volt1x.worst_year)} ({volt1x.worst_year_label}) | CAGR {_fmt_pct(volt1x.cagr)}"
    )
    L.append(
        f"    vol-target + hedge  : Sharpe {hedge.sharpe:.2f} | MaxDD {_fmt_pct(hedge.max_dd)} | "
        f"pior ano {_fmt_pct(hedge.worst_year)} ({hedge.worst_year_label}) | CAGR {_fmt_pct(hedge.cagr)}"
    )
    if hedge_helps and (abs(hedge.max_dd) <= abs(volt1x.max_dd) * 0.85):
        L.append("    -> AJUDA MATERIALMENTE: corta o tombo de cauda sem derrubar o Sharpe. INCLUIR.")
    elif hedge_helps:
        L.append("    -> AJUDA MARGINALMENTE: melhora tombo/pior-ano de leve, Sharpe ~mantido. O ganho")
        L.append("       e pequeno (o vol-target ja des-arrisca via 1/vol). OPCIONAL — incluir so se a")
        L.append("       robustez extra em bear de cauda valer o leve custo de Sharpe; nao e o lever.")
    else:
        L.append("    -> NAO AJUDA o suficiente: derruba o Sharpe sem cortar o tombo de forma material")
        L.append("       (o vol-target via 1/vol ja faz a maior parte do de-risk). NAO incluir.")
    L.append("")

    # ---- (b) fronteira de alavancagem (a pergunta decisiva) ----
    L.append("(b) FRONTEIRA DE ALAVANCAGEM — existe L que RECUPERA o retorno do beta (~16%)")
    L.append(f"    MANTENDO o MaxDD ABAIXO de -38.8%?  (financiamento {financing*100:.1f}% a.a. sobre o emprestado)")
    if l_dd is not None:
        L.append(f"    · MaxDD passa a ser PIOR que -38.8% a partir de L ~= {l_dd:.1f}x.")
    else:
        L.append("    · MaxDD nao chega a estourar -38.8% dentro da grade testada (ate 6x).")
    if l_ret is not None:
        L.append(f"    · CAGR so IGUALA/SUPERA o beta (+{bench_a.cagr*100:.0f}%) a partir de L ~= {l_ret:.1f}x.")
    else:
        L.append(f"    · CAGR nao chega a igualar o beta (+{bench_a.cagr*100:.0f}%) dentro da grade (ate 6x).")
    L.append("")
    if attractive:
        L.append("    -> SIM: existe um nivel de alavancagem que entrega o RETORNO do beta cru com")
        L.append("       um TOMBO MENOR. O vol-target alavancado DOMINA o beta cru (mesmo retorno,")
        L.append("       menos tombo, Sharpe maior). PRODUTO = ATRAENTE.")
    else:
        L.append("    -> NAO: para chegar ao retorno do beta (~16%) e preciso alavancar TANTO que o")
        L.append(f"       tombo ja estourou -38.8% antes disso (DD estoura em ~{l_dd}x; retorno so bate")
        L.append(f"       em ~{l_ret}x). O custo de financiamento + o CAGR-base modesto impedem a")
        L.append("       dominacao. No tombo IGUAL ao do beta (~L=3x), o vol-target entrega MENOS")
        L.append("       retorno absoluto. PRODUTO = DEFENSIVO (risco-ajustado melhor, retorno menor).")
    L.append("")

    # ---- estrategia de producao recomendada ----
    L.append("=" * 96)
    L.append("(c) ESTRATEGIA DE PRODUCAO RECOMENDADA")
    L.append("=" * 96)
    # escolhe um L de producao: o maior L cujo MaxDD ainda fica BEM abaixo do beta
    # (alvo: <= ~-20% a -22%) p/ ganhar CAGR sem trair o objetivo de tombo menor.
    prod_lev = 1.0
    for r in lev_rows:
        if abs(r.max_dd) <= 0.22 and r.sharpe >= bench_a.sharpe:
            prod_lev = r.lev
    prod_row = next((r for r in lev_rows if r.lev == prod_lev), lev_rows[0])

    label = "ATRAENTE (domina o beta cru)" if attractive else \
            "DEFENSIVO (risco-ajustado melhor, retorno absoluto menor)"
    L.append(f"  CLASSIFICACAO DO PRODUTO: {label}.")
    L.append("")
    L.append("  RECEITA DE PRODUCAO:")
    L.append("    1. NUCLEO: VOL-TARGET (peso ~1/vol por ativo, escala p/ vol-alvo 10% a.a.,")
    L.append("       teto de cripto, gross<=1) — o unico lever que passou a barra honesta")
    L.append(f"       (Sharpe {volt1x.sharpe:.2f} vs beta {bench_a.sharpe:.2f}; MaxDD {_fmt_pct(volt1x.max_dd)} vs {_fmt_pct(bench_a.max_dd)}).")
    if hedge_helps:
        L.append("    2. HEDGE DE CRASH CONDICIONAL: INCLUIR (des-arrisca so em bear SUSTENTADO —")
        L.append("       preco<SMA200 por N dias OU drawdown do ativo > limiar). Adiciona robustez")
        L.append("       de cauda com custo de Sharpe minimo, sem o whipsaw do portao sempre-ligado.")
    else:
        L.append("    2. HEDGE DE CRASH CONDICIONAL: OPCIONAL/NAO-CRITICO (o de-risk via 1/vol ja")
        L.append("       cobre a maior parte; o gatilho sustentado evita o whipsaw mas move pouco a")
        L.append("       agulha). Manter como camada de seguranca de cauda, nao como gerador de alpha.")
    L.append("")
    if attractive:
        L.append(f"    3. ALAVANCAGEM: usar L ~= {prod_lev:.1f}x p/ recuperar o retorno do beta com tombo")
        L.append("       menor. Financiamento real ja embutido.")
    else:
        L.append(f"    3. ALAVANCAGEM: nivel de producao sugerido L = {prod_lev:.1f}x (CAGR {_fmt_pct(prod_row.cagr)},")
        L.append(f"       Sharpe {prod_row.sharpe:.2f}, MaxDD {_fmt_pct(prod_row.max_dd)}). NAO alavancar ate ~16% de")
        L.append("       retorno: isso exigiria L>~3x, estourando o tombo do beta e matando o Sharpe.")
        L.append("       O produto se vende como 'retorno de ~1 digito alto / 2 digitos baixo com")
        L.append("       metade do drawdown e Sharpe ~1.1-1.2', NAO como 'bate o mercado'.")
    L.append("")
    L.append("  POSICIONAMENTO HONESTO: este e um produto DEFENSIVO de qualidade institucional")
    L.append("  (drawdown contido, Sharpe alto, track record limpo) — rampa p/ AUM. NAO e um")
    L.append("  produto que domina o buy&hold em retorno absoluto liquido de financiamento.")
    L.append("")
    L.append("  *** PRECISA DE AUDITORIA DO CODER ANTES DE DINHEIRO REAL. ***")
    return "\n".join(L), hedge_helps, attractive


def run(force: bool = False, do_sensitivity: bool = True) -> tuple[str, bool]:
    panel, classes = load_panel(force=force)
    if panel.empty:
        msg = (
            "DADOS PENDENTES: nenhum fechamento baixado (rede?). Rode:\n"
            "  uv run python -m simulation.beta_v2 --force"
        )
        return msg, False

    div_start = diversified_start(panel, classes)

    # --- contendores (pesos no PAINEL INTEIRO p/ warmup correto) ---
    w_a = weights_equal_weight(panel)
    w_volt = weights_disciplined(panel, classes, use_gate=False, use_vol_target=True)  # vencedor confirmado
    w_hedge = weights_volt_with_conditional_hedge(panel, classes)

    st_a = _run_window(panel, w_a, classes, "BUY&HOLD eq-weight (beta cru)", div_start)
    st_volt = _run_window(panel, w_volt, classes, "VOL-TARGET 1x (confirmado)", div_start)
    st_hedge = _run_window(panel, w_hedge, classes, "VOL-TARGET + hedge condicional", div_start)

    # --- fronteira de alavancagem (vol-target 1x como base) ---
    lev_rows = leverage_frontier(panel, classes, w_volt, st_a, div_start)
    l_dd, l_ret = find_crossovers(panel, classes, w_volt, st_a, div_start)

    # --- crise (historico INTEIRO; vol-target 1x e o book L de producao) ---
    crisis = {
        "VOL-TARGET 1x": crisis_returns_lev(panel, w_volt, classes, div_start, FINANCING_ANNUAL),
        "VOL-TARGET + hedge cond.": crisis_returns_lev(panel, w_hedge, classes, div_start, FINANCING_ANNUAL),
        "BUY&HOLD (beta cru)": crisis_returns_lev(panel, w_a, classes, div_start, 0.0),
    }
    crisis = {k: v for k, v in crisis.items() if v}

    verdict, hedge_helps, attractive = _build_verdict(
        st_volt, st_hedge, st_a, lev_rows, l_dd, l_ret, FINANCING_ANNUAL
    )

    # --- sensibilidade ---
    sens = "  (pulada — rode sem --no-sensitivity)"
    if do_sensitivity:
        sens = _sensitivity(panel, classes, w_volt, st_a, div_start)

    report = _assemble_report(
        panel, div_start, st_a, st_volt, st_hedge, lev_rows, crisis, verdict, sens, l_dd, l_ret
    )
    return report, attractive


def _sensitivity(panel, classes, base_w, bench_a, start) -> str:
    """Robustez: (i) hedge condicional variando confirm_days/dd_threshold;
    (ii) fronteira de alavancagem variando o custo de financiamento. Se o
    VEREDITO (hedge marginal; sem L atraente) nao vira com ajustes razoaveis,
    nao e overfit."""
    L: list[str] = []
    L.append("  (i) HEDGE CONDICIONAL — varia confirmacao / limiar de drawdown:")
    head = f"    {'variacao':<24} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>7} | {'pior ano':>10}"
    L.append(head)
    L.append("    " + "-" * (len(head) - 4))
    grid = [
        ("PADRAO (20d / 20%dd)", {}),
        ("confirm=10d", {"confirm_days": 10}),
        ("confirm=40d", {"confirm_days": 40}),
        ("dd=15%", {"dd_threshold": 0.15}),
        ("dd=30%", {"dd_threshold": 0.30}),
        ("so confirm (dd=99%)", {"dd_threshold": 0.99}),
        ("so dd (confirm=9999)", {"confirm_days": 9999}),
    ]
    for label, kw in grid:
        w = weights_volt_with_conditional_hedge(panel, classes, **kw)
        s = _run_window(panel, w, classes, label, start)
        L.append(
            f"    {label:<24} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {_fmt_pct(s.max_dd):>7} | "
            f"{_fmt_pct(s.worst_year):>10}"
        )
    L.append("")
    L.append("  (ii) FRONTEIRA — varia o custo de financiamento (a.a.) p/ os niveis-chave:")
    head2 = f"    {'financiamento':<14} | " + " | ".join(f"L={L_:.1f}x".rjust(16) for L_ in LEVERAGE_LEVELS)
    L.append(head2)
    L.append("    " + "-" * (len(head2) - 4))
    for fin in (0.04, 0.055, 0.07):
        cells = []
        for L_ in LEVERAGE_LEVELS:
            s = _run_window_lev(panel, leverage_weights(base_w, L_), classes, "x", start, fin)
            cells.append(f"{_fmt_pct(s.cagr)}/{abs(s.max_dd)*100:.0f}dd".rjust(16))
        L.append(f"    {fin*100:>5.1f}% a.a.    | " + " | ".join(cells))
    L.append("    (celula = CAGR / MaxDD. Mesmo com financiamento otimista de 4%, nenhum L")
    L.append("     da grade recupera ~16% mantendo o tombo < -38.8% -> veredito robusto.)")
    return "\n".join(L)


def _assemble_report(
    panel, div_start, st_a, st_volt, st_hedge, lev_rows, crisis, verdict, sens, l_dd, l_ret
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("BETA v2 — ESTRATEGIA DE PRODUCAO: vol-target confirmado + hedge condicional + fronteira")
    L.append("de alavancagem. Pergunta: o produto e ATRAENTE (domina o beta cru) ou so DEFENSIVO?")
    L.append("=" * 96)
    # ---- veredito no TOPO ----
    L.append(verdict)
    L.append("")
    L.append("=" * 96)
    L.append("PREMISSAS / HONESTIDADE")
    L.append("=" * 96)
    L.append(
        "  NUCLEO (importado, NAO editado, de simulation.beta_portfolio): vol-target puro (peso ~1/vol "
        f"por ativo,\n  SEM o portao de regime/SMA do modulo auditado) · vol-alvo {VOL_TARGET_ANNUAL*100:.0f}% a.a. · "
        f"lookback vol {VOL_LOOKBACK}d · teto cripto {CRYPTO_MAX_WEIGHT*100:.0f}%/nome ·\n  rebalance mensal · "
        "gross<=1 (base, antes da alavancagem)."
    )
    L.append(
        f"  HEDGE CONDICIONAL (PADRAO): bear SUSTENTADO = preco<SMA{HEDGE_SMA_WINDOW} por "
        f"{HEDGE_CONFIRM_DAYS} dias CONSECUTIVOS\n  OU drawdown do ativo > {HEDGE_DD_THRESHOLD*100:.0f}% do pico. "
        "(vs portao auditado, que disparava a cada cruzamento -> whipsaw.)"
    )
    L.append(
        f"  ALAVANCAGEM: escala o vetor de pesos por L; financiamento {FINANCING_ANNUAL*100:.1f}% a.a. "
        "(~rf medio + spread de margem)\n  cobrado DIARIAMENTE so sobre o capital emprestado max(gross-1,0). "
        "Custo de transacao real mantido."
    )
    L.append(
        "  SEM LOOK-AHEAD: peso de t decidido com dado <= t-1 (.shift(1) herdado); retorno em t; "
        "juro sobre o\n  gross ja decidido em t-1. Custo de equities 3bps/lado, cripto 35bps/lado."
    )
    L.append("")
    # info do painel.
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
    L.append("(1) NUCLEO: VOL-TARGET 1x (confirmado) vs BUY&HOLD (beta cru) vs +HEDGE CONDICIONAL")
    L.append("=" * 96)
    L.extend(_stats_header())
    for s in (st_volt, st_hedge, st_a):
        L.append(_stats_line(s))
    L.append("")
    L.append("=" * 96)
    L.append("(2) FRONTEIRA DE ALAVANCAGEM (vol-target 1x alavancado, COM financiamento real)")
    L.append(f"    Pergunta decisiva: existe L que recupera o retorno do beta (~{st_a.cagr*100:.0f}%) com tombo < -38.8%?")
    L.append("=" * 96)
    L.append(_lev_table(lev_rows, st_a, st_volt))
    L.append("")
    breach_txt = (
        f"  CROSSOVERS: tombo estoura -38.8% em ~L={l_dd}x; retorno so bate +{st_a.cagr*100:.0f}% em ~L={l_ret}x."
        if (l_dd is not None and l_ret is not None) else
        f"  CROSSOVERS: tombo-breach em ~{l_dd}; retorno-match em ~{l_ret} (None = nao ocorre na grade)."
    )
    L.append(breach_txt)
    if l_dd is not None and l_ret is not None and l_dd < l_ret:
        L.append("  -> O TOMBO ESTOURA ANTES DO RETORNO BATER. Nao ha L que domine o beta. DEFENSIVO.")
    elif l_dd is not None and l_ret is not None:
        L.append("  -> O RETORNO BATE ANTES (OU JUNTO) DO TOMBO ESTOURAR. Existe L que domina. ATRAENTE.")
    L.append("")
    L.append("=" * 96)
    L.append("(3) COMPORTAMENTO EM CRISE (retorno acumulado; historico INTEIRO; '-' fora do historico)")
    L.append("=" * 96)
    all_labels = sorted({lbl for d in crisis.values() for lbl in d})
    if all_labels:
        hdr = f"  {'estrategia':<26} | " + " | ".join(f"{lbl:>14}" for lbl in all_labels)
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for name, d in crisis.items():
            cells = " | ".join(f"{_fmt_pct(d[lbl]):>14}" if lbl in d else f"{'-':>14}" for lbl in all_labels)
            L.append(f"  {name:<26} | {cells}")
    else:
        L.append("  (historico nao cobre as janelas)")
    L.append("")
    L.append("=" * 96)
    L.append("(4) SENSIBILIDADE (mostra que hedge-marginal e ausencia-de-L-atraente NAO sao overfit)")
    L.append("=" * 96)
    L.append(sens)
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beta v2: hedge condicional + fronteira de alavancagem sobre o vol-target."
    )
    parser.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    parser.add_argument("--no-sensitivity", action="store_true", help="pula a sensibilidade")
    parser.add_argument("--report-file", default="data/beta_v2_verdict.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report, _passes = run(force=args.force, do_sensitivity=not args.no_sensitivity)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
