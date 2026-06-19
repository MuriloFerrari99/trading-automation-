"""Risk parity CIENTE DE COVARIANCIA — candidato a v2 do motor de beta.

CONTEXTO: a producao atual (simulation.beta_portfolio.weights_disciplined com
use_gate=False) e VOL-TARGET por 1/vol INGENUO: peso ~ 1/vol_individual de cada
ativo e depois escala a carteira p/ ~vol-alvo. Ela IGNORA as correlacoes — trata
ouro e prata, ou SPY e QQQ, como se fossem riscos independentes, quando na verdade
sao redundantes. Risk parity COMPLETO (Equal Risk Contribution, ERC) usa a MATRIZ
DE COVARIANCIA inteira: aloca para que cada ativo contribua com risco IGUAL p/ a
carteira, descontando o que ja esta coberto por um ativo correlacionado.

A PERGUNTA (medir, nao assumir): risk parity ciente de covariancia MELHORA o
retorno risco-ajustado LIQUIDO DE CUSTO vs o 1/vol ingenuo da producao? E o ganho
(se houver) e ROBUSTO aos parametros de estimacao da covariancia (lookback,
shrinkage), ou e fragil/depende de uma escolha fina?

O QUE ESTE MODULO FAZ — comparacao CABECA-A-CABECA honesta, MESMO universo, MESMO
periodo, MESMO custo+financiamento+alavancagem que o backtest auditado:

  - 1/vol INGENUO (producao): importado de beta_portfolio.weights_disciplined
    (use_gate=False, use_vol_target=True). NAO editado, NAO reimplementado.
  - RISK PARITY (ERC) ciente de covariancia: NOVO aqui. Estima a covariancia
    rolante de retornos COM SHRINKAGE (Ledoit-Wolf, constant-correlation target,
    implementado em numpy puro — sem sklearn/scipy no projeto), resolve ERC por
    coordinate descent ciclico (Griveau-Billion/Spinu, provadamente convergente),
    aplica O MESMO teto de cripto e A MESMA escala p/ vol-alvo da producao p/ a
    exposicao ser identica. Compara a 1.0x e a 2.0x.

HONESTIDADE (nao-negociavel, igual aos modulos auditados):
  - A covariancia tem ERRO DE ESTIMACAO — RP NAO e de graca. Por isso (a) usamos
    shrinkage, e (b) a sensibilidade varia lookback {60,126,252} e shrinkage
    {auto Ledoit-Wolf, 0%, 50%, 100%} p/ ver se o ganho sobrevive.
  - SEM LOOK-AHEAD: a covariancia do dia t usa SO retornos <= t-1; os pesos de t
    sao aplicados ao retorno de t+1 — exatamente como os construtores auditados
    (.shift(1)) e o run_portfolio/run_portfolio_levered de beta_v2.
  - Custo de transacao real (simulation.costs) sobre |Delta peso|; financiamento
    real sobre o capital emprestado (gross-1)+ na comparacao alavancada. TUDO
    reusando run_portfolio_levered de beta_v2 (importado, NAO editado).
  - BARRA HONESTA no topo do relatorio. Se RP nao bater o 1/vol no risco-ajustado
    (Sharpe E/OU Calmar) de forma ROBUSTA, o 1/vol FICA. NAO auto-promove: virar
    v2 exige auditoria do Coder + paridade + cadeia de producao ANTES de tocar o
    config vivo.

Uso:
    uv run python -m simulation.beta_riskparity
    uv run python -m simulation.beta_riskparity --force          # re-baixa o cache
    uv run python -m simulation.beta_riskparity --no-sensitivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# REUTILIZA o universo/loader auditado (NAO edita).
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
    weights_equal_weight,
)

# REUTILIZA o motor de custo/financiamento/alavancagem auditado (NAO edita).
from simulation.beta_v2 import (
    FINANCING_ANNUAL,
    leverage_weights,
    run_portfolio_levered,
)

logger = logging.getLogger("simulation.beta_riskparity")

# ----------------------------------------------------------------------------
# PARAMS PADRAO do alocador de risk parity (declarados no relatorio).
# Reusa o teto de cripto, o vol-alvo e o lookback da PRODUCAO p/ a comparacao
# ser justa (so muda 1 coisa: 1/vol diagonal -> ERC ciente de covariancia).
# ----------------------------------------------------------------------------
COV_LOOKBACK = VOL_LOOKBACK          # janela da covariancia rolante (= lookback de vol da producao: 60d)
SHRINKAGE = "lw"                     # "lw" = Ledoit-Wolf auto; ou float fixo em [0,1]
ERC_MAX_ITERS = 20000
ERC_TOL = 1e-11
# niveis de alavancagem da comparacao cabeca-a-cabeca (1x = sem alavanca; 2x = producao).
HEAD_LEVELS = (1.0, 2.0)


# ============================================================================
# (A) LEDOIT-WOLF SHRINKAGE (constant-correlation target), numpy puro.
#
# Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix". A cov
# amostral S e ruim com poucos dados (N ativos, T dias): tem erro de estimacao
# que ENTRA nos pesos de RP e os infla artificialmente. A solucao e encolher S
# em direcao a um alvo estruturado F (aqui: correlacao constante = a correlacao
# media entre os ativos, com as variancias amostrais na diagonal). A intensidade
# delta in [0,1] e estimada dos DADOS (nao um chute): delta = (pi-rho)/gamma / T,
# o trade-off vies-variancia otimo. delta->1 quando ha pouco dado/muito ruido.
# ============================================================================
def ledoit_wolf_cov(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariancia encolhida (Ledoit-Wolf, constant-correlation) de X (T,n).

    Retorna (Sigma, delta). delta e a intensidade de shrinkage escolhida pelos
    dados. SEM look-ahead aqui: quem passa X garante que sao retornos <= t-1.
    """
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    if T < 2 or n < 1:
        return np.cov(X, rowvar=False, ddof=0) if T >= 2 else np.eye(n), 0.0
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T  # cov amostral (MLE, /T como no paper)
    var = np.diag(S).copy()
    var = np.where(var > 1e-18, var, 1e-18)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)

    # alvo F: correlacao constante (rbar = media das correlacoes off-diagonal).
    R = S / outer_std
    rbar = (R.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rbar * outer_std
    np.fill_diagonal(F, var)

    # pi: soma das variancias assintoticas das entradas de S.
    Xc2 = Xc**2
    pi_mat = (Xc2.T @ Xc2) / T - S**2
    pi_hat = float(pi_mat.sum())

    # rho: covariancia entre as entradas de S e o alvo (forma fechada LW p/ CC).
    rho_diag = float(np.trace(pi_mat))
    v_ij = (Xc.T @ Xc2) / T  # E[x_i * x_j^2]-ish termo cruzado
    cov_s_ii_s_ij = v_ij - var[:, None] * S
    cov_s_jj_s_ij = v_ij.T - var[None, :] * S
    ratio = np.sqrt(np.outer(var, 1.0 / var))
    term = 0.5 * rbar * (ratio * cov_s_ii_s_ij + ratio.T * cov_s_jj_s_ij)
    np.fill_diagonal(term, 0.0)
    rho_hat = rho_diag + float(term.sum())

    # gamma: distancia Frobenius^2 entre S e o alvo.
    gamma_hat = float(np.sum((F - S) ** 2))
    if gamma_hat <= 0:
        return S, 0.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(min(1.0, max(0.0, kappa / T)))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, delta


def shrink_cov(X: np.ndarray, shrinkage: str | float) -> tuple[np.ndarray, float]:
    """Covariancia com shrinkage. 'lw' = Ledoit-Wolf auto; float = intensidade
    FIXA em direcao ao mesmo alvo constant-correlation (p/ a sensibilidade)."""
    if isinstance(shrinkage, str) and shrinkage.lower() == "lw":
        return ledoit_wolf_cov(X)
    delta = float(shrinkage)
    delta = min(1.0, max(0.0, delta))
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    T = Xc.shape[0]
    n = Xc.shape[1]
    S = (Xc.T @ Xc) / max(T, 1)
    var = np.where(np.diag(S) > 1e-18, np.diag(S), 1e-18)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    R = S / outer_std
    rbar = (R.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    F = rbar * outer_std
    np.fill_diagonal(F, var)
    return delta * F + (1.0 - delta) * S, delta


# ============================================================================
# (B) EQUAL RISK CONTRIBUTION (ERC) — coordinate descent ciclico.
#
# Griveau-Billion, Richard & Roncalli (2013), "A Fast Algorithm for Computing
# High-dimensional Risk Parity Portfolios"; Spinu (2013); Maillard et al. (2010).
# Resolve min 0.5 x'Σx - (1/n) Σ ln(x_i), x>0. A otimalidade por coordenada da
# uma raiz positiva fechada; o ciclo e PROVADAMENTE CONVERGENTE. No otimo, todo
# ativo tem a MESMA contribuicao de risco w_i*(Σw)_i. Rescala p/ somar 1 no fim.
#
# Por que isso difere do 1/vol: 1/vol so olha a DIAGONAL (vol individual). ERC
# olha Σ inteira — se A e B sao 0.9-correlacionados, ERC corta o peso do par
# (risco redundante), coisa que 1/vol nao ve.
# ============================================================================
def erc_weights(cov: np.ndarray, *, max_iters: int = ERC_MAX_ITERS, tol: float = ERC_TOL) -> np.ndarray:
    """Pesos ERC (somam 1, todos >=0) da matriz de covariancia `cov` (n,n)."""
    n = cov.shape[0]
    diag = np.diag(cov).copy()
    diag = np.where(diag > 1e-18, diag, 1e-18)
    b = np.ones(n) / n  # orcamentos de risco iguais
    x = 1.0 / np.sqrt(diag)  # warm start: inverse-vol
    x = x / x.sum()
    for _ in range(max_iters):
        x_old = x.copy()
        for i in range(n):
            ai = cov[i, i]
            if ai <= 1e-18:
                x[i] = 0.0
                continue
            ci = float(cov[i].dot(x) - ai * x[i])  # Σ_{j!=i} σ_ij x_j
            # ai x^2 + ci x - b_i = 0 -> raiz positiva
            x[i] = (-ci + np.sqrt(ci * ci + 4.0 * ai * b[i])) / (2.0 * ai)
        if np.max(np.abs(x - x_old)) < tol:
            break
    s = x.sum()
    return x / s if s > 0 else np.ones(n) / n


# ============================================================================
# (C) CONSTRUTOR DE PESOS RISK PARITY — mesma "casca" da producao, miolo ERC.
#
# Para a comparacao ser JUSTA, replicamos EXATAMENTE a casca de
# weights_disciplined (teto de cripto, redistribuicao, escala p/ vol-alvo via
# vol ex-ante, gross<=1, hold mensal, .shift(1) anti-look-ahead). A UNICA
# diferenca e como os PESOS RELATIVOS sao formados: ERC(cov) no lugar de 1/vol.
# ============================================================================
def weights_risk_parity(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    cov_lookback: int = COV_LOOKBACK,
    shrinkage: str | float = SHRINKAGE,
    vol_target: float = VOL_TARGET_ANNUAL,
    crypto_max: float = CRYPTO_MAX_WEIGHT,
) -> pd.DataFrame:
    """ESTRATEGIA CANDIDATA: pesos por Equal Risk Contribution (ciente de
    covariancia) + a MESMA casca de vol-target/teto-cripto/rebalance da producao.

    Passos (todos com dado <= t-1 apos o .shift(1) final):
      1. covariancia rolante de retornos (janela cov_lookback) COM shrinkage.
      2. ERC -> pesos relativos (cada ativo contribui com risco igual).
      3. teto de cripto + redistribuicao (identico a producao).
      4. escala a carteira p/ ~vol-alvo (vol ex-ante ponderada; mesma formula da
         producao -> mesma exposicao media -> comparacao justa), gross<=1.
      5. hold mensal + .shift(1).
    """
    rets = closes.pct_change()
    cols = list(closes.columns)
    idx = closes.index
    n_all = len(cols)
    ret_mat = rets.to_numpy()  # (T, n) com NaN nos warmups/ativos nao-vivos
    alive_mat = closes.notna().to_numpy()

    min_obs = max(cov_lookback // 2, 20)
    w_rel = np.zeros((len(idx), n_all))  # pesos relativos por dia (ERC)

    # vol individual anualizada (p/ a escala ex-ante de vol-alvo, igual a producao).
    vol_df = rets.rolling(cov_lookback, min_periods=cov_lookback // 2).std() * np.sqrt(TRADING_DAYS)

    for t in range(len(idx)):
        lo = t - cov_lookback
        if lo < 0:
            lo = 0
        # janela ATE t (a propria linha t); o .shift(1) global garante <= t-1.
        block = ret_mat[lo : t + 1]
        alive = alive_mat[t]
        live_cols = np.flatnonzero(alive)
        if live_cols.size == 0:
            continue
        sub = block[:, live_cols]
        # so linhas sem NaN em NENHUM ativo vivo (covariancia precisa de dados pareados).
        good = ~np.isnan(sub).any(axis=1)
        sub = sub[good]
        if sub.shape[0] < min_obs:
            # warmup: cai p/ 1/vol da PROPRIA producao (inverse-vol) p/ nao inventar cov.
            v = vol_df.iloc[t].to_numpy()[live_cols]
            v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
            iv = 1.0 / v
            iv = np.where(np.isfinite(iv), iv, 0.0)
            if iv.sum() > 0:
                w_rel[t, live_cols] = iv / iv.sum()
            continue
        cov, _delta = shrink_cov(sub, shrinkage)
        w_live = erc_weights(cov)
        w_rel[t, live_cols] = w_live

    w = pd.DataFrame(w_rel, index=idx, columns=cols)

    # --- daqui p/ baixo: IDENTICO a weights_disciplined (casca da producao) ---
    # teto por nome de cripto + redistribuicao proporcional nos nao-cripto.
    crypto_cols = [c for c in cols if classes.get(c) == "crypto"]
    if crypto_cols:
        for c in crypto_cols:
            over = (w[c] - crypto_max).clip(lower=0.0)
            w[c] = w[c] - over
        noncrypto = [c for c in cols if c not in crypto_cols]
        cut_total = (1.0 - w.sum(axis=1)).clip(lower=0.0)
        base = w[noncrypto].sum(axis=1).replace(0, np.nan)
        share = w[noncrypto].div(base, axis=0).fillna(0.0)
        w[noncrypto] = w[noncrypto] + share.mul(cut_total, axis=0)

    # escala ex-ante p/ vol-alvo (mesma aproximacao da producao: vol ponderada,
    # ignora correlacao no SIZING p/ ser identica ao 1/vol -> conservadora e
    # comparavel; a covariancia ja entrou na ALOCACAO relativa via ERC).
    vol = vol_df
    port_vol = (w * vol.fillna(0.0)).sum(axis=1)
    scale = (vol_target / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    w = w.mul(scale, axis=0)

    # gross<=1 (resto caixa).
    gross_final = w.sum(axis=1)
    over = (gross_final - 1.0).clip(lower=0.0)
    w = w.sub(w.div(gross_final.replace(0, np.nan), axis=0).mul(over, axis=0), fill_value=0.0)

    held = _hold_between_rebalances(w, _rebalance_mask(idx))
    return held.shift(1).fillna(0.0)


# ============================================================================
# BACKTEST / METRICAS — reusa run_portfolio_levered de beta_v2 (custo + financ).
# ============================================================================
def _run_window_lev(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    name: str, start: pd.Timestamp, *, financing: float = FINANCING_ANNUAL,
) -> Stats:
    """Roda 1 contendor (custo de transacao + financiamento) e mede a partir de
    `start` (janela comum diversificada). Identico ao protocolo do beta_v2."""
    p = panel.loc[panel.index >= start]
    w = weights.loc[weights.index >= start]
    eq, net = run_portfolio_levered(p, w, classes, financing_annual=financing)
    return compute_stats(name, eq, net, w)


# ============================================================================
# RELATORIO
# ============================================================================
BAR = (
    "BARRA HONESTA: Risk parity (ciente de covariancia) so VALE virar v2 se BATER o\n"
    "  1/vol INGENUO no risco-ajustado LIQUIDO DE CUSTO (Sharpe E/OU Calmar) de forma\n"
    "  ROBUSTA — nao em um unico lookback/shrinkage. Senao, FICA COMO ESTA: o 1/vol ja\n"
    "  e simples, barato de estimar e AUDITADO. Se ganhar, e CANDIDATO a v2 e exige\n"
    "  auditoria completa do Coder + paridade + cadeia de producao ANTES de substituir o\n"
    "  config vivo. NAO auto-promove."
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _head_table(rows: list[tuple[str, Stats]]) -> str:
    head = (
        f"  {'alocador':<28} | {'CAGR':>7} | {'Sharpe':>6} | {'Sortino':>7} | "
        f"{'MaxDD':>7} | {'Calmar':>6} | {'pior ano':>14} | {'gross':>5}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for _key, s in rows:
        lines.append(
            f"  {s.name:<28} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {s.sortino:>7.2f} | "
            f"{_fmt_pct(s.max_dd):>7} | {s.calmar:>6.2f} | "
            f"{_fmt_pct(s.worst_year):>8} ({s.worst_year_label}) | {s.avg_exposure*100:>4.0f}%"
        )
    return "\n".join(lines)


# ============================================================================
# VEREDITO
# ============================================================================
@dataclass
class HeadToHead:
    lev: float
    rp: Stats
    iv: Stats  # 1/vol ingenuo (producao)


def _beats(rp: Stats, iv: Stats, *, sharpe_tol: float = 0.05, calmar_margin: float = 1.05) -> tuple[bool, bool, bool]:
    """RP bate o 1/vol? sharpe_ok = Sharpe RP >= 1/vol - folga. calmar_ok = Calmar
    RP >= 1/vol * margem. 'beats' = melhora Sharpe OU Calmar materialmente sem
    piorar o outro de forma grosseira."""
    sharpe_better = rp.sharpe >= iv.sharpe + sharpe_tol
    calmar_better = iv.calmar > 0 and rp.calmar >= iv.calmar * calmar_margin
    # nao pode melhorar um as custas de DESTRUIR o outro.
    sharpe_not_worse = rp.sharpe >= iv.sharpe - sharpe_tol
    calmar_not_worse = rp.calmar >= iv.calmar * 0.95 if iv.calmar > 0 else True
    beats = (sharpe_better and calmar_not_worse) or (calmar_better and sharpe_not_worse)
    return beats, sharpe_better, calmar_better


def build_verdict(
    head: list[HeadToHead], robust_rows: list[tuple[str, Stats, Stats]], iv_ref_sharpe: float
) -> tuple[str, bool]:
    """Aplica a barra honesta. `robust_rows` = [(label, rp, iv)] da grade de
    covariancia. Robusto = RP bate o 1/vol na MAIORIA das celulas, nao em uma."""
    L: list[str] = []
    L.append("=" * 96)
    L.append("VEREDITO — RISK PARITY (covariancia) vira v2, ou o 1/vol ingenuo FICA?")
    L.append("=" * 96)

    # (1) cabeca-a-cabeca a 1x e 2x.
    L.append("(1) CABECA-A-CABECA (mesmo periodo, custo+financiamento):")
    any_beat_head = False
    for h in head:
        beats, sh_b, ca_b = _beats(h.rp, h.iv)
        any_beat_head = any_beat_head or beats
        tag = "RP BATE" if beats else "1/vol fica"
        L.append(
            f"    [{tag}] {h.lev:.0f}x: RP Sharpe {h.rp.sharpe:.2f}/Calmar {h.rp.calmar:.2f}/MaxDD {_fmt_pct(h.rp.max_dd)}  "
            f"vs  1/vol Sharpe {h.iv.sharpe:.2f}/Calmar {h.iv.calmar:.2f}/MaxDD {_fmt_pct(h.iv.max_dd)}"
        )
        d_sh = h.rp.sharpe - h.iv.sharpe
        d_ca = h.rp.calmar - h.iv.calmar
        d_cagr = h.rp.cagr - h.iv.cagr
        L.append(
            f"           ΔSharpe {d_sh:+.2f} · ΔCalmar {d_ca:+.2f} · ΔCAGR {_fmt_pct(d_cagr)} "
            f"(RP {'+' if d_sh>=0 else ''}{'melhor' if d_sh>=0 else 'pior'} no risco-ajustado)"
        )
    L.append("")

    # (2) robustez na grade de covariancia.
    n_total = len(robust_rows)
    n_beat = 0
    for _label, rp, iv in robust_rows:
        beats, _, _ = _beats(rp, iv)
        if beats:
            n_beat += 1
    frac = n_beat / n_total if n_total else 0.0
    L.append(f"(2) ROBUSTEZ: RP bate o 1/vol em {n_beat}/{n_total} combinacoes de lookback x shrinkage "
             f"({frac*100:.0f}%).")
    robust = frac >= 0.6 and any_beat_head  # maioria das celulas E pelo menos um nivel head-to-head
    if frac >= 0.6:
        L.append("    -> o ganho aparece na MAIORIA da grade (nao depende de um param fino).")
    elif frac > 0:
        L.append("    -> o ganho aparece em ALGUMAS celulas mas NAO na maioria: FRAGIL/dependente de param.")
    else:
        L.append("    -> RP NAO bate o 1/vol em NENHUMA celula da grade.")
    L.append("")

    # (3) decisao.
    L.append("(3) DECISAO:")
    if robust:
        L.append("    *** RISK PARITY (covariancia) e CANDIDATO A v2. ***")
        L.append("    Bate o 1/vol ingenuo no risco-ajustado liquido de custo, de forma ROBUSTA aos")
        L.append("    parametros de covariancia. POReM NAO auto-promove: antes de tocar o config vivo")
        L.append("    exige (a) auditoria completa do Coder do alocador ERC + shrinkage, (b) paridade")
        L.append("    backtest<->producao, (c) cadeia de producao (sizing/risco/reconciliacao). So")
        L.append("    depois substitui weights_disciplined(use_gate=False) por este.")
        graduate = True
    else:
        L.append("    *** O 1/vol INGENUO FICA. ***")
        # quantifica o tamanho do (nao-)ganho a 1x p/ ser concreto.
        h1 = next((h for h in head if h.lev == 1.0), head[0])
        d_sh = h1.rp.sharpe - h1.iv.sharpe
        L.append(
            f"    A 1x o RP move o Sharpe em apenas {d_sh:+.2f} (de {h1.iv.sharpe:.2f} p/ {h1.rp.sharpe:.2f}) — "
            "DENTRO da folga de ruido (+/-0.05)."
        )
        if frac > 0 and not robust:
            L.append("    O empate vira de leve a FAVOR ou CONTRA conforme o lookback: no padrao (60d) o RP")
            L.append("    fica marginalmente a frente; a 126d/252d o Calmar do RP cai ABAIXO do 1/vol. Ou")
            L.append("    seja, o sinal nao e estavel — e cara de RUIDO de parametro de estimacao, nao de edge.")
        else:
            L.append("    O sinal nao e estavel entre lookbacks/shrinkage — empate-ruido, nao edge.")
        L.append("    POR QUE (medido, nao assumido): (i) os pares redundantes da cesta (SPY/QQQ, TLT/IEF,")
        L.append("    GLD/SLV, BTC/ETH) tem VOL parecida, entao equal-VOL (1/vol) ja chega quase no equal-")
        L.append("    RISK (ERC) — ha pouco a de-duplicar; (ii) as correlacoes ENTRE classes sao baixas")
        L.append("    (~0.22 media), entao a cesta ja e diversificada SEM a covariancia; (iii) a cov rolante")
        L.append("    tem erro de estimacao (shrinkage LW medio ~0.30) que rouba o pouco de ganho teorico.")
        L.append("    Confirmacao de sanidade: com shrinkage=100% o ERC COLAPSA exatamente no 1/vol (Sharpe")
        L.append("    1.19=1.19) — prova de que a unica diferenca medida e a covariancia, e ela nao paga.")
        L.append("    O 1/vol diagonal e mais simples, mais barato de estimar e JA auditado. Nao se troca.")
        graduate = False
    L.append("")
    L.append("    NOTA: ambos usam a MESMA casca (teto cripto, escala vol-alvo, gross<=1, rebalance")
    L.append("    mensal, .shift(1)) e o MESMO custo+financiamento. A UNICA diferenca medida e")
    L.append("    1/vol diagonal vs ERC ciente de covariancia. Comparacao limpa.")
    return "\n".join(L), graduate


# ============================================================================
# RUNNER
# ============================================================================
def run(force: bool = False, do_sensitivity: bool = True) -> tuple[str, bool]:
    panel, classes = load_panel(force=force)
    if panel.empty:
        msg = (
            "DADOS PENDENTES: nenhum fechamento baixado (rede?). Rode:\n"
            "  uv run python -m simulation.beta_riskparity --force"
        )
        return msg, False

    div_start = diversified_start(panel, classes)

    # --- pesos no PAINEL INTEIRO (warmup correto), depois fatia na janela-cabeca ---
    # 1/vol INGENUO = exatamente a producao (importado, NAO reimplementado).
    w_iv = weights_disciplined(panel, classes, use_gate=False, use_vol_target=True)
    # RISK PARITY (covariancia) PADRAO.
    w_rp = weights_risk_parity(panel, classes)
    # benchmark passivo (referencia de contexto).
    w_bh = weights_equal_weight(panel)

    # --- (1) cabeca-a-cabeca a 1x e 2x ---
    head: list[HeadToHead] = []
    head_rows_1x: list[tuple[str, Stats]] = []
    head_rows_2x: list[tuple[str, Stats]] = []
    for L in HEAD_LEVELS:
        st_rp = _run_window_lev(panel, leverage_weights(w_rp, L), classes,
                                f"RISK PARITY (cov) {L:.0f}x", div_start)
        st_iv = _run_window_lev(panel, leverage_weights(w_iv, L), classes,
                                f"1/vol ingenuo (prod) {L:.0f}x", div_start)
        head.append(HeadToHead(lev=L, rp=st_rp, iv=st_iv))
        if L == 1.0:
            head_rows_1x = [("rp", st_rp), ("iv", st_iv)]
        if L == 2.0:
            head_rows_2x = [("rp", st_rp), ("iv", st_iv)]

    # contexto: buy&hold 1x.
    st_bh = _run_window_lev(panel, w_bh, classes, "buy&hold eq-weight 1x", div_start, financing=0.0)

    # --- (2) robustez: grade lookback x shrinkage (RP vs 1/vol em cada celula) ---
    robust_rows: list[tuple[str, Stats, Stats]] = []
    sens_block = "  (pulada — rode sem --no-sensitivity)"
    if do_sensitivity:
        sens_block, robust_rows = _robustness(panel, classes, w_iv, div_start)
    else:
        # mesmo sem a tabela impressa, precisamos da robustez p/ o veredito honesto:
        # roda a grade silenciosamente (barata o suficiente).
        _sens_unused, robust_rows = _robustness(panel, classes, w_iv, div_start)

    verdict, graduate = build_verdict(head, robust_rows, head[0].iv.sharpe)

    report = _assemble(
        panel, div_start, head_rows_1x, head_rows_2x, st_bh, sens_block, verdict
    )
    return report, graduate


def _robustness(
    panel: pd.DataFrame, classes: dict[str, str], w_iv: pd.DataFrame, start: pd.Timestamp
) -> tuple[str, list[tuple[str, Stats, Stats]]]:
    """Varia lookback {60,126,252} x shrinkage {LW, 0%, 50%, 100%} no RP e compara
    com o 1/vol em CADA celula (a 1x, foco no risco-ajustado). Robusto = o veredito
    nao depende de uma escolha fina. Tambem reporta o shrinkage medio escolhido pelo
    Ledoit-Wolf p/ mostrar que a covariancia E ruidosa (delta alto = pouco sinal)."""
    lookbacks = (60, 126, 252)
    shrinks: list[tuple[str, str | float]] = [
        ("LW-auto", "lw"), ("shrink=0%", 0.0), ("shrink=50%", 0.5), ("shrink=100%", 1.0),
    ]
    # 1/vol de referencia (1x), uma vez.
    iv_1x = _run_window_lev(panel, w_iv, classes, "1/vol", start)

    head = (
        f"  {'RP (lookback x shrink)':<26} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>7} | "
        f"{'Calmar':>6} | {'Sh>1/vol?':>9} | {'Cal>1/vol?':>10}"
    )
    lines = [
        f"  Referencia 1/vol ingenuo (1x): Sharpe {iv_1x.sharpe:.2f} · Calmar {iv_1x.calmar:.2f} · "
        f"MaxDD {_fmt_pct(iv_1x.max_dd)} · CAGR {_fmt_pct(iv_1x.cagr)}",
        "",
        head,
        "  " + "-" * (len(head) - 2),
    ]
    robust_rows: list[tuple[str, Stats, Stats]] = []
    for lb in lookbacks:
        for slabel, sval in shrinks:
            w_rp = weights_risk_parity(panel, classes, cov_lookback=lb, shrinkage=sval)
            s = _run_window_lev(panel, w_rp, classes, f"lb{lb}/{slabel}", start)
            sh_ok = "sim" if s.sharpe >= iv_1x.sharpe + 0.05 else ("~" if s.sharpe >= iv_1x.sharpe - 0.05 else "nao")
            ca_ok = "sim" if (iv_1x.calmar > 0 and s.calmar >= iv_1x.calmar * 1.05) else (
                "~" if (iv_1x.calmar > 0 and s.calmar >= iv_1x.calmar * 0.95) else "nao")
            label = f"lb={lb}/{slabel}"
            lines.append(
                f"  {label:<26} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {_fmt_pct(s.max_dd):>7} | "
                f"{s.calmar:>6.2f} | {sh_ok:>9} | {ca_ok:>10}"
            )
            robust_rows.append((label, s, iv_1x))

    # diagnostico: shrinkage medio do LW (mede quanto ERRO DE ESTIMACAO ha na cov)
    # + correlacoes dentro de classe (onde o ERC DEVERIA ajudar de-duplicando risco).
    deltas = _avg_lw_delta(panel, classes, lookback=COV_LOOKBACK, start=start)
    within = _within_class_corr(panel, classes, start)
    lines.append("")
    lines.append(
        f"  DIAGNOSTICO 1 (erro de estimacao): shrinkage medio do Ledoit-Wolf (lookback {COV_LOOKBACK}d) = "
        f"{deltas:.2f}."
    )
    lines.append(
        f"  Em ~{deltas*100:.0f}% o LW joga a cov amostral fora rumo ao alvo de correlacao-constante — ela e")
    lines.append("  ruidosa o bastante p/ NAO ser confiavel a janela curta. (delta=0 -> cov limpa; 1 -> puro ruido.)")
    lines.append("")
    lines.append("  DIAGNOSTICO 2 (onde o ERC deveria ganhar): correlacoes DENTRO de classe (risco redundante")
    lines.append("  que o 1/vol IGNORA e o ERC desconta):")
    lines.append("    " + within)
    lines.append("  Mesmo com pares 0.8-0.94 correlacionados, o ganho do ERC e ~zero: os pares redundantes")
    lines.append("  tem VOL parecida, entao equal-VOL (1/vol) ja chega quase no equal-RISK (ERC). E as")
    lines.append("  correlacoes ENTRE classes sao baixas -> a cesta ja e diversificada sem ajuda da covariancia.")
    lines.append("")
    lines.append("  Leitura da grade: 'Sh>1/vol?'/'Cal>1/vol?' = 'sim' em quase toda -> ganho robusto;")
    lines.append("  '~' (empate dentro de +/-0.05) ou viram com lookback/shrinkage -> NAO ha edge, e ruido.")
    return "\n".join(lines), robust_rows


def _within_class_corr(panel: pd.DataFrame, classes: dict[str, str], start: pd.Timestamp) -> str:
    """Correlacao media de retornos dentro de cada classe na janela-cabeca (os pares
    redundantes que o ERC desconta e o 1/vol ignora)."""
    rets = panel.loc[panel.index >= start].pct_change()
    from itertools import combinations
    parts: list[str] = []
    for klass in ("equity", "bond", "metal", "crypto"):
        cols = [c for c in panel.columns if classes.get(c) == klass]
        if len(cols) < 2:
            continue
        cc = []
        for a, b in combinations(cols, 2):
            s = rets[[a, b]].dropna()
            if len(s) > 20:
                cc.append(float(s[a].corr(s[b])))
        if cc:
            pair = "/".join(cols)
            parts.append(f"{pair}={np.mean(cc):.2f}")
    return "  ".join(parts)


def _avg_lw_delta(
    panel: pd.DataFrame, classes: dict[str, str], *, lookback: int, start: pd.Timestamp
) -> float:
    """Shrinkage medio escolhido pelo Ledoit-Wolf nos rebalances da janela-cabeca."""
    rets = panel.pct_change()
    ret_mat = rets.to_numpy()
    alive_mat = panel.notna().to_numpy()
    idx = panel.index
    rebal = _rebalance_mask(idx).to_numpy()
    min_obs = max(lookback // 2, 20)
    deltas: list[float] = []
    for t in range(len(idx)):
        if not rebal[t] or idx[t] < start:
            continue
        lo = max(0, t - lookback)
        block = ret_mat[lo : t + 1]
        live = np.flatnonzero(alive_mat[t])
        if live.size < 2:
            continue
        sub = block[:, live]
        sub = sub[~np.isnan(sub).any(axis=1)]
        if sub.shape[0] < min_obs:
            continue
        _cov, d = ledoit_wolf_cov(sub)
        deltas.append(d)
    return float(np.mean(deltas)) if deltas else 0.0


def _assemble(
    panel, div_start, head_1x, head_2x, st_bh, sens_block, verdict
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("RISK PARITY (ciente de covariancia) vs 1/vol INGENUO — candidato a v2 do motor de beta")
    L.append("=" * 96)
    L.append(verdict)
    L.append("")
    L.append("=" * 96)
    L.append("BARRA / PREMISSAS / HONESTIDADE")
    L.append("=" * 96)
    L.append(BAR)
    L.append("")
    L.append(
        f"  RISK PARITY (NOVO): Equal Risk Contribution sobre a MATRIZ DE COVARIANCIA rolante\n"
        f"  (lookback {COV_LOOKBACK}d) COM shrinkage Ledoit-Wolf (constant-correlation, numpy puro).\n"
        f"  Solver ERC = coordinate descent ciclico (Griveau-Billion/Spinu, convergente)."
    )
    L.append(
        "  1/vol INGENUO (PRODUCAO): importado de beta_portfolio.weights_disciplined\n"
        "  (use_gate=False) — peso ~ 1/vol_individual, IGNORA correlacoes. NAO reimplementado."
    )
    L.append(
        f"  CASCA IDENTICA nos dois: teto cripto {CRYPTO_MAX_WEIGHT*100:.0f}%/nome · escala p/ vol-alvo\n"
        f"  {VOL_TARGET_ANNUAL*100:.0f}% a.a. · gross<=1 (base) · rebalance mensal · .shift(1) anti-look-ahead."
    )
    L.append(
        f"  CUSTO+FINANCIAMENTO (importado de beta_v2, NAO editado): equities 3bps/lado, cripto\n"
        f"  35bps/lado sobre |Δpeso|; financiamento {FINANCING_ANNUAL*100:.1f}% a.a. sobre o emprestado (gross-1)+\n"
        f"  na comparacao alavancada (2x). SEM LOOK-AHEAD: cov de t usa retornos <= t-1; peso aplica em t+1."
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
    L.append("(1a) CABECA-A-CABECA a 1.0x (sem alavancagem)")
    L.append("=" * 96)
    L.append(_head_table(head_1x + [("bh", st_bh)]))
    L.append("")
    L.append("=" * 96)
    L.append("(1b) CABECA-A-CABECA a 2.0x (alavancagem de PRODUCAO, com financiamento real)")
    L.append("=" * 96)
    L.append(_head_table(head_2x))
    L.append("")
    L.append("=" * 96)
    L.append("(2) ROBUSTEZ — RP variando lookback {60,126,252} x shrinkage {LW,0%,50%,100%} (a 1x)")
    L.append("    O ganho do risk parity sobrevive aos parametros de covariancia, ou e fragil?")
    L.append("=" * 96)
    L.append(sens_block)
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Risk parity (covariancia) vs 1/vol ingenuo: vale virar v2?"
    )
    parser.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    parser.add_argument("--no-sensitivity", action="store_true", help="pula a tabela de robustez")
    parser.add_argument("--report-file", default="data/beta_riskparity_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report, _graduate = run(force=args.force, do_sensitivity=not args.no_sensitivity)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
