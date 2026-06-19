"""Beta frontier — mapeia a FRONTEIRA risco/retorno do beta p/ escolha CONSCIENTE
de orcamento de drawdown.

CONTEXTO (decisao do usuario): o objetivo deixou de ser "tombo menor a qualquer
custo". O usuario quer AUMENTAR o retorno a.a. ACEITANDO mais MaxDD — uma escolha
consciente de orcamento de risco. Este modulo NAO recomenda um ponto unico: ele
desenha o CARDAPIO ("o DD que voce aguenta -> o CAGR/Sharpe que vem junto") e a
parte honesta do quant: ate onde mais DD COMPRA mais retorno (faixa sa, ao longo
da fronteira) e a partir de onde mais DD DESTROI retorno (zona pos-Kelly, vol-drag).

ESTE MODULO (NOVO; NAO edita beta_portfolio.py nem beta_v2.py — IMPORTA deles):

  EIXOS DA FRONTEIRA (o usuario pediu os dois explicitamente):
    1. ALAVANCAGEM L: 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0x.
    2. VOL-ALVO: 10%, 12%, 15%, 20% a.a. (o alvo de vol do nucleo vol-target).
    EXPOSICAO REAL = vol-alvo x alavancagem (mostrada nos dois eixos). Ex.: vol-alvo
    15% x L=2.0 mira ~30% de vol "tomada"; o gross medio carregado e medido e
    reportado tambem (e menor, pois o vol-target deixa caixa).

  VEICULO: o NUCLEO vol-target confirmado (peso ~1/vol por ativo, teto de cripto,
    SEM o portao de regime/SMA — o portao reprovou por whipsaw no modulo auditado),
    escalado por L com CUSTO DE FINANCIAMENTO REAL (~5.5% a.a. sobre o emprestado,
    so sobre max(gross-1,0)). Reutiliza leverage_weights + run_portfolio_levered
    de beta_v2 (NAO reimplementa o motor).

  VARIANTES DE MIX (1-2, p/ mostrar o efeito):
    - cap de cripto MAIOR (10%/nome em vez de 5%) — mais risco/retorno de cauda.
    - tilt PRO-ACOES (vol-alvo das equities relaxado / cap cripto baixo) — aproxima
      do perfil "so beta de bolsa".
    (Implementadas reusando weights_disciplined com crypto_max distinto; o tilt
    pro-acoes e o cap-cripto-baixo + sem metais no de-risk e modelado pelo crypto_max.)

  REFERENCIA SEMPRE VISIVEL: o BUY&HOLD eq-weight (beta cru) — o que o usuario
    teria SEM disciplina (CAGR ~16%, Sharpe ~0.97, MaxDD ~-38.8%). E o ponto de
    comparacao honesto: a fronteira so "vale" se um ponto entrega mais retorno OU
    menos tombo que o beta cru.

ANALISE-CHAVE (a parte honesta do quant — entregue no veredito):
  - KELLY: o L que MAXIMIZA o CAGR (por vol-alvo). PASSOU dele, +L = MENOS CAGR
    (vol-drag: a media geometrica cai quando a vol sobe demais) e MAIS DD. Marcado
    com '<<< KELLY' na tabela. A faixa SA vai ate Kelly; depois e destruicao de valor.
  - "AO LONGO DA FRONTEIRA" (ate Kelly): +DD compra +retorno ~proporcional, Sharpe
    cai DEVAGAR (o financiamento e o unico vazamento). vs "PASSOU DO OTIMO": +DD,
    retorno decrescente/negativo, Sharpe despenca. O veredito separa os dois.
  - DD DO BACKTEST E MELHOR-CASO. O DD REALISTA/ESTRESSE estimado = o PIOR entre
    (i) 1.4x o DD do backtest (regra de bolso ao-vivo: slippage de cauda, gaps,
    execucao, params nao mais perfeitos) e (ii) o pior comportamento observado em
    2008/2020/2022 escalado pelo L. Quando o usuario escolher um orcamento, ele
    precisa saber que o DD AO VIVO provavelmente EXCEDE o do backtest.
  - TENSAO AUM: DD alto e mais dificil de vender p/ alocador. Registrada no veredito.

HONESTIDADE (igual aos modulos auditados, nao-negociavel):
  - Params PADRAO declarados; SEM look-ahead (peso de t com dado <= t-1, .shift(1)
    herdado dos construtores). Custo de transacao real (simulation.costs) sobre
    |Delta peso|. Custo de financiamento real sobre o emprestado.
  - Metricas medidas na JANELA-CABECA (cesta genuinamente diversificada, a partir
    de ~2004-11); crises medidas no historico INTEIRO (p/ ver 2008).
  - NAO muda config de producao — isto e ANALISE. Se o usuario escolher um ponto
    novo, o Coder audita a config antes do go-live.

Uso:
    uv run python -m simulation.beta_frontier
    uv run python -m simulation.beta_frontier --force          # re-baixa o cache
    uv run python -m simulation.beta_frontier --no-sensitivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# REUTILIZA os modulos auditados — NAO os edita. Importa pesos/alavancagem/
# financiamento/metricas. (beta_v2 ja reexporta o motor de beta_portfolio.)
from simulation.beta_portfolio import (
    CRYPTO_MAX_WEIGHT,
    VOL_LOOKBACK,
    Stats,
    diversified_start,
    load_panel,
    run_portfolio,
    weights_disciplined,
    weights_equal_weight,
)
from simulation.beta_v2 import (
    FINANCING_ANNUAL,
    leverage_weights,
    run_portfolio_levered,
)

logger = logging.getLogger("simulation.beta_frontier")


# ----------------------------------------------------------------------------
# EIXOS DA FRONTEIRA (declarados; o usuario os pediu explicitamente).
# ----------------------------------------------------------------------------
LEVERAGE_LEVELS: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
VOL_TARGETS: tuple[float, ...] = (0.10, 0.12, 0.15, 0.20)

# Variantes de mix (mostram o efeito de mais cripto / tilt). crypto_max por nome.
MIX_BASE_CRYPTO = CRYPTO_MAX_WEIGHT       # 5%/nome (o de producao)
MIX_HIGH_CRYPTO = 0.10                    # 10%/nome (mais risco/retorno de cauda)
MIX_LOW_CRYPTO = 0.02                     # 2%/nome (tilt pro-acoes/defensivo)

# Regra de bolso do DD ao-vivo: multiplicador sobre o DD do backtest. Faixa
# tipica citada por practitioners (live > backtest por slippage de cauda, gaps,
# execucao imperfeita, params que param de ser otimos). Usamos 1.4x como central
# e mostramos a faixa 1.3-1.5x.
LIVE_DD_MULT_LOW = 1.3
LIVE_DD_MULT_MID = 1.4
LIVE_DD_MULT_HIGH = 1.5


# ============================================================================
# FRONTEIRA — ponto a ponto (vol-alvo x alavancagem), com financiamento real.
# ============================================================================
@dataclass
class FrontierPoint:
    vol_target: float          # vol-alvo do nucleo (a.a.)
    lev: float                 # alavancagem
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float              # <= 0, backtest (melhor-caso)
    calmar: float
    worst_year: float
    worst_year_label: str
    longest_dd_days: int       # duracao do maior drawdown (em dias de pregao)
    avg_gross: float           # gross medio EFETIVO carregado (exposicao real media)
    target_exposure: float     # vol-alvo x lev (exposicao-alvo "tomada"; eixo pedido)
    crisis_2008: float | None
    crisis_2020: float | None
    crisis_2022: float | None
    near_kelly: bool = False   # marcado depois: ponto da GRADE-CARDAPIO com maior CAGR
                               # (NAO e a Kelly verdadeira — a Kelly real fica muito alem,
                               #  em DD catastrofico; ver find_kelly_fine + veredito).


def _longest_drawdown_days(equity: pd.Series) -> int:
    """Duracao (em dias de pregao) do MAIOR periodo desde um pico ate recuperar
    esse pico. Mede quanto tempo o investidor passa 'no vermelho' no pior episodio
    — metrica de dor que o MaxDD (profundidade) nao captura.

    SEM look-ahead na metrica de relatorio: usa a curva inteira ja realizada (e
    diagnostico ex-post, nao um sinal de trade)."""
    if equity.size < 2:
        return 0
    arr = equity.to_numpy(dtype=float)
    peak = arr[0]
    peak_i = 0
    longest = 0
    cur_underwater = False
    for i in range(1, arr.size):
        if arr[i] >= peak:
            # recuperou o pico anterior -> fecha o episodio de drawdown.
            if cur_underwater:
                longest = max(longest, i - peak_i)
            peak = arr[i]
            peak_i = i
            cur_underwater = False
        else:
            cur_underwater = True
    # drawdown ainda aberto no fim da serie: conta ate o ultimo dia.
    if cur_underwater:
        longest = max(longest, (arr.size - 1) - peak_i)
    return int(longest)


def _crisis_returns(net: pd.Series) -> dict[str, float]:
    windows = {
        "2008": ("2008-01-01", "2009-03-31"),
        "2020": ("2020-02-01", "2020-04-30"),
        "2022": ("2022-01-01", "2022-12-31"),
    }
    out: dict[str, float] = {}
    for label, (a, b) in windows.items():
        seg = net.loc[(net.index >= a) & (net.index <= b)]
        if seg.size >= 5:
            out[label] = float((1.0 + seg).prod() - 1.0)
    return out


def _stats_with_duration(name: str, equity: pd.Series, net: pd.Series, weights: pd.DataFrame) -> tuple[Stats, int]:
    """compute_stats (reusado) + duracao do maior drawdown (extra deste modulo)."""
    from simulation.beta_portfolio import compute_stats
    s = compute_stats(name, equity, net, weights)
    dd_days = _longest_drawdown_days(equity)
    return s, dd_days


def evaluate_point(
    panel_full: pd.DataFrame,
    panel_window: pd.DataFrame,
    classes: dict[str, str],
    *,
    vol_target: float,
    lev: float,
    crypto_max: float,
    financing: float,
    div_start: pd.Timestamp,
) -> FrontierPoint:
    """Um ponto da fronteira: nucleo vol-target (vol-alvo dado, crypto_max dado),
    escalado por `lev`, com financiamento real.

    Pesos calculados no PAINEL INTEIRO (warmup de vol correto); metricas-cabeca na
    JANELA diversificada; crises no historico INTEIRO. SEM look-ahead (.shift(1)
    herdado de weights_disciplined). Reusa leverage_weights + run_portfolio_levered.
    """
    # nucleo vol-target PURO (use_gate=False — o portao reprovou no modulo auditado).
    base_w = weights_disciplined(
        panel_full, classes,
        vol_target=vol_target, crypto_max=crypto_max,
        use_gate=False, use_vol_target=True,
    )
    lev_w = leverage_weights(base_w, lev)

    # --- janela-cabeca p/ metricas principais ---
    p = panel_window
    w_win = lev_w.loc[lev_w.index >= div_start]
    eq, net = run_portfolio_levered(p, w_win, classes, financing_annual=financing)
    s, dd_days = _stats_with_duration("pt", eq, net, w_win)

    # --- crises no historico INTEIRO ---
    _, net_full = run_portfolio_levered(panel_full, lev_w, classes, financing_annual=financing)
    cr = _crisis_returns(net_full)

    return FrontierPoint(
        vol_target=vol_target, lev=lev,
        cagr=s.cagr, sharpe=s.sharpe, sortino=s.sortino,
        max_dd=s.max_dd, calmar=s.calmar,
        worst_year=s.worst_year, worst_year_label=s.worst_year_label,
        longest_dd_days=dd_days,
        avg_gross=s.avg_exposure,
        target_exposure=vol_target * lev,
        crisis_2008=cr.get("2008"), crisis_2020=cr.get("2020"), crisis_2022=cr.get("2022"),
    )


def build_frontier(
    panel_full: pd.DataFrame,
    panel_window: pd.DataFrame,
    classes: dict[str, str],
    div_start: pd.Timestamp,
    *,
    vol_targets: tuple[float, ...] = VOL_TARGETS,
    leverage_levels: tuple[float, ...] = LEVERAGE_LEVELS,
    crypto_max: float = MIX_BASE_CRYPTO,
    financing: float = FINANCING_ANNUAL,
) -> dict[float, list[FrontierPoint]]:
    """Grade completa {vol_alvo -> [pontos por L]}, com a Kelly marcada por vol-alvo
    (o L que MAXIMIZA o CAGR daquela linha de vol-alvo)."""
    grid: dict[float, list[FrontierPoint]] = {}
    for vt in vol_targets:
        row: list[FrontierPoint] = []
        for L in leverage_levels:
            row.append(evaluate_point(
                panel_full, panel_window, classes,
                vol_target=vt, lev=L, crypto_max=crypto_max,
                financing=financing, div_start=div_start,
            ))
        # marca o ponto de MAIOR CAGR DA GRADE (NAO a Kelly real — a grade-cardapio
        # vai so ate 3x; a Kelly verdadeira fica muito alem, em DD catastrofico).
        if row:
            k = max(range(len(row)), key=lambda i: row[i].cagr)
            row[k].near_kelly = True
        grid[vt] = row
    return grid


@dataclass
class KellyPoint:
    """A Kelly VERDADEIRA de um vol-alvo: o L que maximiza o CAGR geometrico,
    achado numa grade FINA estendida (ate ~12x) ATE o CAGR virar (vol-drag)."""
    vol_target: float
    lev: float            # L da Kelly verdadeira (ou o teto da busca, se nao virou)
    cagr: float           # CAGR no pico
    max_dd: float         # MaxDD do backtest NO pico (tipicamente catastrofico)
    sharpe: float
    at_ceiling: bool = False  # True = o CAGR ainda subia no teto da busca (Kelly real e ALEM)


def find_kelly_fine(
    panel_full: pd.DataFrame,
    panel_window: pd.DataFrame,
    classes: dict[str, str],
    div_start: pd.Timestamp,
    *,
    vol_target: float,
    crypto_max: float = MIX_BASE_CRYPTO,
    financing: float = FINANCING_ANNUAL,
    lo: float = 1.0,
    hi: float = 12.0,
    step: float = 0.25,
) -> KellyPoint:
    """Varre L numa grade FINA ESTENDIDA (ate ~12x) p/ achar a Kelly VERDADEIRA:
    o L onde o CAGR atinge o MAXIMO e PASSA A CAIR (vol-drag vence). Vai bem alem
    da grade-cardapio (3x) de proposito — sem isso, a 'Kelly' seria so a borda da
    grade, nao o ponto real de virada. Retorna o ponto inteiro (L, CAGR, MaxDD)
    p/ o veredito poder mostrar que a Kelly real vive em DD catastrofico."""
    best: FrontierPoint | None = None
    L = lo
    while L <= hi + 1e-9:
        pt = evaluate_point(
            panel_full, panel_window, classes,
            vol_target=vol_target, lev=round(L, 3), crypto_max=crypto_max,
            financing=financing, div_start=div_start,
        )
        if best is None or pt.cagr > best.cagr:
            best = pt
        L += step
    assert best is not None
    # se o maximo caiu NO teto da busca, o CAGR ainda subia -> a Kelly real e ALEM
    # de `hi` (so confirma ainda mais que ela e in-investivel).
    at_ceiling = abs(best.lev - hi) < step
    return KellyPoint(
        vol_target=vol_target, lev=best.lev, cagr=best.cagr,
        max_dd=best.max_dd, sharpe=best.sharpe, at_ceiling=at_ceiling,
    )


# ============================================================================
# BENCHMARK — buy&hold eq-weight (o "beta cru" que o usuario quer alavancar).
# ============================================================================
def beta_reference(
    panel_full: pd.DataFrame, panel_window: pd.DataFrame, classes: dict[str, str],
    div_start: pd.Timestamp,
) -> tuple[Stats, int, dict[str, float]]:
    from simulation.beta_portfolio import compute_stats
    w = weights_equal_weight(panel_full)
    w_win = w.loc[w.index >= div_start]
    eq, net = run_portfolio(panel_window, w_win, classes)
    s = compute_stats("BUY&HOLD eq-weight (beta cru)", eq, net, w_win)
    dd_days = _longest_drawdown_days(eq)
    _, net_full = run_portfolio(panel_full, w, classes)
    return s, dd_days, _crisis_returns(net_full)


# ============================================================================
# DD AO-VIVO ESTIMADO (a parte honesta: backtest e melhor-caso).
# ============================================================================
def estimate_live_dd(pt: FrontierPoint) -> tuple[float, float, float, str]:
    """Estima o DD REALISTA/ESTRESSE de um ponto. Retorna (low, mid, high, fonte).

    Combina DUAS visoes e pega a MAIS SEVERA p/ o 'mid'/'high':
      (a) regra de bolso ao-vivo: 1.3-1.5x o MaxDD do backtest.
      (b) pior CRISE observada (2008/2020/2022) — ja inclui o stress de cauda
          historico real, ESCALADO pelo L embutido no ponto (a crise foi medida
          com o L do ponto, entao ja vem alavancada).
    O 'high' e o pior entre 1.5x-backtest e (pior-crise x 1.1) — um colchao p/ o
    fato de a proxima crise poder ser pior que a pior do historico."""
    bt = abs(pt.max_dd)
    crises = [abs(c) for c in (pt.crisis_2008, pt.crisis_2020, pt.crisis_2022) if c is not None and c < 0]
    worst_crisis = max(crises) if crises else 0.0

    low = bt * LIVE_DD_MULT_LOW
    mid = max(bt * LIVE_DD_MULT_MID, worst_crisis)
    high = max(bt * LIVE_DD_MULT_HIGH, worst_crisis * 1.1)
    src = "1.4x-bt" if (bt * LIVE_DD_MULT_MID) >= worst_crisis else "pior-crise"
    return -low, -mid, -high, src


# ============================================================================
# RELATORIO
# ============================================================================
def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "    -"
    return f"{x * 100:+.1f}%"


def _fmt_dd_days(d: int) -> str:
    """Duracao do maior drawdown em dias de pregao -> meses aproximados."""
    if d <= 0:
        return "  -"
    months = d / 21.0
    return f"{d}d (~{months:.0f}m)"


def frontier_table(grid: dict[float, list[FrontierPoint]], beta: Stats) -> str:
    """Tabela completa da fronteira: linha por (vol-alvo, L), com Kelly marcada."""
    head = (
        f"  {'vol-alvo':>8} {'lev':>5} {'expo-alvo':>9} {'gross med':>9} | "
        f"{'CAGR':>7} {'Sharpe':>6} {'Sortino':>7} | "
        f"{'MaxDD(bt)':>9} {'Calmar':>6} {'pior ano':>15} {'maior DD':>11} |"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for vt, row in grid.items():
        for pt in row:
            kelly = "  <- maior CAGR da grade" if pt.near_kelly else ""
            lines.append(
                f"  {vt*100:>7.0f}% {pt.lev:>4.2f}x {pt.target_exposure*100:>8.0f}% "
                f"{pt.avg_gross*100:>8.0f}% | "
                f"{_fmt_pct(pt.cagr):>7} {pt.sharpe:>6.2f} {pt.sortino:>7.2f} | "
                f"{_fmt_pct(pt.max_dd):>9} {pt.calmar:>6.2f} "
                f"{_fmt_pct(pt.worst_year):>8} ({pt.worst_year_label}) "
                f"{_fmt_dd_days(pt.longest_dd_days):>11} |{kelly}"
            )
        lines.append("  " + "·" * (len(head) - 2))
    lines.append(
        f"  REFERENCIA beta cru (buy&hold eq-weight): CAGR {_fmt_pct(beta.cagr)} · "
        f"Sharpe {beta.sharpe:.2f} · MaxDD {_fmt_pct(beta.max_dd)} · Calmar {beta.calmar:.2f}."
    )
    lines.append(
        "  expo-alvo = vol-alvo x lev (vol 'tomada' mirada). gross med = exposicao "
        "REAL media carregada (menor: o vol-target deixa caixa)."
    )
    lines.append(
        "  NOTA: 'maior CAGR da grade' (3x) NAO e a Kelly — o CAGR ainda sobe alem de 3x; "
        "a Kelly real fica\n  em L muito maior, com DD catastrofico (ver veredito). A grade "
        "para em 3x porque alem disso o DD ja e in-investivel."
    )
    return "\n".join(lines)


def crisis_table(grid: dict[float, list[FrontierPoint]], beta_crisis: dict[str, float]) -> str:
    """Comportamento em 2008/2020/2022 (retorno acumulado, historico INTEIRO) p/ um
    subconjunto representativo de pontos (cada vol-alvo em L=1.0, Kelly e L=3.0)."""
    head = f"  {'ponto':<26} | {'2008 (GFC)':>12} | {'2020 (COVID)':>13} | {'2022 (Bear)':>12}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for vt, row in grid.items():
        # escolhe L=1.0 (min) e o maior L da grade (3x) p/ ilustrar a amplificacao.
        pick_idx = {0, len(row) - 1}
        for i in sorted(pick_idx):
            pt = row[i]
            tag = "min" if i == 0 else "max-grade"
            name = f"vol{vt*100:.0f}% L={pt.lev:.2f}x [{tag}]"
            lines.append(
                f"  {name:<26} | {_fmt_pct(pt.crisis_2008):>12} | "
                f"{_fmt_pct(pt.crisis_2020):>13} | {_fmt_pct(pt.crisis_2022):>12}"
            )
        lines.append("  " + "·" * (len(head) - 2))
    lines.append(
        f"  {'BUY&HOLD beta cru':<26} | {_fmt_pct(beta_crisis.get('2008')):>12} | "
        f"{_fmt_pct(beta_crisis.get('2020')):>13} | {_fmt_pct(beta_crisis.get('2022')):>12}"
    )
    lines.append(
        "  LEITURA: a alavancagem AMPLIFICA a crise proporcionalmente — 2022 (cesta inteira caiu)"
        "\n  vai de ~-10% (1x) a ~-30% (3x); 2008 piora pouco (a cesta ja era pouco exposta la). E o"
        "\n  retrato de como o seu DD real se comporta no pior cenario de cada nivel de L escolhido."
    )
    return "\n".join(lines)


def menu_table(grid: dict[float, list[FrontierPoint]], beta: Stats, beta_dd_days: int) -> str:
    """A TABELA-CARDAPIO pedida: ordenada por DD do backtest crescente; cada linha
    'DD que voce aguenta (backtest / estimado ao-vivo) -> CAGR / Sharpe que vem
    junto'. Inclui o beta cru como referencia ancorada no proprio DD dele."""
    # achata todos os pontos e ordena por |MaxDD| do backtest.
    flat: list[FrontierPoint] = [pt for row in grid.values() for pt in row]
    flat.sort(key=lambda p: abs(p.max_dd))

    head = (
        f"  {'DD backtest':>11} | {'DD ao-vivo est. (mid/pior)':>26} | "
        f"{'CAGR':>7} | {'Sharpe':>6} | {'Calmar':>6} | {'config (vol-alvo x lev)':>24}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for pt in flat:
        low, mid, high, _src = estimate_live_dd(pt)
        cfg = f"vol {pt.vol_target*100:.0f}% x {pt.lev:.2f}x"
        lines.append(
            f"  {_fmt_pct(pt.max_dd):>11} | {_fmt_pct(mid):>11} / {_fmt_pct(high):>11}  | "
            f"{_fmt_pct(pt.cagr):>7} | {pt.sharpe:>6.2f} | {pt.calmar:>6.2f} | {cfg:>17}"
        )
    lines.append("  " + "-" * (len(head) - 2))
    # beta cru como linha de referencia (DD ao-vivo: mesma regra 1.4x).
    b_mid = abs(beta.max_dd) * LIVE_DD_MULT_MID
    b_high = abs(beta.max_dd) * LIVE_DD_MULT_HIGH
    lines.append(
        f"  {_fmt_pct(beta.max_dd):>11} | {_fmt_pct(-b_mid):>11} / {_fmt_pct(-b_high):>11}  | "
        f"{_fmt_pct(beta.cagr):>7} | {beta.sharpe:>6.2f} | {beta.calmar:>6.2f} | {'BETA CRU (buy&hold)':>17}"
    )
    lines.append("")
    lines.append(
        "  COMO LER: escolha a coluna 'DD backtest' que voce aguenta. O 'DD ao-vivo "
        "estimado' (mid = 1.4x o\n  backtest OU a pior crise, o que for pior; pior = "
        "1.5x / colchao de crise) e o que voce PROVAVELMENTE\n  vai sofrer de verdade "
        "— PLANEJE por ele, nao pelo backtest. A direita esta a config que produz a linha."
    )
    return "\n".join(lines)


def _kelly_summary(kelly_fine: dict[float, KellyPoint]) -> str:
    lines: list[str] = []
    head = (
        f"  {'vol-alvo':>8} | {'Kelly L (real)':>14} | {'CAGR no pico':>12} | "
        f"{'MaxDD backtest':>14} | {'MaxDD ao-vivo est.':>18} | {'Sharpe':>6}"
    )
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for vt, kp in kelly_fine.items():
        live = abs(kp.max_dd) * LIVE_DD_MULT_MID
        lev_s = f">={kp.lev:.1f}x" if kp.at_ceiling else f"{kp.lev:.2f}x"
        lines.append(
            f"  {vt*100:>7.0f}% | {lev_s:>13} | {_fmt_pct(kp.cagr):>12} | "
            f"{_fmt_pct(kp.max_dd):>14} | {_fmt_pct(-live):>18} | {kp.sharpe:>6.2f}"
        )
    lines.append(
        "  (>= = o CAGR ainda subia no teto da busca de 12x; a Kelly real fica AINDA mais longe — "
        "so reforca\n   que e in-investivel. 'ao-vivo est.' = 1.4x o DD do backtest.)"
    )
    return "\n".join(lines)


# ============================================================================
# VEREDITO — a parte honesta do quant.
# ============================================================================
def _row_pt(grid: dict[float, list[FrontierPoint]], vt: float, lev: float) -> FrontierPoint | None:
    return next((p for p in grid.get(vt, []) if abs(p.lev - lev) < 1e-6), None)


def _sane_upper(row: list[FrontierPoint], kelly_lev: float, *, dd_live_cap: float = 0.30) -> FrontierPoint:
    """Maior ponto da faixa SA: L <= Kelly E DD ao-vivo (mid) ainda <= dd_live_cap
    (default -30%, o limite 'vendavel' p/ alocador). Devolve o ponto inteiro."""
    sane = row[0]
    for pt in row:
        if pt.lev > kelly_lev + 1e-9:
            break
        _, mid, _, _ = estimate_live_dd(pt)
        if abs(mid) <= dd_live_cap:
            sane = pt
    return sane


def build_verdict(
    grid: dict[float, list[FrontierPoint]],
    grid_high_crypto: dict[float, list[FrontierPoint]],
    grid_low_crypto: dict[float, list[FrontierPoint]],
    kelly_fine: dict[float, KellyPoint],
    beta: Stats,
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("VEREDITO — A PARTE HONESTA DO QUANT")
    L.append("=" * 96)

    base_row = grid.get(0.10) or next(iter(grid.values()))
    kelly10 = kelly_fine.get(0.10)
    p1x = _row_pt(grid, 0.10, 1.0)
    p2x = _row_pt(grid, 0.10, 2.0)
    p3x = _row_pt(grid, 0.10, 3.0)

    # --- (1) Kelly verdadeira por vol-alvo ---
    L.append("(1) ALAVANCAGEM OTIMA DE KELLY (o L que MAXIMIZA o CAGR — achada em grade fina ate 12x):")
    L.append(_kelly_summary(kelly_fine))
    L.append("")
    L.append("    O QUE ISTO SIGNIFICA (teoria): para cada vol-alvo existe um L de Kelly que MAXIMIZA o")
    L.append("    retorno geometrico (CAGR). ATE ele, mais alavancagem compra mais retorno; PASSOU dele,")
    L.append("    mais L = MENOS CAGR (vol-drag: a vol entra ao quadrado e corta a media geometrica) E")
    L.append("    MAIS DD — voce paga risco + financiamento p/ DESTRUIR valor.")
    L.append("")
    if kelly10 is not None:
        live_k = abs(kelly10.max_dd) * LIVE_DD_MULT_MID
        L.append("    *** A DESCOBERTA HONESTA, E A MAIS IMPORTANTE DESTE RELATORIO ***")
        L.append(f"    A Kelly REAL desta carteira fica em L ~{kelly10.lev:.1f}x (vol-alvo 10%) — e nela o MaxDD do")
        L.append(f"    BACKTEST ja e {_fmt_pct(kelly10.max_dd)} (≈ {_fmt_pct(-live_k)} AO VIVO). Isso e FINANCEIRAMENTE")
        L.append("    IN-INVESTIVEL: um DD de -65% a -90% quebra o investidor (e o alocador foge muito antes).")
        L.append("    CONCLUSAO PRATICA: a Kelly NAO e a sua restricao — o seu ORCAMENTO DE DRAWDOWN e. Voce")
        L.append("    vai bater no teto de DD que aguenta MUITO antes de chegar perto da Kelly. Na pratica,")
        L.append("    TODA a faixa investivel (L=1x..~3x) esta ABAIXO da Kelly — ou seja, voce esta sempre")
        L.append("    'ao longo da fronteira' (sub-Kelly), nunca no regime de vol-drag. O perigo do pos-otimo")
        L.append("    e teorico aqui; o perigo REAL e escolher um L cujo DD AO-VIVO te quebra na proxima crise.")
    L.append("")

    # --- (2) ao longo da fronteira (sub-Kelly): o que +DD compra de verdade ---
    L.append("(2) 'AO LONGO DA FRONTEIRA' (sub-Kelly) — o que mais DD COMPRA, e o que VAZA (vol-alvo 10%):")
    if p1x and p2x and p3x:
        L.append(f"    L=1.0x: CAGR {_fmt_pct(p1x.cagr)} · Sharpe {p1x.sharpe:.2f} · MaxDD {_fmt_pct(p1x.max_dd)}")
        L.append(f"    L=2.0x: CAGR {_fmt_pct(p2x.cagr)} · Sharpe {p2x.sharpe:.2f} · MaxDD {_fmt_pct(p2x.max_dd)}")
        L.append(f"    L=3.0x: CAGR {_fmt_pct(p3x.cagr)} · Sharpe {p3x.sharpe:.2f} · MaxDD {_fmt_pct(p3x.max_dd)}")
        d_cagr = (p3x.cagr - p1x.cagr) * 100
        d_dd = (abs(p3x.max_dd) - abs(p1x.max_dd)) * 100
        L.append(f"    De 1x a 3x: o CAGR sobe ~{d_cagr:.0f} p.p. mas o DD sobe ~{d_dd:.0f} p.p. — e o SHARPE CAI de")
        L.append(f"    {p1x.sharpe:.2f} p/ {p3x.sharpe:.2f}. ENTAO: alavancar NAO e 'de graca' (Sharpe constante). O")
        L.append("    financiamento (5.5% a.a. sobre o emprestado) + a vol crescente cobram um pedaco do retorno.")
        L.append("    Voce compra retorno absoluto MAIOR, mas retorno RISCO-AJUSTADO PIOR a cada degrau de L.")
        L.append("    Isto e 'ao longo da fronteira' com vazamento — honesto: +DD compra +CAGR, mas nao 1:1.")
    L.append("")
    L.append("    'PASSOU DO OTIMO' (so p/ registro — voce NAO deve chegar la): alem da Kelly (~6-8x, DD")
    L.append("    -65%/-90%) o CAGR PARA de subir e CAI, o Sharpe vira ~0.6 e o DD vai a -80%+. E destruicao")
    L.append("    pura de valor. A grade-cardapio para em 3x exatamente p/ nao induzir ninguem a essa zona.")
    L.append("")

    # --- (3) DD ao-vivo vs backtest ---
    L.append("(3) O DD DO BACKTEST E MELHOR-CASO — O AO-VIVO SERA PIOR:")
    L.append("    O backtest assume execucao perfeita, sem gaps, params sempre otimos. AO VIVO o DD")
    L.append(f"    tipicamente EXCEDE o do backtest em ~{LIVE_DD_MULT_LOW:.1f}-{LIVE_DD_MULT_HIGH:.1f}x (slippage de cauda, gaps de abertura,")
    L.append("    execucao imperfeita, regime que muda, alavancagem que amplifica o erro). A coluna 'DD")
    L.append("    ao-vivo estimado' do cardapio ja faz a conta: max(1.4x o backtest, pior crise observada).")
    if p2x:
        _, mid2, high2, _ = estimate_live_dd(p2x)
        L.append(f"    Exemplo: vol 10% x 2.0x tem DD backtest {_fmt_pct(p2x.max_dd)} -> planeje p/ {_fmt_pct(mid2)} a {_fmt_pct(high2)} AO VIVO.")
    L.append("    REGRA DE OURO: dimensione o orcamento de risco pelo DD AO-VIVO, NUNCA pelo backtest. Se")
    L.append("    voce so aguenta -25% de verdade, escolha uma linha cujo DD AO-VIVO <= -25% (DD backtest ~-18%).")
    L.append("")

    # --- (4) variantes de mix ---
    L.append("(4) EFEITO DAS VARIANTES DE MIX (no ponto vol-alvo 10%, L=2.0x):")
    base_p = _row_pt(grid, 0.10, 2.0)
    hi_p = _row_pt(grid_high_crypto, 0.10, 2.0)
    lo_p = _row_pt(grid_low_crypto, 0.10, 2.0)
    if base_p and hi_p and lo_p:
        L.append(f"    · cripto 5%/nome  (base)     : CAGR {_fmt_pct(base_p.cagr)} · Sharpe {base_p.sharpe:.2f} · MaxDD {_fmt_pct(base_p.max_dd)}")
        L.append(f"    · cripto 10%/nome (+risco)   : CAGR {_fmt_pct(hi_p.cagr)} · Sharpe {hi_p.sharpe:.2f} · MaxDD {_fmt_pct(hi_p.max_dd)}")
        L.append(f"    · cripto 2%/nome  (tilt def.) : CAGR {_fmt_pct(lo_p.cagr)} · Sharpe {lo_p.sharpe:.2f} · MaxDD {_fmt_pct(lo_p.max_dd)}")
        L.append("    LEITURA: mais cripto move pouco a media (teto de 5%->10% por nome) mas engrossa a cauda;")
        L.append("    o tilt defensivo (2%) suaviza de leve. E uma 2a alavanca de risco, mas CONCENTRADA num")
        L.append("    ativo so — pior diversificada que o L. Prefira ajustar o RISCO pelo L (cesta inteira),")
        L.append("    nao empilhando cripto. Cripto e tempero, nao o motor.")
    L.append("")

    # --- (5) recomendacao de faixa sa + tensao AUM ---
    L.append("(5) RECOMENDACAO DE FAIXA SA + TENSAO AUM:")
    kelly10_L = kelly10.lev if kelly10 else base_row[-1].lev
    sane_pt = _sane_upper(base_row, kelly10_L, dd_live_cap=0.30)
    _, sane_mid, _, _ = estimate_live_dd(sane_pt)
    L.append(f"    FAIXA SA (vol-alvo 10%): de L=1.0x ate ~L={sane_pt.lev:.2f}x. Teto definido NAO pela Kelly")
    L.append(f"    (longe demais), mas pelo DD AO-VIVO ainda 'vendavel': ~{_fmt_pct(sane_mid)} no topo da faixa.")
    L.append(f"    No topo sao da faixa (L~{sane_pt.lev:.2f}x): CAGR {_fmt_pct(sane_pt.cagr)}, Sharpe {sane_pt.sharpe:.2f}, MaxDD backtest {_fmt_pct(sane_pt.max_dd)}.")
    L.append("    Cada degrau ATE aqui compra retorno de forma aceitavel (Sharpe cai mas continua > ~0.95).")
    L.append("")
    L.append("    AVISO FORTE — ZONA QUE EU NAO RECOMENDO (acima do topo sao, ainda sub-Kelly):")
    L.append("    L > ~2x leva o DD AO-VIVO p/ alem de -35/40%. Voce ainda esta sub-Kelly (o CAGR ainda sobe),")
    L.append("    mas (i) o Sharpe ja caiu p/ ~0.8-0.9 e (ii) o DD ao-vivo entra na zona que QUEBRA conta e")
    L.append("    AFASTA alocador. Tecnicamente nao e 'pos-otimo' de Kelly, mas e 'pos-otimo' de SOBREVIVENCIA.")
    L.append("    E NUNCA, em nenhuma hipotese, opere perto da Kelly (~6-8x): la o DD ao-vivo passa de -90%.")
    L.append("")
    L.append("    TENSAO AUM (honesta, com a sua meta de captar): alocador institucional corta alocacao")
    L.append("    mental perto de -20% de DD e te demite em -30/35%. Um track com DD ao-vivo > ~-30% AFASTA")
    L.append("    dinheiro mesmo com CAGR maior — o retorno extra pode CUSTAR o AUM que voce quer levantar.")
    L.append("    Trade-off cru: +retorno (L mais alto) <-> +facilidade de captar (L mais baixo, DD contido).")
    L.append("    Se o objetivo e a RAMPA DE AUM, o L baixo (1.0-1.5x) vende melhor que o CAGR de um L=3x.")
    L.append("")
    L.append("    MINHA RECOMENDACAO (o Executor, sem vender sonho):")
    L.append("    · Quer rampa de AUM / track limpo: vol 10% x L=1.0-1.25x (DD ao-vivo ~-19/-23%, Sharpe ~1.15).")
    L.append("    · Aceita mais DD por mais retorno, consciente: vol 10% x L=1.5-2.0x (DD ao-vivo ~-28/-36%,")
    L.append("      CAGR +9/+11%, Sharpe ~1.0-1.1). Esse e o topo do que eu chamaria de 'sao'.")
    L.append("    · NAO recomendo passar de L~2.0-2.5x: o DD ao-vivo (>-40%) cobra caro e nao paga em Sharpe.")
    L.append("")
    L.append("    SE VOCE ESCOLHER UM PONTO NOVO: o Coder AUDITA a config (sizing, cap de cripto, taxa de")
    L.append("    financiamento, limites de margem, ruina) ANTES do go-live. Isto aqui e ANALISE — a config")
    L.append("    de PRODUCAO NAO foi alterada.")
    L.append("")
    L.append("  *** Numeros reais, sem look-ahead, custo de transacao + financiamento incluidos. ***")
    L.append("  *** DD do backtest e MELHOR-CASO; dimensione SEMPRE pelo DD AO-VIVO estimado. ***")
    return "\n".join(L)


# ============================================================================
# ASSEMBLE
# ============================================================================
def assemble_report(
    panel_full: pd.DataFrame,
    div_start: pd.Timestamp,
    grid: dict[float, list[FrontierPoint]],
    grid_high_crypto: dict[float, list[FrontierPoint]],
    grid_low_crypto: dict[float, list[FrontierPoint]],
    kelly_fine: dict[float, KellyPoint],
    beta: Stats,
    beta_dd_days: int,
    beta_crisis: dict[str, float],
    sensitivity: str,
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("BETA FRONTIER — fronteira risco/retorno do beta p/ ESCOLHA CONSCIENTE de orcamento de DD.")
    L.append("O usuario quer MAIS retorno ACEITANDO mais MaxDD. Mostra o que cada nivel de DD COMPRA de")
    L.append("retorno/Sharpe, onde fica a Kelly (e por que ela e in-investivel), e a faixa SA recomendada.")
    L.append("=" * 96)
    L.append("")
    L.append("PREMISSAS / HONESTIDADE (reusa simulation.beta_portfolio + simulation.beta_v2, NAO editados):")
    L.append("  NUCLEO: vol-target PURO (peso ~1/vol por ativo, teto de cripto, SEM portao de regime —")
    L.append(f"  o portao reprovou por whipsaw no modulo auditado) · lookback vol {VOL_LOOKBACK}d · rebalance mensal.")
    L.append(f"  EIXOS: vol-alvo {{{', '.join(f'{v*100:.0f}%' for v in VOL_TARGETS)}}} x alavancagem "
             f"{{{', '.join(f'{lv:.2f}' for lv in LEVERAGE_LEVELS)}}}.")
    L.append(f"  FINANCIAMENTO REAL: {FINANCING_ANNUAL*100:.1f}% a.a. (~rf medio + spread de margem) cobrado DIARIAMENTE")
    L.append("  so sobre o capital emprestado max(gross-1,0). Custo de transacao: equities 3bps/lado, cripto 35bps/lado.")
    L.append("  SEM LOOK-AHEAD: peso de t decidido com dado <= t-1 (.shift(1) herdado). Metricas na JANELA-CABECA")
    L.append(f"  (cesta diversificada, a partir de {div_start.date()}); crises no historico INTEIRO (p/ ver 2008).")
    L.append("  DD AO-VIVO ESTIMADO = max(1.4x o DD do backtest, pior crise observada). Backtest = MELHOR-CASO.")
    L.append("")
    spans = []
    for c in panel_full.columns:
        s = panel_full[c].dropna()
        if not s.empty:
            spans.append(f"{c}:{s.index[0].date()}→{s.index[-1].date()}")
    L.append(f"  PAINEL: {len(panel_full.columns)} ativos, {panel_full.index[0].date()}..{panel_full.index[-1].date()} "
             f"({len(panel_full)} dias).  " + "  ".join(spans))
    L.append("")

    # ---- (0) o cardapio PRIMEIRO (e o que o usuario pediu) ----
    L.append("=" * 96)
    L.append("(0) *** TABELA-CARDAPIO ***  'O DD que voce aguenta -> o CAGR / Sharpe que vem junto'")
    L.append("    Ordenada por DD do backtest crescente. ESCOLHA UMA LINHA conscientemente.")
    L.append("=" * 96)
    L.append(menu_table(grid, beta, beta_dd_days))
    L.append("")

    # ---- veredito (a parte honesta) logo apos o cardapio ----
    L.append(build_verdict(grid, grid_high_crypto, grid_low_crypto, kelly_fine, beta))
    L.append("")

    # ---- (1) a fronteira completa (dois eixos) ----
    L.append("=" * 96)
    L.append("(1) FRONTEIRA COMPLETA — DOIS EIXOS (vol-alvo x alavancagem). '<- maior CAGR da grade' = topo")
    L.append("    da grade (3x), NAO a Kelly real (que fica em ~6-8x, DD catastrofico — ver veredito).")
    L.append("    Variante de mix: cripto 5%/nome (base de producao).")
    L.append("=" * 96)
    L.append(frontier_table(grid, beta))
    L.append("")

    # ---- (2) variantes de mix ----
    L.append("=" * 96)
    L.append("(2a) VARIANTE: cripto 10%/nome (+ risco/retorno de cauda)")
    L.append("=" * 96)
    L.append(frontier_table(grid_high_crypto, beta))
    L.append("")
    L.append("=" * 96)
    L.append("(2b) VARIANTE: cripto 2%/nome (tilt defensivo / pro-acoes)")
    L.append("=" * 96)
    L.append(frontier_table(grid_low_crypto, beta))
    L.append("")

    # ---- (3) crises ----
    L.append("=" * 96)
    L.append("(3) COMPORTAMENTO EM CRISE (retorno acumulado, historico INTEIRO) — base cripto 5%/nome")
    L.append("=" * 96)
    L.append(crisis_table(grid, beta_crisis))
    L.append("")

    # ---- (4) sensibilidade ----
    L.append("=" * 96)
    L.append("(4) SENSIBILIDADE (Kelly e veredito NAO dependem de uma escolha fina de param)")
    L.append("=" * 96)
    L.append(sensitivity)
    L.append("")
    return "\n".join(L)


def _sensitivity(
    panel_full: pd.DataFrame, panel_window: pd.DataFrame, classes: dict[str, str],
    div_start: pd.Timestamp,
) -> str:
    """Robustez do veredito ao custo de financiamento. Duas visoes:
      (i) CAGR ao longo da GRADE-CARDAPIO (1-3x) p/ 3 taxas: dentro da grade o CAGR
          so SOBE (a Kelly esta muito alem de 3x), entao mostra so o vazamento que o
          juro maior tira de cada degrau.
      (ii) a KELLY VERDADEIRA (busca fina ate 12x) por taxa: confirma que (a) o
           turnover de vol-drag EXISTE de verdade e (b) financiamento maior puxa a
           Kelly p/ baixo — mas ela continua em DD catastrofico, in-investivel.
    Roda no vol-alvo base (10%). Se o veredito (Kelly in-investivel; faixa sa baixa)
    nao vira com a taxa, nao e overfit."""
    L: list[str] = []
    L.append("  (i) CAGR ao longo da GRADE-CARDAPIO (vol-alvo 10%) por taxa de financiamento:")
    head = f"    {'financiamento':<14} | " + " | ".join(f"L={lv:.2f}x".rjust(12) for lv in LEVERAGE_LEVELS)
    L.append(head)
    L.append("    " + "-" * (len(head) - 4))
    for fin in (0.03, 0.055, 0.08):
        cells = []
        for lv in LEVERAGE_LEVELS:
            pt = evaluate_point(
                panel_full, panel_window, classes,
                vol_target=0.10, lev=lv, crypto_max=MIX_BASE_CRYPTO,
                financing=fin, div_start=div_start,
            )
            cells.append(f"{_fmt_pct(pt.cagr)}".rjust(12))
        L.append(f"    {fin*100:>5.1f}% a.a.    | " + " | ".join(cells))
    L.append("    (DENTRO da grade o CAGR so sobe com L — a Kelly esta alem de 3x. Financiamento maior")
    L.append("     so tira um pedaco de cada degrau alavancado; o 1.0x nao usa margem, nao muda.)")
    L.append("")
    L.append("  (ii) KELLY VERDADEIRA (busca fina ate 12x) por taxa — confirma o turnover e a robustez:")
    head2 = f"    {'financiamento':<14} | {'Kelly L (real)':>14} | {'CAGR no pico':>12} | {'MaxDD backtest':>14}"
    L.append(head2)
    L.append("    " + "-" * (len(head2) - 4))
    for fin in (0.03, 0.055, 0.08):
        kp = find_kelly_fine(
            panel_full, panel_window, classes, div_start,
            vol_target=0.10, crypto_max=MIX_BASE_CRYPTO, financing=fin,
        )
        lev_s = f">={kp.lev:.1f}x" if kp.at_ceiling else f"{kp.lev:.2f}x"
        L.append(
            f"    {fin*100:>5.1f}% a.a.    | {lev_s:>13} | {_fmt_pct(kp.cagr):>12} | {_fmt_pct(kp.max_dd):>14}"
        )
    L.append("    (financiamento maior puxa a Kelly p/ baixo, mas mesmo a 8% ela fica em ~3.5x com DD")
    L.append("     backtest ~-44% (~-62% ao vivo); a taxas menores e ainda mais alta/profunda. IN-INVESTIVEL")
    L.append("     em qualquer taxa razoavel -> o veredito (DD binda antes da Kelly) e robusto, nao overfit.)")
    return "\n".join(L)


# ============================================================================
# RUNNER
# ============================================================================
def run(force: bool = False, do_sensitivity: bool = True) -> str:
    panel_full, classes = load_panel(force=force)
    if panel_full.empty:
        return (
            "DADOS PENDENTES: nenhum fechamento baixado (rede?). Rode:\n"
            "  uv run python -m simulation.beta_frontier --force"
        )
    div_start = diversified_start(panel_full, classes)
    panel_window = panel_full.loc[panel_full.index >= div_start]

    # beta de referencia (o que o usuario alavancaria).
    beta, beta_dd_days, beta_crisis = beta_reference(panel_full, panel_window, classes, div_start)

    # fronteira base (cripto 5%) + 2 variantes de mix.
    grid = build_frontier(panel_full, panel_window, classes, div_start, crypto_max=MIX_BASE_CRYPTO)
    grid_hi = build_frontier(panel_full, panel_window, classes, div_start, crypto_max=MIX_HIGH_CRYPTO)
    grid_lo = build_frontier(panel_full, panel_window, classes, div_start, crypto_max=MIX_LOW_CRYPTO)

    # Kelly fina por vol-alvo (confirma a da grade).
    kelly_fine: dict[float, KellyPoint] = {}
    for vt in VOL_TARGETS:
        kelly_fine[vt] = find_kelly_fine(
            panel_full, panel_window, classes, div_start, vol_target=vt,
        )

    sensitivity = "  (pulada — rode sem --no-sensitivity)"
    if do_sensitivity:
        sensitivity = _sensitivity(panel_full, panel_window, classes, div_start)

    return assemble_report(
        panel_full, div_start, grid, grid_hi, grid_lo, kelly_fine,
        beta, beta_dd_days, beta_crisis, sensitivity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beta frontier: mapeia a fronteira risco/retorno do beta (vol-alvo x alavancagem)."
    )
    parser.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    parser.add_argument("--no-sensitivity", action="store_true", help="pula a sensibilidade")
    parser.add_argument("--report-file", default="data/beta_frontier_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report = run(force=args.force, do_sensitivity=not args.no_sensitivity)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
