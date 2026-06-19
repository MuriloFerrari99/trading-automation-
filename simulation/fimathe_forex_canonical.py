"""TRIBUNAL CANONICO DA FIMATHE EM FOREX — a tecnica REAL no banco dos reus.

Pergunta unica: a FIMATHE CANONICA (fimathe/canonical.py — CR + Zona Neutra como
canal ADJACENTE projetado, so a favor da tendencia, entrada nos 50% do CR OU no
rompimento da fronteira da ZN, stop FORA da ZN, alvo por projecao 2x, Sub Ciclo de
Protecao, multi-timeframe marca D1 / opera H4) tem ALPHA em forex majors, LIQUIDO
de SPREAD + SWAP overnight, batendo o buy&hold do PROPRIO par?

REUSO HONESTO (nao reimplementa custo nem re-baixa dados):
  * loader de dados forex: data.forex_data.load_pair (cache em data/forex_cache/).
  * modelo de CUSTO spread+swap (3x quarta), DEFAULT_COSTS, stressed(): de
    simulation.fimathe_forex (o harness do baseline). Mesmo motor de noites/swap.
  * DSR/PSR/PBO: simulation.statistics.

CORRECAO CRITICA (que o Planner achou) — ALPHA, nao Sharpe vs zero:
  O baseline media Sharpe/DSR contra ZERO. Mas um sistema que so fica comprado num
  bull market mostra Sharpe positivo SEM adicionar alpha. Aqui a metrica de aprovacao
  e o EXCESSO sobre buy&hold: para CADA trade calculamos o retorno do buy&hold do par
  na MESMA janela de exposicao (mesmos bars de entrada/saida) e definimos
      alpha_ret = retorno_liquido_da_estrategia  -  buyhold_na_janela.
  O Sharpe e o DSR de APROVACAO rodam sobre a SERIE DE ALPHA (excesso sobre o B&H),
  nao sobre o retorno absoluto. Reportamos os dois (absoluto e alpha) lado a lado.

GATE 1 (no TOPO do relatorio):
  "PASSA so se, LIQUIDO de spread+swap (INCLUSIVE no estresse 2x):
   DSR(alpha)>=0.95 E Sharpe_liq_anual(alpha)>=0.8 E alpha>0 vs buy&hold do par E PBO<0.5.
   Senao FALHA."

VARREDURA (n_trials HONESTO = TODAS as combinacoes que rodam):
  pares (foco menor vol EURGBP/USDCHF + EURUSD; demais incluidos) x
  combos de timeframe (marca D1 opera H4; marca H4 opera H1) x
  variante de entrada (A=50% do CR; B=rompimento da ZN) x
  poucos params (swing_lookback).

Custo: base E ESTRESSADO 2x (o GATE exige robustez no estresse). SEM look-ahead
(sinal no fechamento de i, entrada no open de i+1; saida gap-aware do harness).

Uso:
    uv run python -m simulation.fimathe_forex_canonical
    uv run python -m simulation.fimathe_forex_canonical --report-file data/fimathe_canonical_verdict.txt
Importa sem efeitos colaterais (uv run python -c "import simulation.fimathe_forex_canonical").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.forex_data import PERIODS_PER_YEAR, load_pair, pip_size
from fimathe.canonical import CanonicalParams, FimatheCanonical
from simulation.statistics import (
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# REUSO do modelo de custo do harness baseline (NAO reimplementa).
from simulation.fimathe_forex import (
    BAR_DAYS,
    CONFIRMED_SWAP,
    DEFAULT_COSTS,
    ForexCost,
    _nights_swap_pips,
    stressed,
)

# --------------------------------------------------------------------------- #
# Configuracao da varredura (n_trials HONESTO = todas as combinacoes que rodam)
# --------------------------------------------------------------------------- #
# Combos multi-timeframe: (marca, opera). A tecnica swing: marca D1 opera H4.
# Tambem testamos marca H4 opera H1 (modalidade mais rapida).
TF_COMBOS = (("D1", "H4"), ("H4", "H1"))

# Foco do briefing: menor vol primeiro, depois os demais majors.
SWEEP_PAIRS = ("EURGBP", "USDCHF", "EURUSD", "GBPUSD", "USDJPY")
LOW_VOL_PAIRS = ("EURGBP", "USDCHF")

# Variantes de entrada: A = pullback aos 50% do CR; B = rompimento da fronteira da ZN.
ENTRY_VARIANTS = ("A", "B")

# Poucos params: ordem do pivo (perna da tendencia). Mantido CURTO p/ n_trials honesto.
SWING_SWEEP = (6, 10)

# Hold maximo (barras) por TF de operacao — swing: dias a semanas.
MAX_HOLD_BARS = {"H4": 60, "H1": 120}

# GATE 1 — criterio de aprovacao canonico (alpha vs buy&hold).
DSR_PASS = 0.95
SHARPE_PASS = 0.8      # Sharpe liquido anual sobre a SERIE DE ALPHA
PBO_PASS = 0.5         # PBO estritamente < 0.5

# Estresse de custo: o GATE exige robustez a 2x (briefing).
STRESS_FACTOR = 2.0


# --------------------------------------------------------------------------- #
# Trade canonico: entrada no open de i+1, saida gap-aware + Sub Ciclo de Protecao
# --------------------------------------------------------------------------- #
@dataclass
class CanonTrade:
    entry_bar: int
    exit_bar: int
    side: int
    gross_ret: float
    net_ret: float            # apos spread + swap
    bh_ret: float             # buy&hold do par na MESMA janela (mesmos bars)
    alpha_ret: float          # net_ret - bh_ret  (EXCESSO sobre o B&H sincronizado)
    hold_bars: int
    hold_days: float
    nights: float
    swap_ret: float
    spread_ret: float
    win: bool


def _exit_fill(
    side: int, op: float, hi: float, low: float, stop: float, target: float
) -> float | None:
    """Preco de saida gap-aware nesta barra, ou None se nem stop nem alvo dispararam.

    Ordem (empate intrabar => STOP, conservador): gap-through do stop preenche no OPEN
    real (pior que o stop); toque intrabar do stop preenche no stop; depois o alvo
    (gap-through no OPEN; toque no alvo). Identico ao harness baseline."""
    if side > 0:
        if op <= stop:        # gap abriu abaixo do stop -> open (pior)
            return op
        if low <= stop:       # tocou o stop intrabar
            return stop
        if op >= target:      # gap abriu acima do alvo -> open
            return op
        if hi >= target:      # tocou o alvo intrabar
            return target
    else:  # short
        if op >= stop:
            return op
        if hi >= stop:
            return stop
        if op <= target:
            return op
        if low <= target:
            return target
    return None


def generate_trades(
    df: pd.DataFrame,
    engine: FimatheCanonical,
    cost: ForexCost,
    *,
    operate_tf: str,
    pip: float,
    df_mark: pd.DataFrame | None = None,
) -> list[CanonTrade]:
    """Gera trades da FIMATHE canonica no TF de operacao.

    - Sinal no FECHAMENTO de i (FimatheCanonical.process), entrada no OPEN de i+1.
    - Saida gap-aware: stop/alvo com fill no OPEN real em gap-through; empate intrabar
      => stop (igual ao harness baseline).
    - Sub Ciclo de Protecao: o stop sobe (breakeven aos 50%, trava aos 100%, etc.)
      conforme o preco avanca — checado barra a barra ANTES do alvo (gestao real).
    - Buy&hold da MESMA janela calculado por trade (p/ alpha vs B&H sincronizado).
    """
    out = engine.process(df, df_mark=df_mark)
    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    lo = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    sig = out["signal"].to_numpy(float)
    val = out["setup_valid"].to_numpy(bool)
    sl = out["stop_loss"].to_numpy(float)
    tp = out["take_profit"].to_numpy(float)
    rng = out["cr_range"].to_numpy(float)
    idx = out.index
    n = len(c)
    max_hold = MAX_HOLD_BARS[operate_tf]
    spread_pips_rt = cost.entry_exit_cost_pips()

    trades: list[CanonTrade] = []
    i = 0
    while i < n - 1:
        if int(sig[i]) == 0 or not val[i]:
            i += 1
            continue
        side = 1 if sig[i] > 0 else -1
        entry = o[i + 1]
        stop0 = sl[i]
        target = tp[i]
        cr = rng[i]
        ok = np.isfinite(entry) and entry > 0 and np.isfinite(stop0) and np.isfinite(target)
        if side > 0:
            ok = ok and (0 < stop0 < entry)
        else:
            ok = ok and (stop0 > entry > 0)
        if not ok:
            i += 1
            continue

        # Sub Ciclo de Protecao: niveis (gatilho_preco, novo_stop) a partir da entrada.
        subcycle = engine.subcycle_stops(entry, side, cr, stop0)
        sc_ptr = 0
        stop = stop0

        end = min(i + 1 + max_hold, n - 1)
        exit_bar = end
        exit_fill = c[end]
        for j in range(i + 1, end + 1):
            # 1) avanca o trailing do sub-ciclo se o preco ja atingiu o(s) gatilho(s).
            while sc_ptr < len(subcycle):
                trig, new_stop = subcycle[sc_ptr]
                hit = (side > 0 and h[j] >= trig) or (side < 0 and lo[j] <= trig)
                if not hit:
                    break
                stop = max(stop, new_stop) if side > 0 else min(stop, new_stop)
                sc_ptr += 1

            # 2) checa saida por stop / alvo (gap-aware; empate intrabar => stop).
            hit = _exit_fill(side, o[j], h[j], lo[j], stop, target)
            if hit is not None:
                exit_bar = j
                exit_fill = hit
                break

        gross_ret = side * (exit_fill / entry - 1.0)
        spread_ret = -spread_pips_rt * pip / entry
        swap_pips, nights = _nights_swap_pips(idx[i + 1], idx[exit_bar], side, cost)
        swap_ret = swap_pips * pip / entry
        net_ret = gross_ret + spread_ret + swap_ret

        # Buy&hold do par na MESMA janela de exposicao (entrada -> saida).
        # ALPHA = excesso da estrategia sobre simplesmente segurar o par no mesmo periodo.
        bh_ret = c[exit_bar] / entry - 1.0
        alpha_ret = net_ret - bh_ret

        hold_bars = exit_bar - (i + 1)
        hold_days = hold_bars * BAR_DAYS[operate_tf]
        trades.append(CanonTrade(
            entry_bar=i + 1, exit_bar=exit_bar, side=side,
            gross_ret=gross_ret, net_ret=net_ret, bh_ret=bh_ret, alpha_ret=alpha_ret,
            hold_bars=hold_bars, hold_days=hold_days, nights=nights,
            swap_ret=swap_ret, spread_ret=spread_ret, win=net_ret > 0,
        ))
        i = exit_bar + 1  # 1 posicao por par por vez
    return trades


# --------------------------------------------------------------------------- #
# Metricas por config (par x combo de TF x variante x swing)
# --------------------------------------------------------------------------- #
@dataclass
class ConfigResult:
    pair: str
    mark_tf: str
    operate_tf: str
    variant: str
    swing: int
    n_trades: int
    hold_days: float
    win_rate: float
    gross_per_trade: float
    net_per_trade: float
    alpha_per_trade: float
    swap_per_trade: float
    spread_per_trade: float
    nights_per_trade: float
    net_total_ret: float
    buyhold_ret: float            # buy&hold do par no periodo coberto (1a entrada->ult saida)
    sharpe_net_annual: float      # Sharpe ABSOLUTO liquido anualizado
    sharpe_alpha_annual: float    # Sharpe do ALPHA (excesso sobre B&H) anualizado
    alpha_returns: np.ndarray     # serie de alpha por trade (p/ DSR/PBO)
    net_returns: np.ndarray       # serie liquida por trade (informativo)
    bars_per_year: int

    @property
    def label(self) -> str:
        return f"{self.pair} {self.mark_tf}->{self.operate_tf} {self.variant} sw={self.swing}"


def _buyhold_return(df: pd.DataFrame, first_bar: int, last_bar: int) -> float:
    c = df["close"].to_numpy(float)
    a = max(0, min(first_bar, len(c) - 1))
    b = max(0, min(last_bar, len(c) - 1))
    if b <= a or c[a] <= 0:
        return 0.0
    return float(c[b] / c[a] - 1.0)


def run_config(
    pair: str, mark_tf: str, operate_tf: str, variant: str, swing: int,
    df_oper: pd.DataFrame, df_mark: pd.DataFrame, cost: ForexCost,
) -> tuple[ConfigResult | None, str]:
    pip = pip_size(pair)
    eng = FimatheCanonical(CanonicalParams(
        swing_lookback=swing, entry_variant=variant, pip_size=pip,
    ))
    trades = generate_trades(
        df_oper, eng, cost, operate_tf=operate_tf, pip=pip, df_mark=df_mark
    )
    if len(trades) < 5:
        out = eng.process(df_oper, df_mark=df_mark)
        n_sig = int((out["signal"] != 0).sum())
        n_tr_nonzero = int((out["trend"] != 0).sum())
        if n_tr_nonzero == 0:
            reason = "0 barras com tendencia estrutural (HH/HL ou LH/LL) — nao arma setup"
        elif n_sig == 0:
            reason = f"{n_tr_nonzero} barras com tendencia mas 0 gatilhos disparados"
        else:
            reason = f"so {len(trades)} trades (<5): amostra insuficiente"
        return None, reason

    net = np.array([t.net_ret for t in trades], float)
    gross = np.array([t.gross_ret for t in trades], float)
    alpha = np.array([t.alpha_ret for t in trades], float)
    holds = np.array([t.hold_days for t in trades], float)
    nights = np.array([t.nights for t in trades], float)
    swap = np.array([t.swap_ret for t in trades], float)
    spread = np.array([t.spread_ret for t in trades], float)

    bpy = PERIODS_PER_YEAR[operate_tf]
    avg_hold_bars = max(float(np.mean([t.hold_bars for t in trades])), 1.0)
    trades_per_year = bpy / avg_hold_bars
    ann = float(np.sqrt(max(trades_per_year, 1.0)))
    sharpe_net_annual = observed_sharpe(net) * ann
    sharpe_alpha_annual = observed_sharpe(alpha) * ann

    first_bar = trades[0].entry_bar
    last_bar = trades[-1].exit_bar
    buyhold = _buyhold_return(df_oper, first_bar, last_bar)

    return ConfigResult(
        pair=pair, mark_tf=mark_tf, operate_tf=operate_tf, variant=variant, swing=swing,
        n_trades=len(trades), hold_days=float(np.mean(holds)),
        win_rate=float(np.mean([t.win for t in trades])),
        gross_per_trade=float(np.mean(gross)), net_per_trade=float(np.mean(net)),
        alpha_per_trade=float(np.mean(alpha)),
        swap_per_trade=float(np.mean(swap)), spread_per_trade=float(np.mean(spread)),
        nights_per_trade=float(np.mean(nights)),
        net_total_ret=float(np.prod(1.0 + net) - 1.0), buyhold_ret=buyhold,
        sharpe_net_annual=sharpe_net_annual, sharpe_alpha_annual=sharpe_alpha_annual,
        alpha_returns=alpha, net_returns=net, bars_per_year=bpy,
    ), "ok"


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.3f}%"


def _config_passes(r: ConfigResult, n_trials: int) -> tuple[bool, float, float]:
    """Aplica o GATE 1 sobre a SERIE DE ALPHA. Devolve (passa, dsr_alpha, pbo_placeholder).

    DSR e Sharpe medidos sobre o EXCESSO sobre buy&hold (nao vs zero). PBO entra no
    nivel da config? Nao — PBO e do PROCESSO de selecao (calculado por grupo). Aqui
    so o DSR/Sharpe/alpha-medio do config; o PBO do grupo e exigido a parte no veredito.
    """
    v = evaluate_edge(
        r.alpha_returns, n_trials=n_trials, periods_per_year=r.bars_per_year,
        min_sharpe_annual=SHARPE_PASS, dsr_threshold=DSR_PASS,
    )
    alpha_pos = r.alpha_per_trade > 0 and r.net_total_ret > r.buyhold_ret
    passes = bool(v.dsr >= DSR_PASS and r.sharpe_alpha_annual >= SHARPE_PASS and alpha_pos)
    return passes, v.dsr, 0.0


def _pbo_for_group(results: list[ConfigResult]) -> float | None:
    """PBO via CSCV sobre a matriz de ALPHA por trade entre configs do grupo."""
    mats = [r.alpha_returns for r in results]
    if len(mats) < 2:
        return None
    min_len = min(len(m) for m in mats)
    if min_len < 20:
        return None
    M = np.column_stack([m[:min_len] for m in mats])
    return probability_of_backtest_overfitting(M, n_splits=10)


def build_report(*, stress: bool) -> tuple[str, bool]:
    """Constroi o relatorio para UM cenario de custo. Devolve (texto, any_full_pass)."""
    cost_tag = f"ESTRESSADO({STRESS_FACTOR:g}x)" if stress else "base"
    lines: list[str] = []

    # ---- monta o grid e carrega dados; n_trials HONESTO = configs que rodam ----
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    data_status: list[str] = []
    grid: list[tuple[str, str, str, str, int]] = []  # (pair, mark, oper, variant, swing)
    needed_tfs = sorted({tf for combo in TF_COMBOS for tf in combo})
    for pair in SWEEP_PAIRS:
        for tf in needed_tfs:
            try:
                df = load_pair(pair, tf)
            except Exception as exc:  # noqa: BLE001
                data_status.append(f"  [DADOS PENDENTES] {pair} {tf}: {exc}")
                continue
            if df is None or df.empty:
                data_status.append(f"  [DADOS PENDENTES] {pair} {tf}: cache vazio")
                continue
            loaded[(pair, tf)] = df
    for pair in SWEEP_PAIRS:
        for (mark_tf, operate_tf) in TF_COMBOS:
            if (pair, mark_tf) not in loaded or (pair, operate_tf) not in loaded:
                continue
            for variant in ENTRY_VARIANTS:
                for sw in SWING_SWEEP:
                    grid.append((pair, mark_tf, operate_tf, variant, sw))
    n_trials = max(len(grid), 1)

    # ---- GATE 1 no topo ----
    lines += [
        "=" * 104,
        "TRIBUNAL CANONICO DA FIMATHE EM FOREX — a tecnica REAL (CR + ZN adjacente projetada)",
        "=" * 104,
        "",
        "GATE 1:",
        '  "PASSA so se, LIQUIDO de spread+swap (INCLUSIVE no estresse 2x):',
        f"   DSR(alpha)>={DSR_PASS:.2f} E Sharpe_liq_anual(alpha)>={SHARPE_PASS:.1f} E "
        "alpha>0 vs buy&hold do par E PBO<0.5.",
        '   Senao FALHA."',
        "",
        "CORRECAO CRITICA (Planner): a metrica de aprovacao e ALPHA (excesso sobre buy&hold",
        "do par na MESMA janela de exposicao), NAO Sharpe vs zero. So-comprar-num-bull nao passa.",
        "",
        f"Custo: {cost_tag}. SEM look-ahead (sinal no close de i, entrada no open de i+1).",
        "Saida gap-aware (open real em gap-through; empate intrabar => stop). Sub Ciclo de",
        "Protecao ativo (breakeven aos 50%, trava aos 100%). SWAP por noite, TRIPLO na quarta.",
        f"n_trials HONESTO (Deflated Sharpe) = pares x combos-TF x variante x swing = {n_trials}.",
        f"Combos de TF (marca->opera): {', '.join(f'{a}->{b}' for a, b in TF_COMBOS)}.",
        "Variantes de entrada: A=pullback aos 50% do CR; B=rompimento da fronteira da ZN.",
        f"Pares de MENOR vol (foco): {', '.join(LOW_VOL_PAIRS)}.",
        "",
    ]

    # custos aplicados (provenance)
    lines.append("CUSTOS APLICADOS (pips):")
    lines.append(f"  {'par':<8} {'spread':>7} {'swap_long':>10} {'swap_short':>11}  fonte")
    for pair in SWEEP_PAIRS:
        ck = DEFAULT_COSTS[pair]
        cc = stressed(ck, STRESS_FACTOR) if stress else ck
        src = "Hantec(long)" if pair in CONFIRMED_SWAP else "PLACEHOLDER honesto"
        lines.append(
            f"  {pair:<8} {cc.spread_pips:>7.2f} {cc.swap_long_pips:>10.2f} "
            f"{cc.swap_short_pips:>11.2f}  {src}"
        )
    lines.append(
        "  (swap negativo = voce PAGA por noite; so EURUSD long vem de numero publico citado.)"
    )
    lines.append("")

    if data_status:
        lines.append("STATUS DE DADOS:")
        lines += data_status
        lines.append(
            "  Para baixar: uv run python -m data.forex_data --timeframes D1,H4,H1 --force"
        )
        lines.append("")

    if not grid:
        lines.append("SEM DADOS — nenhuma config rodou. DADOS PENDENTES.")
        return "\n".join(lines) + "\n", False

    # ---- roda todas as configs ----
    results: list[ConfigResult] = []
    skipped: list[tuple[str, str]] = []
    for (pair, mark_tf, operate_tf, variant, sw) in grid:
        ck = DEFAULT_COSTS[pair]
        cc = stressed(ck, STRESS_FACTOR) if stress else ck
        res, reason = run_config(
            pair, mark_tf, operate_tf, variant, sw,
            loaded[(pair, operate_tf)], loaded[(pair, mark_tf)], cc,
        )
        if res is not None:
            results.append(res)
        else:
            skipped.append((f"{pair} {mark_tf}->{operate_tf} {variant} sw={sw}", reason))

    if skipped:
        lines.append("=" * 104)
        lines.append("CONFIGS SEM AMOSTRA SUFICIENTE (reportadas, nao suavizadas):")
        lines.append("=" * 104)
        for (lbl, reason) in skipped:
            lines.append(f"  {lbl:<34} -> {reason}")
        lines.append("")

    if not results:
        lines.append("Nenhuma config gerou >=5 trades. FALHA por amostra.")
        lines += _failure_banner()
        return "\n".join(lines) + "\n", False

    # ---- tabela por combo de TF ----
    header = (
        f"  {'config':<28} {'#tr':>4} {'hold_d':>7} {'win%':>6} "
        f"{'bruto/tr':>10} {'liq/tr':>10} {'alpha/tr':>10} {'swap/tr':>10} "
        f"{'Shp_liq':>8} {'Shp_alf':>8} {'DSR_alf':>8} {'B&H':>9} ver"
    )
    lines.append("=" * 104)
    lines.append("RESULTADO POR COMBO DE TF (liquido de spread+swap; metrica = ALPHA vs B&H):")
    lines.append("=" * 104)

    any_full_pass = False
    group_pbo: dict[tuple[str, str], float | None] = {}
    for (mark_tf, operate_tf) in TF_COMBOS:
        grp = [r for r in results if r.mark_tf == mark_tf and r.operate_tf == operate_tf]
        if not grp:
            continue
        lines.append(f"\n--- marca {mark_tf} / opera {operate_tf} (barras/ano={PERIODS_PER_YEAR[operate_tf]}) ---")
        lines.append(header)
        pbo = _pbo_for_group(grp)
        group_pbo[(mark_tf, operate_tf)] = pbo
        pbo_ok = (pbo is not None and pbo < PBO_PASS)
        for r in sorted(grp, key=lambda x: (x.pair, x.variant, x.swing)):
            passes_cfg, dsr_alpha, _ = _config_passes(r, n_trials)
            full_pass = passes_cfg and pbo_ok  # GATE exige tambem PBO do grupo < 0.5
            any_full_pass = any_full_pass or full_pass
            flag = "PASSA" if full_pass else "FALHA"
            lines.append(
                f"  {r.label:<28} {r.n_trades:>4} {r.hold_days:>7.1f} "
                f"{r.win_rate*100:>5.1f} {_fmt_pct(r.gross_per_trade):>10} "
                f"{_fmt_pct(r.net_per_trade):>10} {_fmt_pct(r.alpha_per_trade):>10} "
                f"{_fmt_pct(r.swap_per_trade):>10} {r.sharpe_net_annual:>8.2f} "
                f"{r.sharpe_alpha_annual:>8.2f} {dsr_alpha:>8.3f} "
                f"{_fmt_pct(r.buyhold_ret):>9} {flag}"
            )
        pbo_txt = f"{pbo:.2f}" if pbo is not None else "n/a (amostra)"
        lines.append(
            f"  PBO do grupo (selecao entre {len(grp)} configs, sobre ALPHA): {pbo_txt}"
            f"  [{'OK <0.5' if pbo_ok else 'REPROVA >=0.5'}]"
        )

    # ---- melhor por variante (resolve a incerteza 50% vs ZN) ----
    lines.append("\n" + "=" * 104)
    lines.append("QUAL ENTRADA FUNCIONA MELHOR? (50% do CR [A] vs rompimento da ZN [B]):")
    lines.append("=" * 104)
    for variant in ENTRY_VARIANTS:
        vr = [r for r in results if r.variant == variant]
        if not vr:
            lines.append(f"  variante {variant}: sem amostra.")
            continue
        # rank por Sharpe do ALPHA (a metrica que importa)
        best = max(vr, key=lambda x: x.sharpe_alpha_annual)
        mean_alpha = float(np.mean([r.alpha_per_trade for r in vr]))
        mean_shp_alpha = float(np.mean([r.sharpe_alpha_annual for r in vr]))
        name = "50% do CR" if variant == "A" else "rompimento da ZN"
        lines.append(
            f"  variante {variant} ({name}): {len(vr)} configs | "
            f"Shp_alpha medio={mean_shp_alpha:+.2f} alpha/tr medio={_fmt_pct(mean_alpha)} | "
            f"melhor={best.label} Shp_alpha={best.sharpe_alpha_annual:+.2f} "
            f"alpha/tr={_fmt_pct(best.alpha_per_trade)}"
        )

    # ---- melhor config global ----
    lines.append("\n" + "=" * 104)
    lines.append("MELHOR CONFIG GLOBAL (por Sharpe do ALPHA):")
    lines.append("=" * 104)
    best_overall = max(results, key=lambda x: x.sharpe_alpha_annual)
    bp, dsr_b, _ = _config_passes(best_overall, n_trials)
    lines.append(
        f"  {best_overall.label}: #tr={best_overall.n_trades} hold={best_overall.hold_days:.1f}d "
        f"win%={best_overall.win_rate*100:.1f}"
    )
    lines.append(
        f"    bruto/tr={_fmt_pct(best_overall.gross_per_trade)} "
        f"liq/tr={_fmt_pct(best_overall.net_per_trade)} "
        f"alpha/tr={_fmt_pct(best_overall.alpha_per_trade)} "
        f"swap/tr={_fmt_pct(best_overall.swap_per_trade)}"
    )
    lines.append(
        f"    Sharpe_liq={best_overall.sharpe_net_annual:+.2f} "
        f"Sharpe_ALPHA={best_overall.sharpe_alpha_annual:+.2f} DSR(alpha)={dsr_b:.3f} "
        f"total_liq={_fmt_pct(best_overall.net_total_ret)} (B&H={_fmt_pct(best_overall.buyhold_ret)})"
    )

    # ---- recorte: menor vol ----
    lines.append("\n" + "=" * 104)
    lines.append("RECORTE — PARES DE MENOR VOLATILIDADE (EURGBP, USDCHF):")
    lines.append("=" * 104)
    low = [r for r in results if r.pair in LOW_VOL_PAIRS]
    if low:
        for r in sorted(low, key=lambda x: x.sharpe_alpha_annual, reverse=True)[:5]:
            beats = "alpha>0" if r.alpha_per_trade > 0 else "alpha<=0"
            lines.append(
                f"  {r.label}: Shp_alpha={r.sharpe_alpha_annual:+.2f} "
                f"alpha/tr={_fmt_pct(r.alpha_per_trade)} liq/tr={_fmt_pct(r.net_per_trade)} ({beats})"
            )
    else:
        lines.append("  sem dados para os pares de menor vol.")

    # ---- quanto o swap pesou ----
    lines.append("\n" + "=" * 104)
    lines.append("QUANTO O SWAP COMEU DO EDGE:")
    lines.append("=" * 104)
    gross_avg = float(np.mean([r.gross_per_trade for r in results]))
    net_avg = float(np.mean([r.net_per_trade for r in results]))
    alpha_avg = float(np.mean([r.alpha_per_trade for r in results]))
    swap_avg = float(np.mean([r.swap_per_trade for r in results]))
    spread_avg = float(np.mean([r.spread_per_trade for r in results]))
    nights_avg = float(np.mean([r.nights_per_trade for r in results]))
    holds_avg = float(np.mean([r.hold_days for r in results]))
    lines.append(
        f"  Media entre {len(results)} configs: bruto/tr={_fmt_pct(gross_avg)}  "
        f"swap/tr={_fmt_pct(swap_avg)}  spread/tr={_fmt_pct(spread_avg)}  "
        f"liq/tr={_fmt_pct(net_avg)}  alpha/tr={_fmt_pct(alpha_avg)}"
    )
    lines.append(f"  Hold medio={holds_avg:.1f} dias  noites medias/trade={nights_avg:.1f}")
    if gross_avg > 0:
        lines.append(
            f"  O SWAP sozinho consumiu ~{(-swap_avg/gross_avg)*100:.0f}% do retorno BRUTO medio; "
            f"spread+swap ~{(-(swap_avg+spread_avg)/gross_avg)*100:.0f}%."
        )
    else:
        lines.append(
            "  Retorno BRUTO medio <=0: a entrada canonica nao gera edge bruto positivo medio; "
            "o swap apenas enterra mais fundo."
        )

    # ---- veredito do cenario ----
    lines.append("\n" + "=" * 104)
    if any_full_pass:
        lines += [
            "VEREDITO (este cenario): ao menos uma config passou o GATE 1 (DSR_alpha>=0.95 E",
            "Sharpe_alpha>=0.8 E alpha>0 E PBO<0.5). NAO comemorar: exige AUDITORIA INDEPENDENTE",
            "do Coder antes de qualquer passo rumo a dinheiro real.",
        ]
    else:
        lines += _failure_banner()
    lines.append("=" * 104)
    return "\n".join(lines) + "\n", any_full_pass


def _failure_banner() -> list[str]:
    return [
        "##############################################################################",
        "#                                                                            #",
        "#   F A L H A   —   FIMATHE CANONICA EM FOREX NAO TEM ALPHA                   #",
        "#                                                                            #",
        "#   Nenhuma config (par x combo-TF x variante x swing) passou o GATE 1:       #",
        "#   DSR(alpha)>=0.95 E Sharpe_alpha>=0.8 E alpha>0 vs buy&hold E PBO<0.5.      #",
        "#   A tecnica REAL (CR + ZN adjacente, a favor da tendencia, stop fora da     #",
        "#   ZN, alvo 2x, sub-ciclo) tambem nao bate segurar o par, liquido de         #",
        "#   spread+swap. Resultado HONESTO e valioso: o lever price-based esta        #",
        "#   esgotado; restaria dado alternativo. NAO seguir p/ dinheiro real.         #",
        "#                                                                            #",
        "##############################################################################",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tribunal CANONICO da FIMATHE em forex (alpha vs buy&hold; spread+swap)"
    )
    parser.add_argument("--report-file", default="data/fimathe_canonical_verdict.txt")
    parser.add_argument(
        "--no-stress", action="store_true",
        help="pula o cenario estressado 2x (default: roda base E estresse)",
    )
    args = parser.parse_args(argv)

    base_text, base_pass = build_report(stress=False)
    text = base_text
    stress_pass = False
    if not args.no_stress:
        stress_text, stress_pass = build_report(stress=True)
        text += "\n\n" + stress_text

    # Veredito FINAL: o GATE exige robustez no estresse -> so PASSA se passou nos DOIS.
    overall_pass = base_pass and (stress_pass or args.no_stress)
    final: list[str] = ["", "=" * 104, "VEREDITO FINAL (base + estresse 2x):", "=" * 104]
    if overall_pass:
        final += [
            "  Ao menos uma config passou o GATE 1 NO BASE E NO ESTRESSE 2x.",
            "  >>> PRECISA DE AUDITORIA INDEPENDENTE DO CODER antes de qualquer passo rumo a",
            "      dinheiro real. NAO comemorar; reproduzir e auditar primeiro.",
        ]
    else:
        why = "no estresse 2x" if base_pass and not stress_pass else "em nenhum cenario"
        final += [f"  Nenhuma config passou o GATE 1 {why}."]
        final += _failure_banner()
    text += "\n" + "\n".join(final) + "\n"

    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
