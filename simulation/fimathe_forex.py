"""TRIBUNAL HONESTO DA FIMATHE EM FOREX (baseline) — o primeiro teste.

Pergunta única: a FIMATHE (FimatheEngine, motor ATUAL, entrada no rompimento)
tem edge em forex majors LÍQUIDO de SPREAD + SWAP, batendo buy&hold do próprio
par? Este é o BASELINE. A "técnica pura" do Marcelo Ferreira está sendo extraída
em paralelo; depois trocamos a entrada/timeframe e reusamos ESTE harness.

O QUE ESTE MÓDULO FAZ DE HONESTO
- Roda a FimatheEngine COMO ESTÁ (simulation/breakout_tribunal.py é o irmão de
  cripto/equities; aqui é forex CFD com custo overnight). Saída gap-aware: stop/
  alvo com fill no OPEN real quando há gap-through; empate intrabar => stop.
- SEM look-ahead: sinal no FECHAMENTO da barra i, entrada no OPEN da barra i+1.
- CUSTO FOREX CFD modelado EXPLICITAMENTE (ForexCost):
    * SPREAD por par, cobrado na ENTRADA e na SAÍDA (em pips).
    * SWAP overnight por NOITE aberta, com TRIPLO na quarta-feira (rollover de
      3 dias). Defaults a partir dos números públicos da Hantec; placeholders
      HONESTOS (marcados) onde não há valor confirmado. É o swap que mata holds
      longos — reportamos hold médio em DIAS e o swap acumulado por trade.
- BENCHMARK = buy&hold do MESMO par no MESMO período (ALPHA, não Sharpe vs zero).
- n_trials HONESTO no Deflated Sharpe = pares × timeframes × params de busca.
- PBO via CSCV sobre a matriz de retornos por config (anti data-snooping).

KILL-CRITERION (no topo do relatório): PASSA só se DSR>=0.95 E Sharpe_líq_anual>=1.0
E retorno líquido (após spread+swap)>0 E supera buy&hold do par. Senão FALHA.

Uso:
    uv run python -m simulation.fimathe_forex
    uv run python -m simulation.fimathe_forex --report-file data/fimathe_forex_verdict.txt
    uv run python -m simulation.fimathe_forex --timeframes D1,H4   # subconjunto

Importa sem efeitos colaterais (uv run python -c "import simulation.fimathe_forex").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.forex_data import (
    DEFAULT_PAIRS,
    PERIODS_PER_YEAR,
    TIMEFRAMES,
    load_pair,
    pip_size,
)
from fimathe.engine import FimatheEngine
from simulation.statistics import (
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# Limites de hold por timeframe (barras). Em D1 ~20 dias; intraday mais barras
# mas, convertido a dias, holds parecidos (ver coluna "hold médio (dias)").
MAX_HOLD_BARS = {"D1": 20, "H4": 30, "H1": 48}
# Quantos dias-calendário 1 barra representa (p/ contar noites de swap).
BAR_DAYS = {"D1": 1.0, "H4": 4.0 / 24.0, "H1": 1.0 / 24.0}
# Varredura HONESTA de swing_period por timeframe (entra no n_trials).
SWING_SWEEP = {"D1": (10, 20, 30), "H4": (20, 40, 60), "H1": (24, 48, 96)}

# Kill-criterion deste baseline.
DSR_PASS = 0.95
SHARPE_PASS = 1.0

# Pares de MENOR volatilidade (reportar separável, conforme briefing).
LOW_VOL_PAIRS = ("EURGBP", "USDCHF")


# --------------------------------------------------------------------------- #
# Modelo de CUSTO forex CFD: spread (entrada+saída) + swap overnight (3x quarta)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForexCost:
    """Custo de CFD forex em PIPS. Parametrizável por par.

    - `spread_pips`: spread típico cobrado na ENTRADA e na SAÍDA (cada uma).
    - `swap_long_pips` / `swap_short_pips`: swap overnight por NOITE, em pips, no
      sinal correto (NEGATIVO = você PAGA). Long EURUSD paga; alguns shorts
      recebem, mas mantemos placeholders conservadores (custo) onde não há número
      confirmado — um edge que só sobrevive com swap otimista não é edge.
    - Quarta-feira: swap TRIPLO (rollover de fim de semana de 3 dias na Qua).

    PROVENIÊNCIA dos defaults (ver DEFAULT_COSTS): spreads = faixas típicas de
    corretora de varejo; swaps ancorados nos números públicos da Hantec (ex.:
    EURUSD long ~−10 pips/noite). Onde não há valor confirmado, o número é um
    PLACEHOLDER HONESTO marcado no relatório — não é dado de corretora auditado.
    """

    spread_pips: float
    swap_long_pips: float
    swap_short_pips: float
    name: str = "fx"

    def entry_exit_cost_pips(self) -> float:
        """Spread pago na entrada + na saída (round-trip)."""
        return 2.0 * self.spread_pips


# Defaults por par. swap em PIPS/NOITE (negativo = paga).
# EURUSD long ~−10 pips/noite ancorado nos números da Hantec citados no briefing.
# Os demais são placeholders honestos calibrados pela mesma ordem de grandeza
# (carry de juros): pares com USD comprado vs EUR/CHF/JPY tendem a custo menor;
# vendido tende a custo maior. NÃO são números de corretora confirmados.
DEFAULT_COSTS: dict[str, ForexCost] = {
    # par      spread  swap_long  swap_short
    "EURUSD": ForexCost(0.6, -10.0, -3.0, "EURUSD"),   # long ~-10 (Hantec); short placeholder
    "EURGBP": ForexCost(1.0, -4.0, -5.0, "EURGBP"),    # placeholders (baixo carry)
    "USDCHF": ForexCost(1.2, -2.0, -9.0, "USDCHF"),    # long USD recebe menos custo; placeholders
    "GBPUSD": ForexCost(0.9, -8.0, -4.0, "GBPUSD"),    # placeholders
    "USDJPY": ForexCost(0.9, 3.0, -12.0, "USDJPY"),    # long USD/JPY costuma RECEBER carry; placeholders
}
# Pares cujo swap é PLACEHOLDER (não confirmado) — sinalizado no relatório.
CONFIRMED_SWAP = {"EURUSD"}  # só o long de EURUSD vem de número público citado


def stressed(cost: ForexCost, factor: float = 1.5) -> ForexCost:
    """Cenário estressado: spread e CUSTO de swap pioram (factor>1).

    Aplica o fator só ao componente de custo (swap negativo fica mais negativo;
    swap positivo/carry recebido NÃO vira lucro maior — fica no máximo zero)."""
    sl = cost.swap_long_pips * factor if cost.swap_long_pips < 0 else 0.0
    ss = cost.swap_short_pips * factor if cost.swap_short_pips < 0 else 0.0
    return ForexCost(cost.spread_pips * factor, sl, ss, f"{cost.name}_stress{factor:g}x")


def _nights_swap_pips(
    entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, side: int, cost: ForexCost
) -> tuple[float, float]:
    """Swap acumulado (pips) e nº de noites entre entrada e saída.

    Conta cada virada de dia-calendário (UTC) como 1 noite; quarta-feira = 3x.
    side: +1 long, -1 short. Retorna (swap_total_pips_assinado, n_noites)."""
    if not (isinstance(entry_ts, pd.Timestamp) and isinstance(exit_ts, pd.Timestamp)):
        return 0.0, 0.0
    d0 = entry_ts.normalize()
    d1 = exit_ts.normalize()
    nights = pd.date_range(d0, d1, freq="D")[1:]  # cada meia-noite cruzada = 1 noite
    if len(nights) == 0:
        return 0.0, 0.0
    per_night = cost.swap_long_pips if side > 0 else cost.swap_short_pips
    total = 0.0
    for d in nights:
        mult = 3.0 if d.weekday() == 2 else 1.0  # 2 = quarta-feira
        total += per_night * mult
    return float(total), float(len(nights))


# --------------------------------------------------------------------------- #
# Geração de trades (long+short) gap-aware, com custo forex aplicado por trade
# --------------------------------------------------------------------------- #
@dataclass
class FXTrade:
    entry_bar: int
    exit_bar: int
    side: int               # +1 long, -1 short
    gross_ret: float        # retorno do par no hold (sinal já aplicado), SEM custo
    net_ret: float          # após spread + swap (em fração de preço)
    hold_bars: int
    hold_days: float
    nights: float
    swap_pips: float        # swap acumulado assinado (pips)
    swap_ret: float         # swap em fração de preço (negativo = custo)
    spread_ret: float       # spread round-trip em fração (negativo)
    win: bool


def generate_trades(
    df: pd.DataFrame,
    engine: FimatheEngine,
    cost: ForexCost,
    *,
    timeframe: str,
    pip: float,
) -> list[FXTrade]:
    """Setups LONG e SHORT da FimatheEngine, saída gap-aware, custo forex CFD.

    Sinal no fechamento de i; entrada no OPEN de i+1 (sem look-ahead)."""
    max_hold = MAX_HOLD_BARS[timeframe]
    if len(df) < engine.swing_period + max_hold + 5:
        return []
    out = engine.process(df)
    idx = out.index
    o = out["open"].to_numpy(float); h = out["high"].to_numpy(float)
    lo = out["low"].to_numpy(float); c = out["close"].to_numpy(float)
    sig = out["signal"].to_numpy(float); val = out["setup_valid"].to_numpy(bool)
    sl = out["stop_loss"].to_numpy(float); tp = out["take_profit_2"].to_numpy(float)
    n = len(c)
    spread_pips_rt = cost.entry_exit_cost_pips()  # round-trip em pips

    trades: list[FXTrade] = []
    i = engine.swing_period
    while i < n - 1:
        s = int(sig[i])
        if s == 0 or not val[i]:
            i += 1
            continue
        side = 1 if s > 0 else -1
        entry = o[i + 1]
        stop = sl[i]; target = tp[i]
        ok = np.isfinite(entry) and entry > 0 and np.isfinite(stop) and np.isfinite(target)
        if side > 0:
            ok = ok and (0 < stop < entry)
        else:
            ok = ok and (stop > entry > 0)
        if not ok:
            i += 1
            continue

        end = min(i + 1 + max_hold, n - 1)
        exit_bar = end
        exit_fill = c[end]
        for j in range(i + 1, end + 1):
            if side > 0:
                if o[j] <= stop:               # gap-through do stop -> open (pior)
                    exit_bar = j; exit_fill = o[j]; break
                if lo[j] <= stop:              # tocou stop intrabar
                    exit_bar = j; exit_fill = stop; break
                if o[j] >= target:            # gap-through do alvo -> open
                    exit_bar = j; exit_fill = o[j]; break
                if h[j] >= target:            # tocou alvo intrabar
                    exit_bar = j; exit_fill = target; break
            else:  # short
                if o[j] >= stop:               # gap-through do stop -> open (pior)
                    exit_bar = j; exit_fill = o[j]; break
                if h[j] >= stop:
                    exit_bar = j; exit_fill = stop; break
                if o[j] <= target:            # gap-through do alvo -> open
                    exit_bar = j; exit_fill = o[j]; break
                if lo[j] <= target:
                    exit_bar = j; exit_fill = target; break

        # Retorno BRUTO do par no hold, com o sinal aplicado (short ganha na queda).
        gross_ret = side * (exit_fill / entry - 1.0)

        # CUSTO forex: spread round-trip + swap por noite (3x quarta).
        spread_ret = -spread_pips_rt * pip / entry
        swap_pips, nights = _nights_swap_pips(idx[i + 1], idx[exit_bar], side, cost)
        swap_ret = swap_pips * pip / entry  # já assinado (negativo = custo)
        net_ret = gross_ret + spread_ret + swap_ret

        hold_bars = exit_bar - (i + 1)
        hold_days = hold_bars * BAR_DAYS[timeframe]
        trades.append(FXTrade(
            entry_bar=i + 1, exit_bar=exit_bar, side=side,
            gross_ret=gross_ret, net_ret=net_ret,
            hold_bars=hold_bars, hold_days=hold_days, nights=nights,
            swap_pips=swap_pips, swap_ret=swap_ret, spread_ret=spread_ret,
            win=net_ret > 0,
        ))
        i = exit_bar + 1  # 1 posição por par por vez
    return trades


# --------------------------------------------------------------------------- #
# Métricas por (par, timeframe, swing_period)
# --------------------------------------------------------------------------- #
@dataclass
class ConfigResult:
    pair: str
    timeframe: str
    swing: int
    n_trades: int
    hold_days: float
    win_rate: float
    gross_per_trade: float
    net_per_trade: float
    swap_per_trade: float
    spread_per_trade: float
    nights_per_trade: float
    net_total_ret: float          # produto composto dos retornos líquidos por trade
    buyhold_ret: float            # buy&hold do par no MESMO período coberto
    sharpe_net_annual: float
    trade_net_returns: np.ndarray  # série por trade (p/ tribunal)
    bars_per_year: int


def _buyhold_return(df: pd.DataFrame, first_bar: int, last_bar: int) -> float:
    """Retorno buy&hold do par entre a 1ª entrada e a última saída do teste."""
    c = df["close"].to_numpy(float)
    a = max(0, min(first_bar, len(c) - 1))
    b = max(0, min(last_bar, len(c) - 1))
    if b <= a or c[a] <= 0:
        return 0.0
    return float(c[b] / c[a] - 1.0)


def run_config(
    pair: str, timeframe: str, swing: int, df: pd.DataFrame, cost: ForexCost
) -> tuple[ConfigResult | None, str]:
    """Roda 1 config. Retorna (resultado|None, motivo). motivo explica amostras
    insuficientes (ex.: o gate PCM do baseline zera os sinais no diário)."""
    pip = pip_size(pair)
    engine = FimatheEngine(swing_period=swing, pip_size=pip)
    out = engine.process(df)
    # rompimentos CRUS do canal (antes do gate PCM do motor): diagnóstico honesto.
    c = out["close"]; up = out["upper_channel"]; lo = out["lower_channel"]
    raw_breaks = int(((c > up) | (c < lo)).sum())
    n_signals = int((out["signal"] != 0).sum())  # já gateado por PCM
    n_valid = int(out["setup_valid"].sum())
    trades = generate_trades(df, engine, cost, timeframe=timeframe, pip=pip)
    if len(trades) < 5:
        if n_signals == 0 and raw_breaks > 0:
            reason = (
                f"{raw_breaks} rompimentos crus do canal, mas 0 sinais: o gate PCM "
                "(pcm_score>=0.5) do motor atual zera TODOS (corpos fracos no forex)"
            )
        elif n_signals == 0:
            reason = "0 rompimentos do canal no fechamento (sem sinais)"
        elif n_valid == 0:
            reason = (
                f"{n_signals} sinais mas 0 setups válidos "
                "(gate breakout-strength/R:R do baseline reprova todos)"
            )
        else:
            reason = f"só {len(trades)} trades (<5): amostra insuficiente"
        return None, reason

    net = np.array([t.net_ret for t in trades], float)
    gross = np.array([t.gross_ret for t in trades], float)
    holds = np.array([t.hold_days for t in trades], float)
    nights = np.array([t.nights for t in trades], float)
    swap = np.array([t.swap_ret for t in trades], float)
    spread = np.array([t.spread_ret for t in trades], float)

    bpy = PERIODS_PER_YEAR[timeframe]
    # Annualização por TRADE: trades/ano = barras/ano ÷ (hold médio em barras).
    avg_hold_bars = max(float(np.mean([t.hold_bars for t in trades])), 1.0)
    trades_per_year = bpy / avg_hold_bars
    sr_per_trade = observed_sharpe(net)
    sharpe_net_annual = sr_per_trade * float(np.sqrt(max(trades_per_year, 1.0)))

    first_bar = trades[0].entry_bar
    last_bar = trades[-1].exit_bar
    buyhold = _buyhold_return(df, first_bar, last_bar)

    return ConfigResult(
        pair=pair, timeframe=timeframe, swing=swing, n_trades=len(trades),
        hold_days=float(np.mean(holds)), win_rate=float(np.mean([t.win for t in trades])),
        gross_per_trade=float(np.mean(gross)), net_per_trade=float(np.mean(net)),
        swap_per_trade=float(np.mean(swap)), spread_per_trade=float(np.mean(spread)),
        nights_per_trade=float(np.mean(nights)),
        net_total_ret=float(np.prod(1.0 + net) - 1.0), buyhold_ret=buyhold,
        sharpe_net_annual=sharpe_net_annual, trade_net_returns=net, bars_per_year=bpy,
    ), "ok"


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.3f}%"


def build_report(
    pairs: list[str], timeframes: list[str], *, stress: bool = False
) -> str:
    cost_tag = "ESTRESSADO(1.5x)" if stress else "base"
    lines: list[str] = []

    # ---- contar n_trials HONESTO: configs que efetivamente rodam ----
    grid: list[tuple[str, str, int]] = []
    data_status: list[str] = []
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    for tf in timeframes:
        for pair in pairs:
            try:
                df = load_pair(pair, tf)
            except Exception as exc:  # noqa: BLE001
                data_status.append(f"  [DADOS PENDENTES] {pair} {tf}: {exc}")
                continue
            if df.empty:
                data_status.append(f"  [DADOS PENDENTES] {pair} {tf}: cache vazio")
                continue
            loaded[(pair, tf)] = df
            for sw in SWING_SWEEP[tf]:
                grid.append((pair, tf, sw))
    n_trials = max(len(grid), 1)

    # ---- KILL-CRITERION no topo, ANTES dos números ----
    lines += [
        "=" * 100,
        "TRIBUNAL HONESTO DA FIMATHE EM FOREX — BASELINE (motor atual, entrada no rompimento)",
        "=" * 100,
        "",
        "KILL-CRITERION:",
        f'  "PASSA só se: DSR>={DSR_PASS:.2f} E Sharpe_líq_anual>={SHARPE_PASS:.1f} E '
        "retorno líquido (após spread+swap)>0 E supera buy&hold do par.",
        '   Senão FALHA (baseline). Nota: é o motor ATUAL; a técnica pura pode mudar '
        'a entrada/timeframe."',
        "",
        f"Custo: {cost_tag}. SEM look-ahead (sinal no close de i, entrada no open de i+1).",
        "Saída gap-aware (open real em gap-through; empate intrabar => stop).",
        "SWAP overnight por noite aberta, TRIPLO na quarta. Benchmark = buy&hold do PAR (ALPHA).",
        f"n_trials HONESTO (Deflated Sharpe) = pares × timeframes × swing = {n_trials}.",
        f"Pares de MENOR vol (reporte separável): {', '.join(LOW_VOL_PAIRS)}.",
        "",
    ]

    # provenance dos custos
    lines.append("CUSTOS APLICADOS (pips):")
    lines.append(f"  {'par':<8} {'spread':>7} {'swap_long':>10} {'swap_short':>11}  fonte")
    for pair in pairs:
        ck = DEFAULT_COSTS[pair]
        cc = stressed(ck) if stress else ck
        src = "Hantec(long)" if pair in CONFIRMED_SWAP else "PLACEHOLDER honesto"
        lines.append(
            f"  {pair:<8} {cc.spread_pips:>7.2f} {cc.swap_long_pips:>10.2f} "
            f"{cc.swap_short_pips:>11.2f}  {src}"
        )
    lines.append(
        "  (swap negativo = você PAGA por noite; só EURUSD long vem de número público citado.)"
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
        return "\n".join(lines) + "\n"

    # ---- roda todas as configs ----
    results: list[ConfigResult] = []
    skipped: list[tuple[str, str, int, str]] = []  # (par, tf, swing, motivo)
    for (pair, tf, sw) in grid:
        ck = DEFAULT_COSTS[pair]
        cc = stressed(ck) if stress else ck
        res, reason = run_config(pair, tf, sw, loaded[(pair, tf)], cc)
        if res is not None:
            results.append(res)
        else:
            skipped.append((pair, tf, sw, reason))

    # Configs sem amostra suficiente — reportadas com o MOTIVO (não somem do mapa).
    # No baseline FIMATHE em D1, é aqui que o gate PCM reprova TODOS os rompimentos.
    if skipped:
        lines.append("=" * 100)
        lines.append("CONFIGS SEM AMOSTRA SUFICIENTE (reportadas, não suavizadas):")
        lines.append("=" * 100)
        for (pair, tf, sw, reason) in skipped:
            lines.append(f"  {pair:<8} {tf:<4} sw={sw:<3} -> {reason}")
        # Resumo por timeframe inteiramente vazio (ex.: D1).
        empty_tfs = [
            tf for tf in timeframes
            if all(r.timeframe != tf for r in results)
            and any(s[1] == tf for s in skipped)
        ]
        for tf in empty_tfs:
            lines.append(
                f"  >>> {tf}: NENHUMA config gerou trades — a FIMATHE baseline não opera "
                f"neste timeframe (gate PCM/breakout do motor atual)."
            )
        lines.append("")

    if not results:
        lines.append("Nenhuma config gerou trades suficientes (>=5). FALHA por amostra.")
        lines.append("=" * 100)
        lines += [
            "##############################################################################",
            "#   F A L H A   —   BASELINE FIMATHE NAO GERA TRADES EM FOREX                 #",
            "#   O gate PCM/breakout-strength do motor atual reprova todos os rompimentos. #",
            "#   Harness pronto p/ reusar quando a tecnica pura corrigir a entrada.        #",
            "##############################################################################",
        ]
        return "\n".join(lines) + "\n"

    # ---- tabela por par/timeframe (melhor swing por par/tf no veredito) ----
    header = (
        f"  {'par':<8} {'TF':<4} {'sw':>3} {'#tr':>4} {'hold_d':>7} {'win%':>6} "
        f"{'bruto/tr':>10} {'líq/tr':>10} {'swap/tr':>10} {'sprd/tr':>9} "
        f"{'noites':>7} {'Shp_líq':>8} {'DSR':>6} {'B&H':>9} veredito"
    )
    lines.append("=" * 100)
    lines.append("RESULTADO POR PAR / TIMEFRAME / SWING (líquido de spread+swap):")
    lines.append("=" * 100)

    # agrupa por timeframe para legibilidade
    any_pass = False
    swap_share_acc: list[float] = []
    for tf in timeframes:
        tf_res = [r for r in results if r.timeframe == tf]
        if not tf_res:
            continue
        lines.append(f"\n--- {tf} (barras/ano={PERIODS_PER_YEAR[tf]}) ---")
        lines.append(header)
        for r in sorted(tf_res, key=lambda x: (x.pair, x.swing)):
            v = evaluate_edge(
                r.trade_net_returns, n_trials=n_trials,
                periods_per_year=r.bars_per_year, min_sharpe_annual=SHARPE_PASS,
                dsr_threshold=DSR_PASS,
            )
            beats_bh = r.net_total_ret > r.buyhold_ret
            net_pos = r.net_per_trade > 0
            passes = bool(
                v.dsr >= DSR_PASS and r.sharpe_net_annual >= SHARPE_PASS
                and net_pos and beats_bh
            )
            any_pass = any_pass or passes
            flag = "PASSA" if passes else "FALHA"
            # quanto o swap comeu do edge bruto (em pontos de retorno por trade)
            if r.gross_per_trade != 0:
                swap_share = -r.swap_per_trade / abs(r.gross_per_trade)
                swap_share_acc.append(swap_share)
            lines.append(
                f"  {r.pair:<8} {tf:<4} {r.swing:>3} {r.n_trades:>4} {r.hold_days:>7.1f} "
                f"{r.win_rate*100:>5.1f} {_fmt_pct(r.gross_per_trade):>10} "
                f"{_fmt_pct(r.net_per_trade):>10} {_fmt_pct(r.swap_per_trade):>10} "
                f"{_fmt_pct(r.spread_per_trade):>9} {r.nights_per_trade:>7.1f} "
                f"{r.sharpe_net_annual:>8.2f} {v.dsr:>6.3f} {_fmt_pct(r.buyhold_ret):>9} {flag}"
            )

        # PBO por timeframe: matriz (trades alinhados? não — usamos retorno por trade
        # por config; CSCV precisa de linhas comparáveis. Construímos a matriz de
        # retornos por trade truncada ao menor comprimento entre configs do TF).
        mats = [r.trade_net_returns for r in tf_res]
        min_len = min(len(m) for m in mats)
        if len(mats) >= 2 and min_len >= 20:
            M = np.column_stack([m[:min_len] for m in mats])
            pbo = probability_of_backtest_overfitting(M, n_splits=10)
            lines.append(
                f"  PBO ({tf}, seleção entre {len(mats)} configs, informativo): {pbo:.2f}"
            )

    # ---- separável: pares de menor volatilidade ----
    lines.append("\n" + "=" * 100)
    lines.append("RECORTE — PARES DE MENOR VOLATILIDADE (EURGBP, USDCHF):")
    lines.append("=" * 100)
    low = [r for r in results if r.pair in LOW_VOL_PAIRS]
    if low:
        best = sorted(low, key=lambda x: x.sharpe_net_annual, reverse=True)[:4]
        for r in best:
            beats = "supera B&H" if r.net_total_ret > r.buyhold_ret else "NÃO supera B&H"
            lines.append(
                f"  {r.pair} {r.timeframe} sw={r.swing}: Shp_líq={r.sharpe_net_annual:.2f} "
                f"líq/tr={_fmt_pct(r.net_per_trade)} total_líq={_fmt_pct(r.net_total_ret)} "
                f"(B&H={_fmt_pct(r.buyhold_ret)}; {beats})"
            )
    else:
        lines.append("  sem dados para os pares de menor vol.")

    # ---- quanto o SWAP comeu do edge ----
    lines.append("\n" + "=" * 100)
    lines.append("QUANTO O SWAP COMEU DO EDGE:")
    lines.append("=" * 100)
    gross_avg = float(np.mean([r.gross_per_trade for r in results]))
    net_avg = float(np.mean([r.net_per_trade for r in results]))
    swap_avg = float(np.mean([r.swap_per_trade for r in results]))
    spread_avg = float(np.mean([r.spread_per_trade for r in results]))
    nights_avg = float(np.mean([r.nights_per_trade for r in results]))
    holds_avg = float(np.mean([r.hold_days for r in results]))
    lines.append(
        f"  Média entre {len(results)} configs: bruto/tr={_fmt_pct(gross_avg)}  "
        f"swap/tr={_fmt_pct(swap_avg)}  spread/tr={_fmt_pct(spread_avg)}  "
        f"líq/tr={_fmt_pct(net_avg)}"
    )
    lines.append(
        f"  Hold médio={holds_avg:.1f} dias  noites médias/trade={nights_avg:.1f}"
    )
    if gross_avg > 0:
        lines.append(
            f"  O SWAP sozinho consumiu ~{(-swap_avg / gross_avg)*100:.0f}% do retorno BRUTO médio; "
            f"spread+swap consumiram ~{(-(swap_avg+spread_avg)/gross_avg)*100:.0f}%."
        )
    elif gross_avg <= 0:
        lines.append(
            "  Retorno BRUTO médio já é <=0: a FIMATHE baseline não gera edge bruto positivo "
            "suficiente; o swap apenas enterra mais fundo."
        )

    # ---- veredito final em letras grandes ----
    lines.append("\n" + "=" * 100)
    if any_pass:
        lines.append("VEREDITO BASELINE: ao menos uma config PASSOU o kill-criterion. Revisar abaixo.")
    else:
        lines += [
            "##############################################################################",
            "#                                                                            #",
            "#   F A L H A   —   BASELINE FIMATHE EM FOREX NAO TEM EDGE                    #",
            "#                                                                            #",
            "#   Nenhuma config (par x timeframe x swing) passou: DSR>=0.95 E              #",
            "#   Sharpe_liq>=1.0 E liquido>0 E supera buy&hold, simultaneamente.           #",
            "#   Spread + SWAP overnight comem o edge do rompimento. O harness fica        #",
            "#   PRONTO para reusar quando a tecnica pura corrigir entrada/timeframe.      #",
            "#                                                                            #",
            "##############################################################################",
        ]
    lines.append("=" * 100)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tribunal honesto da FIMATHE em forex (baseline; spread+swap)"
    )
    parser.add_argument("--report-file", default="data/fimathe_forex_verdict.txt")
    parser.add_argument("--pairs", default="", help="CSV; vazio = universo default")
    parser.add_argument(
        "--timeframes", default="D1,H4,H1", help="CSV de D1,H4,H1 (default: todos)"
    )
    parser.add_argument(
        "--stress", action="store_true", help="também roda o cenário de custo estressado (1.5x)"
    )
    args = parser.parse_args(argv)

    pairs = (
        [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
        if args.pairs else DEFAULT_PAIRS
    )
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    tfs = [t for t in tfs if t in TIMEFRAMES] or list(TIMEFRAMES)

    text = build_report(pairs, tfs, stress=False)
    if args.stress:
        text += "\n\n" + build_report(pairs, tfs, stress=True)

    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
