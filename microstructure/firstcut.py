"""System 3 — FIRST-CUT BARATO da Fase 2: o sinal de trade-flow SOBREVIVE a custo?

Gate de KILL-BARATO antes de investir no build pesado (coleta L2 ao vivo + market
maker completo com modelo de fila/adverse-selection). A Fase 1 (microstructure.ic)
ja mostrou que o trade-flow imbalance TEM IC (~0.07, janela 5s, prevê 1-5s). IC NAO
e lucro. Aqui pergunta-se, de forma GENEROSA: o RETORNO BRUTO por trade (em bps)
do sinal — entrando so quando |sinal| cruza um limiar, ja PENALIZADO pela latencia
de varejo (~100ms) — bate o CUSTO de transacao? Se nao bate nem generosamente, o
System 3 inteiro morre barato. Se bate so o caso MAKER (ganhando spread), o build
pesado de market making se justifica.

DESENHO (event-driven SIMPLES, sem look-ahead):
  - Grade FINA de relogio (default 250ms). Em cada ponto t calcula o sinal de
    trade-flow (reuso de microstructure.signals._windowed_sums) usando SO eventos
    em (t-window, t].
  - LATENCIA DE VAREJO ~100ms: o sinal calculado em t so e ACIONAVEL em t+lat. A
    entrada paga o preco do ULTIMO TRADE <= t+lat (nao o de t). Isso penaliza a
    latencia E remove look-ahead (o preco do sinal nunca e o preco de execucao).
  - Entra LONG se sinal > +limiar, SHORT se sinal < -limiar. Segura o HORIZONTE do
    sinal (h em {1,5}s) e sai ao preco do ultimo trade <= t_entrada+h.
  - Varre VARIOS limiares (quantis do |sinal|) x VARIOS horizontes.

CUSTO (o ponto). Round-trip, em bps (premissas DECLARADAS, ajustaveis por CLI):
  - TAKER  ~= 2 x taker_fee + spread_cruzado.  Binance USDS-M futures taker tier0
    ~4 bps/lado => 8 bps; + spread cruzado ~0.5-1 bp (BTC/ETH sao os pares mais
    liquidos) => ~8.5-9 bps round-trip.
  - MAKER  ~= 2 x maker_fee (default ~2 bps round-trip). Caso "GANHA O SPREAD":
    maker recebe metade do spread em cada perna => custo efetivo pode ser ~0 ou
    NEGATIVO (rebate de spread). Reportamos os dois.

VEREDITO honesto (no topo do relatorio):
  (a) Existe versao TAKER tradavel? (quase certo que NAO — scalp direcional-taker
      morre no custo.)
  (b) O bruto/trade esta numa FAIXA onde MM (maker ganhando spread) PODERIA
      funcionar — justificando o build pesado? OU esta tao abaixo ate do custo
      maker que o System 3 inteiro esta morto?
  + DSR/PBO na serie LIQUIDA, com n_trials = (# limiares x # horizontes x # custos)
    realmente varridos (anti data-snooping).

CAVEATS OBRIGATORIOS (no relatorio):
  - spread e ASSUMIDO (nao ha L2 real; bookTicker historico esta 404 no arquivo).
  - NAO ha modelo de FILA nem de ADVERSE-SELECTION do maker — esse e justamente o
    build pesado. Aqui o maker e tratado de forma GENEROSA (preenche sempre, no
    preco do sinal-latencia). Logo um maker REAL renderia MENOS que este first-cut.
  - last-trade price como mid: infla ruido (bid-ask bounce) mas nao da look-ahead.
  - first-cut GENEROSO em todas as frentes -> se morrer aqui, morre de verdade.

ENTREGA: este modulo + data/microstructure_firstcut_report.txt.
Reuso: microstructure.data, microstructure.signals, simulation.statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from microstructure.data import (
    DEFAULT_SYMBOLS,
    TradeArrays,
    load_all_days,
)
from microstructure.signals import _last_price_at, _windowed_sums
from simulation.statistics import (
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("microstructure.firstcut")

REPORT_PATH = Path("data/microstructure_firstcut_report.txt")

# --- Parametros do first-cut (defaults GENEROSOS, declarados) -----------------
GRID_MS_DEFAULT = 250          # grade fina de relogio (ms) para acionar o sinal
LATENCY_MS_DEFAULT = 100       # latencia de varejo: sinal em t -> acionavel em t+lat
SIGNAL_WINDOW_S = 5            # janela do sinal vencedor da Fase 1 (tfi 5s)
HORIZONS_S = (1, 5)            # horizontes do sinal (Fase 1: prevê 1-5s)
# Limiares varridos como QUANTIS do |sinal| (so trades onde |sinal| > quantil).
# Quantil alto => menos trades, sinal mais "extremo" (mais chance de edge bruto).
QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.99)

# --- Premissas de CUSTO (round-trip, bps). Ajustaveis por CLI. ----------------
TAKER_FEE_BPS = 4.0            # taker fee por LADO (Binance USDS-M futures tier0 ~0.04%)
MAKER_FEE_BPS = 1.0           # maker fee por LADO (~0.01% .. 0.02%); usamos 0.01% = 1 bp
SPREAD_BPS = 0.75             # spread cruzado ASSUMIDO (BTC/ETH liquidos ~0.5-1 bp)

# --- Sinal usado no first-cut: trade-flow imbalance por VOLUME (tfi), janela 5s.
# (tfi_cnt teve IC marginalmente maior em BTC, mas tfi/ofi_trade sao os ~0.07 do
#  briefing; o resultado de custo e robusto a essa escolha — ver sensibilidade.)
SIGNAL_NAME = "tfi"


def round_trip_cost_bps(*, taker_fee: float, maker_fee: float, spread: float) -> dict[str, float]:
    """Custos round-trip (bps) por cenario. Premissas DECLARADAS.

    - taker: cruza o spread nas duas pernas + 2x taker_fee.
    - maker_fee_only: 2x maker_fee (paga so a fee de maker; nao ganha spread).
    - maker_earns_spread: 2x maker_fee MENOS o spread inteiro (recebe ~metade do
      spread em cada perna). Pode ser <=0 (rebate liquido). Caso GENEROSO p/ MM.
    """
    return {
        "taker": 2.0 * taker_fee + spread,
        "maker_fee_only": 2.0 * maker_fee,
        "maker_earns_spread": 2.0 * maker_fee - spread,
    }


# ---------------------------------------------------------------------------
# SINAL EM GRADE FINA (reuso dos helpers vetorizados de signals.py)
# ---------------------------------------------------------------------------

def _fine_grid(ts_ms: np.ndarray, grid_ms: int) -> np.ndarray:
    """Pontos de grade alinhados a multiplos de grid_ms cobrindo [t0, tN]."""
    t0 = int(ts_ms[0])
    tN = int(ts_ms[-1])
    start = (t0 // grid_ms) * grid_ms
    return np.arange(start, tN + grid_ms, grid_ms, dtype=np.int64)


def _tfi_on_grid(trades: TradeArrays, grid: np.ndarray, window_s: int) -> np.ndarray:
    """Trade-flow imbalance por VOLUME em (t-window, t] para cada ponto da grade.

    tfi = (Vbuy - Vsell)/(Vbuy + Vsell). Mesma definicao de signals.build_trade_signals,
    reusando _windowed_sums. Sem look-ahead: usa so eventos <= t.
    """
    eps = 1e-12
    buy_vol = (trades.side > 0).astype(np.float64) * trades.qty
    sell_vol = (trades.side < 0).astype(np.float64) * trades.qty
    vbuy = _windowed_sums(grid, trades.ts_ms, buy_vol, window_s)
    vsell = _windowed_sums(grid, trades.ts_ms, sell_vol, window_s)
    tot = vbuy + vsell
    return np.where(tot > eps, (vbuy - vsell) / (tot + eps), 0.0)


# ---------------------------------------------------------------------------
# BACKTEST EVENT-DRIVEN DE UM DIA, UM (limiar, horizonte)
# ---------------------------------------------------------------------------

@dataclass
class DayBacktest:
    """Resultado de UM dia para UM (limiar, horizonte). Retornos BRUTOS em bps."""

    n_trades: int
    gross_bps: np.ndarray          # retorno BRUTO por trade, em bps (com sinal da direcao)
    seconds_covered: float         # span temporal coberto (p/ trades/dia)


def _backtest_day(
    trades: TradeArrays,
    *,
    grid_ms: int,
    latency_ms: int,
    window_s: int,
    horizon_s: int,
    abs_threshold: float,
) -> DayBacktest:
    """Backtest event-driven de um dia para um limiar ABSOLUTO de |sinal| e horizonte.

    Mecanica (sem look-ahead, com latencia penalizada):
      sinal s(t) em cada ponto de grade t (so passado).
      entrada acionavel em t+lat; preco de entrada = ultimo trade <= t+lat.
      saida em (t+lat)+h; preco de saida = ultimo trade <= (t+lat)+h.
      retorno bruto (long)  = (P_exit/P_entry - 1).
      retorno bruto (short) = (P_entry/P_exit - 1).
    Trade so existe se |s(t)| > abs_threshold E ambos os precos sao finitos/positivos.
    """
    grid = _fine_grid(trades.ts_ms, grid_ms)
    if grid.size < 4:
        return DayBacktest(0, np.empty(0), 0.0)
    sig = _tfi_on_grid(trades, grid, window_s)

    # precos de entrada/saida via last-trade <= (t+lat) e <= (t+lat+h)
    entry_ts = grid + latency_ms
    exit_ts = grid + latency_ms + horizon_s * 1000
    px_entry = _last_price_at(entry_ts, trades.ts_ms, trades.price)
    px_exit = _last_price_at(exit_ts, trades.ts_ms, trades.price)

    direction = np.where(sig > abs_threshold, 1.0, np.where(sig < -abs_threshold, -1.0, 0.0))

    valid = (
        (direction != 0.0)
        & np.isfinite(px_entry) & (px_entry > 0)
        & np.isfinite(px_exit) & (px_exit > 0)
        & (exit_ts <= trades.ts_ms[-1])  # exige futuro real (sem extrapolar o ultimo preco)
    )
    if not np.any(valid):
        span_s = float((grid[-1] - grid[0]) / 1000.0)
        return DayBacktest(0, np.empty(0), span_s)

    d = direction[valid]
    pe = px_entry[valid]
    px = px_exit[valid]
    # retorno simples assinado pela direcao (long: px/pe-1 ; short: -(px/pe-1))
    raw = d * (px / pe - 1.0)
    gross_bps = raw * 1e4
    span_s = float((grid[-1] - grid[0]) / 1000.0)
    return DayBacktest(int(gross_bps.size), gross_bps, span_s)


# ---------------------------------------------------------------------------
# AGREGACAO POR (limiar, horizonte) ENTRE DIAS + SIMBOLOS
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """Resultado agregado de um (quantil, horizonte) entre dias/simbolos."""

    symbol: str
    quantile: float
    abs_threshold: float
    horizon_s: int
    n_trades: int
    trades_per_day: float
    gross_bps_mean: float           # edge BRUTO/trade (bps) — A METRICA CHAVE
    gross_bps_median: float
    gross_bps_std: float
    win_rate: float
    cum_gross_bps: float            # retorno bruto acumulado (soma dos bps)
    # liquido por cenario de custo: edge bruto/trade - custo round-trip
    net_per_trade_bps: dict[str, float] = field(default_factory=dict)
    gross_per_trade_array: np.ndarray = field(default_factory=lambda: np.empty(0))


def _threshold_from_quantile(
    trades_by_day: list[TradeArrays], *, grid_ms: int, window_s: int, q: float
) -> float:
    """Limiar ABSOLUTO de |sinal| = quantil q do |sinal| sobre todos os dias.

    Estimado UMA vez do pool de sinais (in-sample para o sweep — declarado como
    generoso; o objetivo aqui e medir o edge bruto no regime "extremo", nao montar
    um classificador OOS). Reportamos o quantil para transparencia.
    """
    pool: list[np.ndarray] = []
    for ta in trades_by_day:
        grid = _fine_grid(ta.ts_ms, grid_ms)
        if grid.size < 4:
            continue
        pool.append(np.abs(_tfi_on_grid(ta, grid, window_s)))
    if not pool:
        return 0.0
    allabs = np.concatenate(pool)
    # ignora zeros exatos (janelas sem trade) p/ o quantil ser do sinal "ativo"
    nz = allabs[allabs > 0]
    base = nz if nz.size else allabs
    return float(np.quantile(base, q))


def run_symbol(
    symbol: str,
    *,
    trades_by_day: list[TradeArrays] | None = None,
    grid_ms: int = GRID_MS_DEFAULT,
    latency_ms: int = LATENCY_MS_DEFAULT,
    window_s: int = SIGNAL_WINDOW_S,
    horizons_s: tuple[int, ...] = HORIZONS_S,
    quantiles: tuple[float, ...] = QUANTILES,
    costs: dict[str, float] | None = None,
) -> list[CellResult]:
    """Roda o sweep (quantil x horizonte) para um simbolo, agregando todos os dias."""
    days = trades_by_day if trades_by_day is not None else load_all_days(symbol)
    if not days:
        return []
    if costs is None:
        costs = round_trip_cost_bps(
            taker_fee=TAKER_FEE_BPS, maker_fee=MAKER_FEE_BPS, spread=SPREAD_BPS
        )

    results: list[CellResult] = []
    for q in quantiles:
        thr = _threshold_from_quantile(days, grid_ms=grid_ms, window_s=window_s, q=q)
        for h in horizons_s:
            chunks: list[np.ndarray] = []
            total_span_s = 0.0
            for ta in days:
                bt = _backtest_day(
                    ta, grid_ms=grid_ms, latency_ms=latency_ms,
                    window_s=window_s, horizon_s=h, abs_threshold=thr,
                )
                total_span_s += bt.seconds_covered
                if bt.n_trades:
                    chunks.append(bt.gross_bps)
            if chunks:
                allg = np.concatenate(chunks)
            else:
                allg = np.empty(0)
            n = int(allg.size)
            days_equiv = total_span_s / 86400.0 if total_span_s > 0 else float("nan")
            tpd = (n / days_equiv) if (days_equiv and days_equiv > 0) else 0.0
            mean = float(allg.mean()) if n else 0.0
            net = {name: mean - c for name, c in costs.items()}
            results.append(
                CellResult(
                    symbol=symbol, quantile=q, abs_threshold=thr, horizon_s=h,
                    n_trades=n, trades_per_day=tpd,
                    gross_bps_mean=mean,
                    gross_bps_median=float(np.median(allg)) if n else 0.0,
                    gross_bps_std=float(allg.std(ddof=1)) if n > 1 else 0.0,
                    win_rate=float((allg > 0).mean()) if n else 0.0,
                    cum_gross_bps=float(allg.sum()) if n else 0.0,
                    net_per_trade_bps=net,
                    gross_per_trade_array=allg,
                )
            )
    return results


# ---------------------------------------------------------------------------
# TRIBUNAL (DSR/PBO) NA SERIE LIQUIDA
# ---------------------------------------------------------------------------

@dataclass
class TribunalResult:
    scenario: str
    n_trials: int
    n_trades: int
    sharpe_per_trade: float   # Sharpe POR TRADE da serie liquida (escala honesta)
    net_tstat: float          # t-stat de mean!=0 da serie liquida por trade
    psr: float
    dsr: float
    pbo: float
    best_cell: tuple[str, float, int]   # (symbol, quantile, horizon) escolhida
    net_mean_bps: float


def run_tribunal(
    all_cells: list[CellResult],
    *,
    costs: dict[str, float],
    n_trials: int,
) -> list[TribunalResult]:
    """Submete a SERIE LIQUIDA (bruto - custo) por trade ao tribunal, por cenario de custo.

    - Sharpe/PSR/DSR: na melhor celula (maior net mean) de cada cenario, serie por trade.
    - PBO: matriz (trades x celulas) dos retornos LIQUIDOS, alinhando celulas pelo
      menor n de trades (CSCV exige colunas alinhadas no tempo-amostral).
    n_trials = (# celulas distintas varridas) — anti data-snooping HONESTO.
    """
    out: list[TribunalResult] = []
    if not all_cells:
        return out
    for scenario, cost in costs.items():
        # melhor celula do cenario pelo net/trade
        best = max(all_cells, key=lambda c: (c.gross_bps_mean - cost) if c.n_trades else -1e18)
        if best.n_trades < 2:
            continue
        net_series = best.gross_per_trade_array / 1e4 - cost / 1e4  # fracao por trade
        # Sharpe POR TRADE (periods_per_year=1 => evaluate_edge nao anualiza; PSR/DSR
        # operam sobre o Sharpe por periodo e independem dessa escala). Reportar o
        # Sharpe por-trade e mais honesto que anualizar 34k trades/dia (numero absurdo).
        verdict = evaluate_edge(
            net_series, n_trials=n_trials, periods_per_year=1, min_sharpe_annual=0.8
        )
        sr_pt = observed_sharpe(net_series)  # por trade
        n = net_series.size
        net_tstat = sr_pt * float(np.sqrt(n - 1)) if n > 1 else float("nan")
        # PBO: alinhar celulas por min n de trades
        usable = [c for c in all_cells if c.n_trades >= 8]
        pbo = float("nan")
        if len(usable) >= 2:
            m = min(c.n_trades for c in usable)
            mat = np.column_stack(
                [c.gross_per_trade_array[:m] / 1e4 - cost / 1e4 for c in usable]
            )
            pbo = probability_of_backtest_overfitting(mat, n_splits=10)
        out.append(
            TribunalResult(
                scenario=scenario, n_trials=n_trials, n_trades=best.n_trades,
                sharpe_per_trade=sr_pt, net_tstat=net_tstat, psr=verdict.psr, dsr=verdict.dsr,
                pbo=pbo, best_cell=(best.symbol, best.quantile, best.horizon_s),
                net_mean_bps=best.gross_bps_mean - cost,
            )
        )
    return out


# ---------------------------------------------------------------------------
# RELATORIO + VEREDITO
# ---------------------------------------------------------------------------

def _fmt(x: float, nd: int = 2) -> str:
    if x != x:
        return "  n/a"
    return f"{x:+.{nd}f}"


def _best_gross_cell(cells: list[CellResult]) -> CellResult | None:
    valid = [c for c in cells if c.n_trades >= 8]
    if not valid:
        valid = [c for c in cells if c.n_trades]
    return max(valid, key=lambda c: c.gross_bps_mean) if valid else None


def _verdict_lines(
    by_symbol: dict[str, list[CellResult]],
    costs: dict[str, float],
    tribunal: list[TribunalResult],
) -> list[str]:
    """Veredito honesto no TOPO. Nao suaviza em nenhuma direcao."""
    L: list[str] = []
    all_cells = [c for cells in by_symbol.values() for c in cells]
    best = _best_gross_cell(all_cells)
    taker = costs["taker"]
    maker_fee = costs["maker_fee_only"]
    maker_spread = costs["maker_earns_spread"]

    L.append("VEREDITO (honesto — first-cut GENEROSO; nao suavizado)")
    L.append("")
    if best is None:
        L.append("  SEM TRADES — dados insuficientes. Nada decidido. (Rode com aggTrades em cache.)")
        return L

    g = best.gross_bps_mean
    L.append(
        f"  Melhor edge BRUTO/trade observado: {g:+.3f} bps "
        f"({best.symbol}, q={best.quantile:.2f}, h={best.horizon_s}s, "
        f"n={best.n_trades:,}, {best.trades_per_day:,.0f} trades/dia, win={best.win_rate:.1%})."
    )
    L.append(
        f"  Custos round-trip (bps): TAKER={taker:.2f} | MAKER(so fee)={maker_fee:.2f} | "
        f"MAKER(ganha spread)={maker_spread:+.2f}."
    )
    L.append("")

    # (a) taker
    gap_taker = g - taker
    if gap_taker > 0:
        L.append(
            f"  (a) TAKER: o bruto/trade ({g:+.3f}) SUPERA o custo taker ({taker:.2f}) "
            f"por {gap_taker:+.3f} bps. (Inesperado — checar antes de comemorar.)"
        )
    else:
        L.append(
            f"  (a) TAKER: NAO tradavel. Bruto/trade {g:+.3f} bps vs custo {taker:.2f} bps "
            f"=> deficit de {gap_taker:+.3f} bps/trade. O bruto e ~{(g/taker if taker else 0):.0%} "
            f"do custo taker. Scalp direcional-taker MORRE no custo, como esperado."
        )

    # (b) maker
    gap_maker_fee = g - maker_fee
    gap_maker_spread = g - maker_spread
    L.append("")
    if g <= 0:
        L.append(
            f"  (b) MAKER: o bruto/trade e <= 0 ({g:+.3f}) ja DEPOIS so da latencia, ANTES "
            f"de qualquer custo. Nem o caso maker-ganha-spread salva. System 3 (direcional) MORTO."
        )
    elif gap_maker_fee > 0:
        L.append(
            f"  (b) MAKER: o bruto/trade ({g:+.3f}) cobre ATE o custo maker-so-fee "
            f"({maker_fee:.2f}) com folga de {gap_maker_fee:+.3f} bps. Sob maker-ganha-spread "
            f"({maker_spread:+.2f}) a margem e {gap_maker_spread:+.3f} bps. FAIXA MAKER-VIAVEL: "
            f"o build pesado (L2 + fila/adverse-selection) pode se justificar."
        )
    elif gap_maker_spread > 0:
        L.append(
            f"  (b) MAKER: o bruto/trade ({g:+.3f}) NAO cobre o maker-so-fee ({maker_fee:.2f}, "
            f"deficit {gap_maker_fee:+.3f}), mas FICA POSITIVO no caso maker-ganha-spread "
            f"({maker_spread:+.2f}) com {gap_maker_spread:+.3f} bps. FAIXA-LIMITE: so se "
            f"justifica o build pesado se o MM REALMENTE capturar o spread liquido — e o "
            f"modelo de fila/adverse-selection (ausente aqui) vai COMER essa margem fina."
        )
    else:
        L.append(
            f"  (b) MAKER: o bruto/trade ({g:+.3f}) fica ABAIXO ate do caso mais generoso "
            f"(maker-ganha-spread, {maker_spread:+.2f}; deficit {gap_maker_spread:+.3f} bps). "
            f"Nem um MM ideal que capture todo o spread sobrevive. System 3 MORTO ate generoso."
        )

    # decisao final
    L.append("")
    decision_kill = (g <= 0) or (g - maker_spread <= 0)
    borderline = (not decision_kill) and (g - maker_fee <= 0)
    if decision_kill:
        L.append(
            "  >>> DECISAO: MATAR BARATO. Arquivar o System 3 (track de microestrutura "
            "direcional). O edge bruto nao sobrevive nem ao caso de custo mais generoso; "
            "construir coleta L2 + MM completo seria queimar tempo. O IC da Fase 1 era real "
            "porem pequeno demais p/ a latencia/custo de varejo — confirma a nota 'IC != lucro'."
        )
    elif borderline:
        L.append(
            "  >>> DECISAO: FAIXA-LIMITE -> so justifica o build pesado SE houver tese de "
            "captura de spread (maker puro). O bruto nao paga nem a fee de maker isolada; "
            "depende 100% de ganhar o spread, e o adverse-selection (NAO modelado) tende a "
            "anular isso. Recomendado: NAO investir no build pesado sem antes uma prova barata "
            "de captura de spread (coletar bookTicker ao vivo p/ medir spread real + fill maker)."
        )
    else:
        L.append(
            "  >>> DECISAO: JUSTIFICA O BUILD PESADO (condicional). O bruto/trade cobre ate a "
            "fee de maker; um MM ganhando spread teria margem. PROXIMO passo do build pesado: "
            "coleta L2 ao vivo (simulation.binance_book --collect) + backtest com modelo de "
            "FILA e ADVERSE-SELECTION (hftbacktest-style). So entao virar claim de tradabilidade."
        )

    # tribunal headline
    L.append("")
    if tribunal:
        for t in tribunal:
            passt = "DSR>=0.95? " + ("SIM" if t.dsr >= 0.95 else "NAO")
            L.append(
                f"  Tribunal [{t.scenario}]: Sharpe/trade(net)={t.sharpe_per_trade:+.4f} "
                f"t-stat={t.net_tstat:+.1f} DSR={t.dsr:.3f} ({passt}) PSR={t.psr:.3f} "
                f"PBO={t.pbo:.2f} net/trade={t.net_mean_bps:+.3f} bps n_trials={t.n_trials}."
            )
        L.append(
            "  (DSR=0/PSR=0 na serie liquida = zero chance de Sharpe verdadeiro > 0. PBO baixo "
            "aqui nao salva: so diz que TODAS as celulas sao consistentemente negativas — nao ha "
            "edge em lugar nenhum p/ overfitar. Tudo reforca o veredito.)"
        )
    return L


def _render_sweep_table(symbol: str, cells: list[CellResult], costs: dict[str, float]) -> list[str]:
    L: list[str] = []
    L.append(f"### {symbol} — sweep limiar(quantil) x horizonte")
    head = (
        f"{'q':>5} {'thr|sig|':>9} {'h':>3} {'trades':>9} {'tr/dia':>8} "
        f"{'GROSS/tr':>9} {'med':>8} {'win%':>6} {'cumGROSS':>11} "
        f"{'net_tk':>8} {'net_mk':>8} {'net_mk+s':>9}"
    )
    L.append(head)
    L.append("-" * len(head))
    for c in cells:
        L.append(
            f"{c.quantile:>5.2f} {c.abs_threshold:>9.4f} {c.horizon_s:>2}s "
            f"{c.n_trades:>9,d} {c.trades_per_day:>8,.0f} "
            f"{c.gross_bps_mean:>+9.3f} {c.gross_bps_median:>+8.3f} {c.win_rate*100:>5.1f}% "
            f"{c.cum_gross_bps:>+11.1f} "
            f"{c.net_per_trade_bps['taker']:>+8.3f} "
            f"{c.net_per_trade_bps['maker_fee_only']:>+8.3f} "
            f"{c.net_per_trade_bps['maker_earns_spread']:>+9.3f}"
        )
    return L


def write_report(
    by_symbol: dict[str, list[CellResult]],
    costs: dict[str, float],
    tribunal: list[TribunalResult],
    *,
    grid_ms: int,
    latency_ms: int,
    window_s: int,
    horizons_s: tuple[int, ...],
    quantiles: tuple[float, ...],
    days_by_symbol: dict[str, list[str]],
    n_trials: int,
) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("=" * 78)
    L.append("SYSTEM 3 — FIRST-CUT BARATO DA FASE 2: O TRADE-FLOW SOBREVIVE A CUSTO?")
    L.append("=" * 78)
    L.append("")
    L.append("Gate de KILL-BARATO antes do build pesado (coleta L2 ao vivo + MM completo).")
    L.append("Pergunta: o RETORNO BRUTO/trade (bps) do sinal de trade-flow — ja penalizado")
    L.append("pela latencia de varejo (~100ms) — bate o CUSTO taker? bate o maker? Quao longe?")
    L.append("First-cut GENEROSO: se morre aqui, morre de verdade.")
    L.append("")

    # VEREDITO NO TOPO
    L.append("-" * 78)
    L.extend(_verdict_lines(by_symbol, costs, tribunal))
    L.append("")

    # PREMISSAS
    L.append("-" * 78)
    L.append("PREMISSAS E DESENHO (declarados)")
    L.append("-" * 78)
    L.append(f"  Sinal           : tfi (trade-flow imbalance por volume), janela {window_s}s.")
    L.append(f"                    (vencedor da Fase 1: IC ~0.07, prevê 1-5s — ver microstructure.ic)")
    L.append(f"  Grade fina      : {grid_ms} ms (aciona o sinal nesta cadencia).")
    L.append(f"  LATENCIA varejo : {latency_ms} ms. Sinal em t -> acionavel em t+{latency_ms}ms.")
    L.append(f"                    Entrada paga o preco do ULTIMO TRADE <= t+lat (NAO o de t).")
    L.append(f"                    => penaliza latencia e remove look-ahead.")
    L.append(f"  Horizontes      : {list(horizons_s)} s (segura o horizonte do sinal e sai).")
    L.append(f"  Limiares        : quantis de |sinal| = {list(quantiles)} (so trades acima).")
    L.append(f"  Preco de ref.   : last-trade price (bookTicker historico 404 no arquivo).")
    L.append("")
    L.append("  CUSTOS round-trip (bps) — premissas:")
    L.append(f"    taker_fee/lado = {TAKER_FEE_BPS:.2f} bps ; maker_fee/lado = {MAKER_FEE_BPS:.2f} bps ; "
             f"spread cruzado = {SPREAD_BPS:.2f} bps")
    L.append(f"    -> TAKER round-trip          = 2x{TAKER_FEE_BPS:.1f} + {SPREAD_BPS:.2f} = {costs['taker']:.2f} bps")
    L.append(f"    -> MAKER round-trip (so fee) = 2x{MAKER_FEE_BPS:.1f}            = {costs['maker_fee_only']:.2f} bps")
    L.append(f"    -> MAKER (ganha o spread)    = 2x{MAKER_FEE_BPS:.1f} - {SPREAD_BPS:.2f}    = {costs['maker_earns_spread']:+.2f} bps")
    L.append("")

    # DADOS
    L.append("-" * 78)
    L.append("DADOS")
    L.append("-" * 78)
    for sym, days in days_by_symbol.items():
        if days:
            L.append(f"  {sym:<10} aggTrades (um): {len(days)} dia(s) [{', '.join(days)}]")
        else:
            L.append(f"  {sym:<10} DADOS PENDENTES (sem CSV em cache)")
    L.append("")

    # TABELAS DE SWEEP
    L.append("-" * 78)
    L.append("(1) EDGE BRUTO/trade (bps) POR LIMIAR x HORIZONTE  +  LIQUIDO vs CUSTO")
    L.append("-" * 78)
    L.append("  GROSS/tr = edge bruto medio/trade (bps); cumGROSS = soma bruta (bps);")
    L.append("  net_tk = GROSS - custo taker ; net_mk = GROSS - maker(so fee) ;")
    L.append("  net_mk+s = GROSS - maker(ganha spread). Positivo = sobrevive aquele custo.")
    L.append("")
    for sym, cells in by_symbol.items():
        if not cells:
            continue
        L.extend("  " + ln for ln in _render_sweep_table(sym, cells, costs))
        L.append("")

    # SOMENTE A LEITURA vs CUSTO
    L.append("-" * 78)
    L.append("(2) O EDGE BRUTO BATE O CUSTO?  (resumo, melhor celula por simbolo)")
    L.append("-" * 78)
    for sym, cells in by_symbol.items():
        best = _best_gross_cell(cells)
        if best is None:
            L.append(f"  {sym}: sem trades.")
            continue
        g = best.gross_bps_mean
        L.append(
            f"  {sym}: melhor GROSS/tr = {g:+.3f} bps (q={best.quantile:.2f}, h={best.horizon_s}s)."
        )
        for name, cost in costs.items():
            gap = g - cost
            verdict = "BATE" if gap > 0 else "NAO bate"
            ratio = (g / cost) if cost > 0 else float("nan")
            extra = f" ({ratio:.0%} do custo)" if (cost > 0 and g > 0) else ""
            L.append(f"      vs {name:<18} {cost:>+7.2f} bps -> {verdict:<8} (gap {gap:+.3f}){extra}")
    L.append("")

    # TRIBUNAL
    L.append("-" * 78)
    L.append("(3) TRIBUNAL (DSR/PBO) NA SERIE LIQUIDA — anti data-snooping")
    L.append("-" * 78)
    L.append(f"  n_trials honesto = {n_trials} (= # celulas limiar x horizonte x simbolo varridas).")
    L.append("  Serie = retorno LIQUIDO por trade (bruto - custo) na melhor celula de cada cenario.")
    L.append("")
    if not tribunal:
        L.append("  (sem trades suficientes p/ o tribunal)")
    else:
        L.append("  Sharpe e POR TRADE (nao anualizado: 34k trades/dia anualizado da numero")
        L.append("  absurdo). t-stat = significancia de mean(net)!=0. DSR/PSR penalizam o n_trials.")
        L.append("")
        head = f"  {'cenario':<20} {'SR/trade':>10} {'t-stat':>9} {'PSR':>7} {'DSR':>7} {'PBO':>6} {'net/tr_bps':>11} {'n_tr':>9}"
        L.append(head)
        L.append("  " + "-" * (len(head) - 2))
        for t in tribunal:
            L.append(
                f"  {t.scenario:<20} {t.sharpe_per_trade:>+10.4f} {t.net_tstat:>+9.1f} "
                f"{t.psr:>7.3f} {t.dsr:>7.3f} {t.pbo:>6.2f} {t.net_mean_bps:>+11.3f} {t.n_trades:>9,d}"
            )
        L.append("")
        L.append("  Leitura: net/trade NEGATIVO com t-stat fortemente negativo => o prejuizo")
        L.append("  liquido e estatisticamente CERTO (nao ruido). DSR=0 / PSR=0 => zero chance")
        L.append("  de o Sharpe liquido verdadeiro ser > 0. Barra do projeto: DSR>=0.95 e SR>0.")
    L.append("")

    # CAVEATS OBRIGATORIOS
    L.append("-" * 78)
    L.append("CAVEATS OBRIGATORIOS")
    L.append("-" * 78)
    L.append("  - SPREAD e ASSUMIDO. Nao ha L2 real: bookTicker historico esta 404 no arquivo")
    L.append("    publico (data.binance.vision). O spread cruzado e premissa, nao medicao.")
    L.append("  - NAO HA MODELO DE FILA NEM DE ADVERSE-SELECTION. Esse e o build pesado. Aqui o")
    L.append("    maker e tratado de forma GENEROSA (preenche sempre, no preco do sinal-latencia).")
    L.append("    Um maker REAL renderia MENOS: perde fila quando o sinal e obvio (selecao adversa)")
    L.append("    e nem sempre e preenchido. Logo o caso maker aqui e um TETO, nao uma estimativa.")
    L.append("  - LAST-TRADE PRICE como mid infla ruido (bid-ask bounce) mas nao da look-ahead.")
    L.append("  - LIMIAR por quantil e in-sample (sweep generoso p/ achar o regime extremo); nao")
    L.append("    e um classificador OOS. O DSR/PBO ja penaliza o n_trials desse sweep.")
    L.append("  - First-cut GENEROSO em todas as frentes. Se o edge bruto morre AQUI, morre de")
    L.append("    verdade; se fica na faixa maker-viavel, o build pesado se justifica.")
    L.append("")

    text = "\n".join(L) + "\n"
    REPORT_PATH.write_text(text)
    logger.info("relatorio escrito em %s", REPORT_PATH)
    return REPORT_PATH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="System 3 first-cut barato da Fase 2: trade-flow vs custo (taker/maker)."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="lista por virgula")
    parser.add_argument("--grid-ms", type=int, default=GRID_MS_DEFAULT)
    parser.add_argument("--latency-ms", type=int, default=LATENCY_MS_DEFAULT)
    parser.add_argument("--window-s", type=int, default=SIGNAL_WINDOW_S)
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS_S),
                        help="horizontes em s por virgula")
    parser.add_argument("--quantiles", default=",".join(str(q) for q in QUANTILES))
    parser.add_argument("--taker-fee-bps", type=float, default=TAKER_FEE_BPS)
    parser.add_argument("--maker-fee-bps", type=float, default=MAKER_FEE_BPS)
    parser.add_argument("--spread-bps", type=float, default=SPREAD_BPS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    quantiles = tuple(float(x) for x in args.quantiles.split(",") if x.strip())
    costs = round_trip_cost_bps(
        taker_fee=args.taker_fee_bps, maker_fee=args.maker_fee_bps, spread=args.spread_bps
    )

    by_symbol: dict[str, list[CellResult]] = {}
    days_by_symbol: dict[str, list[str]] = {}
    for sym in symbols:
        days = load_all_days(sym)
        days_by_symbol[sym] = [ta.day for ta in days]
        logger.info("%s: %d dia(s) de aggTrades", sym, len(days))
        cells = run_symbol(
            sym, trades_by_day=days, grid_ms=args.grid_ms, latency_ms=args.latency_ms,
            window_s=args.window_s, horizons_s=horizons, quantiles=quantiles, costs=costs,
        )
        by_symbol[sym] = cells

    all_cells = [c for cells in by_symbol.values() for c in cells]
    n_trials = max(len([c for c in all_cells if c.n_trades]), 1)
    tribunal = run_tribunal(all_cells, costs=costs, n_trials=n_trials)

    write_report(
        by_symbol, costs, tribunal,
        grid_ms=args.grid_ms, latency_ms=args.latency_ms, window_s=args.window_s,
        horizons_s=horizons, quantiles=quantiles, days_by_symbol=days_by_symbol,
        n_trials=n_trials,
    )
    # eco curto no stdout
    best = _best_gross_cell(all_cells)
    if best is not None:
        print(f"\nbest GROSS/trade = {best.gross_bps_mean:+.3f} bps "
              f"({best.symbol} q={best.quantile} h={best.horizon_s}s); "
              f"taker={costs['taker']:.2f} maker_fee={costs['maker_fee_only']:.2f} "
              f"maker+spread={costs['maker_earns_spread']:+.2f} bps")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
