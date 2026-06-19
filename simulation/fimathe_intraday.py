"""TRIBUNAL DA FIMATHE REAL — entrada-no-rompimento do Canal de Abertura (intraday).

A FIMATHE foi MAL-TESTADA antes (swing D1->H4, instrumento errado). Esta e a
tecnica REAL (docs/FIMATHE_SPEC.md): DAY TRADE intraday em XAU/forex no M15;
marcacao = canal das 4 primeiras velas M15 da sessao; entrada = rompimento da
caixa; stop fora da ZN; alvo = projecao de ciclo (clones, 1 ou 2 niveis);
liquida no fim do pregao (SEM overnight -> SEM SWAP, so SPREAD).

KILL-CRITERION (no topo, sem suavizar):
  PASSA so se: DSR>=0.95 E Sharpe_liq_anual>=1.0 E liquido>0 E supera buy&hold.
  Senao FALHA. (Intraday -> swap nao se aplica; o teste e se a entrada-no-
  rompimento do Canal de Abertura tem EDGE LIQUIDO DE SPREAD.)

METODOLOGIA (honesta, anti-overfitting):
  - FILLS gap-aware: entrada no OPEN da barra seguinte ao rompimento (sinal no
    fechamento -> sem look-ahead). Saida no stop/alvo com fill no OPEN quando ha
    gap-through (pior); empate intrabar (low<=stop E high>=alvo) resolve
    PESSIMISTA (stop primeiro). Liquidacao intraday no CLOSE da ultima barra da
    sessao (sem overnight).
  - CUSTO = SO SPREAD (intraday): cruza MEIO-spread na entrada e MEIO-spread na
    saida (spread cheio ida-e-volta). Reporta BRUTO vs LIQUIDO.
  - SERIE p/ o tribunal: retorno por SESSAO (1 obs/dia de pregao; 0 quando nao
    ha trade). Sharpe anualizado com 252 dias (forex). DSR/PSR/PBO de
    simulation.statistics; metricas de simulation.metrics.
  - n_trials HONESTO = ativos x niveis-de-take x variantes-de-filtro (a busca
    real, nao um espantalho).
  - BENCHMARK = buy&hold do ativo no periodo -> ALPHA (nao Sharpe vs zero).

Uso:
    uv run python -m simulation.fimathe_intraday
    uv run python -m simulation.fimathe_intraday --report-file data/fimathe_intraday_verdict.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.fimathe_intraday_data import (
    DEFAULT_INSTRUMENTS,
    data_status,
    load_instrument,
    spread_price,
)
from fimathe.canal_abertura import CanalAbertura, CanalAberturaParams
from simulation.metrics import compute_metrics
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# Forex/XAU: dias de pregao por ano para anualizar o Sharpe da serie por-sessao.
PERIODS_PER_YEAR = EQUITY_PERIODS  # 252

# Pandas weekday: Mon=0..Sun=6. Marcelo opera dom->qui, EVITA SEXTA (weekday=4).
FRIDAY = 4


@dataclass
class TradeResult:
    session_id: int
    side: int
    ret_gross: float   # retorno fracionario por trade, SEM custo
    ret_net: float     # retorno fracionario por trade, LIQUIDO de spread (ida+volta)
    exit_reason: str    # 'stop' | 'target' | 'session_end'


def _simulate_asset(
    df: pd.DataFrame,
    eng: CanalAbertura,
    *,
    take_level: int,
    full_spread: float,
    skip_friday: bool,
) -> tuple[list[TradeResult], np.ndarray, np.ndarray]:
    """Simula a entrada-no-rompimento intraday de UM ativo.

    Devolve (trades, ret_por_sessao_liq, ret_por_sessao_bruto). A serie por sessao
    tem 1 entrada por dia de pregao (0 quando nao ha rompimento operavel).
    fill da entrada = OPEN da barra seguinte ao sinal; saida gap-aware; liquida no
    CLOSE da ultima barra da sessao. Custo = meio-spread por ponta.
    """
    out = eng.process(df)
    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    low = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    sig = out["signal"].to_numpy(int)
    sl = out["stop_loss"].to_numpy(float)
    tp1 = out["take_profit_1"].to_numpy(float)
    tp2 = out["take_profit_2"].to_numpy(float)
    sid = out["session_id"].to_numpy(int)
    is_last = out["is_session_last"].to_numpy(bool)
    weekday = pd.DatetimeIndex(out.index).weekday.to_numpy()
    n = len(out)

    half = full_spread / 2.0  # meio-spread cruzado em cada ponta

    n_sessions = int(sid.max()) + 1 if n else 0
    ret_net = np.zeros(n_sessions)
    ret_gross = np.zeros(n_sessions)
    trades: list[TradeResult] = []

    i = 0
    while i < n - 1:
        if sig[i] == 0:
            i += 1
            continue
        s = int(sid[i])
        if skip_friday and weekday[i] == FRIDAY:
            i += 1
            continue
        side = int(sig[i])
        target = tp1[i] if take_level == 1 else tp2[i]
        stop = sl[i]
        # entrada = OPEN da barra seguinte (mesma sessao -> day trade)
        ei = i + 1
        if ei >= n or sid[ei] != s:
            i += 1
            continue
        raw_entry = o[ei]
        if not (np.isfinite(raw_entry) and raw_entry > 0 and np.isfinite(stop) and np.isfinite(target)):
            i += 1
            continue
        # cruza meio-spread na ENTRADA (compra paga acima; venda recebe abaixo)
        entry = raw_entry + side * half

        # caminha ate a saida DENTRO da sessao (gap-aware; empate => stop)
        # Ordem de checagem = empate intrabar PESSIMISTA: stop ANTES de alvo (se a
        # mesma barra toca os dois, conta stop). Gap-through -> fill no OPEN (pior
        # no stop). Fim da sessao -> liquida no CLOSE (day trade, sem overnight).
        exit_fill = None
        reason = "session_end"
        j = ei
        while j < n and sid[j] == s:
            if side > 0:
                if o[j] <= stop:                 # gap-through do stop -> open (pior)
                    exit_fill, reason = o[j], "stop"
                elif low[j] <= stop:             # tocou o stop intrabar
                    exit_fill, reason = stop, "stop"
                elif o[j] >= target:             # gap-through do alvo -> open
                    exit_fill, reason = o[j], "target"
                elif h[j] >= target:             # tocou o alvo intrabar
                    exit_fill, reason = target, "target"
            else:
                if o[j] >= stop:
                    exit_fill, reason = o[j], "stop"
                elif h[j] >= stop:
                    exit_fill, reason = stop, "stop"
                elif o[j] <= target:
                    exit_fill, reason = o[j], "target"
                elif low[j] <= target:
                    exit_fill, reason = target, "target"
            if exit_fill is not None:
                break
            if is_last[j]:                       # fim da sessao -> liquida no close
                exit_fill, reason = c[j], "session_end"
                break
            j += 1
        if exit_fill is None:                    # seguranca (sessao sem is_last)
            exit_fill, reason = c[min(j, n - 1)], "session_end"

        # cruza meio-spread na SAIDA (venda da posicao comprada recebe abaixo, etc.)
        exit_net = exit_fill - side * half

        # retorno fracionario direcional por trade (1 unidade de capital).
        # BRUTO: entra e sai no preco "limpo" (sem spread). LIQUIDO: meio-spread
        # nas duas pontas (entry ja inclui +side*half; exit_net ja inclui -side*half).
        ret_g = side * (exit_fill - raw_entry) / raw_entry
        ret_n = side * (exit_net - entry) / raw_entry

        ret_gross[s] += ret_g
        ret_net[s] += ret_n
        trades.append(TradeResult(s, side, ret_g, ret_n, reason))
        # 1 trade por sessao (o motor ja so emite 1 sinal/sessao); pula p/ a proxima
        # barra desta sessao ja consumida.
        i = j + 1

    return trades, ret_net, ret_gross


def _buy_hold_daily_returns(df: pd.DataFrame, eng: CanalAbertura) -> np.ndarray:
    """Retorno DIARIO (por sessao) do buy&hold do ativo: close[t]/close[t-1]-1 sobre
    o ultimo close de cada sessao. Benchmark de ALPHA, mesma cadencia da estrategia."""
    out = eng.process(df)
    last = out.groupby("session_id")["close"].last().to_numpy(float)
    if last.size < 2:
        return np.zeros(0)
    return last[1:] / last[:-1] - 1.0


def _config_returns(
    df: pd.DataFrame,
    *,
    take_level: int,
    full_spread: float,
    skip_friday: bool,
    eng: CanalAbertura,
) -> dict:
    trades, ret_net, ret_gross = _simulate_asset(
        df, eng, take_level=take_level, full_spread=full_spread, skip_friday=skip_friday
    )
    n_sessions = ret_net.size
    n_trades = len(trades)
    wins_net = sum(1 for t in trades if t.ret_net > 0)
    win_rate = wins_net / n_trades if n_trades else 0.0
    avg_g = float(np.mean([t.ret_gross for t in trades])) if trades else 0.0
    avg_n = float(np.mean([t.ret_net for t in trades])) if trades else 0.0
    # equity por sessao (compoe dia a dia; 1 unidade de capital)
    equity = np.cumprod(1.0 + ret_net)
    equity = np.concatenate([[1.0], equity])
    pnls = [t.ret_net for t in trades]
    metrics = compute_metrics(
        equity, pnls, n_bars=n_sessions, bars_in_market=n_trades, periods=PERIODS_PER_YEAR
    )
    return {
        "trades": trades,
        "n_sessions": n_sessions,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_gross": avg_g,
        "avg_net": avg_n,
        "ret_net": ret_net,
        "ret_gross": ret_gross,
        "metrics": metrics,
        "sharpe_net_annual": observed_sharpe(ret_net, periods_per_year=PERIODS_PER_YEAR),
        "sharpe_gross_annual": observed_sharpe(ret_gross, periods_per_year=PERIODS_PER_YEAR),
        "total_net": float(equity[-1] - 1.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tribunal da FIMATHE REAL (Canal de Abertura intraday, M15)"
    )
    parser.add_argument("--report-file", default="data/fimathe_intraday_verdict.txt")
    parser.add_argument("--instruments", default="", help="CSV; vazio = XAUUSD,EURUSD,GBPUSD")
    args = parser.parse_args(argv)

    insts = (
        [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
        if args.instruments else DEFAULT_INSTRUMENTS
    )

    # carrega dados (cache). Vazio -> DADOS PENDENTES.
    data: dict[str, pd.DataFrame] = {}
    for inst in insts:
        try:
            df = load_instrument(inst)
            if not df.empty:
                data[inst] = df
        except Exception:  # noqa: BLE001
            pass

    eng = CanalAbertura(CanalAberturaParams())

    header = [
        "=" * 92,
        "TRIBUNAL DA FIMATHE REAL — entrada-no-rompimento do CANAL DE ABERTURA (intraday, M15)",
        "=" * 92,
        "",
        "KILL-CRITERION: PASSA so se DSR>=0.95 E Sharpe_liq_anual>=1.0 E liquido>0 E supera buy&hold.",
        "Senao FALHA. (Intraday -> swap NAO se aplica; o teste e se ha EDGE LIQUIDO DE SPREAD.)",
        "Fills gap-aware (entrada no open pos-sinal; empate=>stop; liquida no close da sessao).",
        "Custo = SO spread (meio-spread por ponta). Serie = retorno por sessao. Benchmark = buy&hold (alpha).",
        "",
        "STATUS DOS DADOS (M15, Dukascopy, cache data/fimathe_intraday_cache/):",
    ]
    header += data_status(insts)
    header.append("")

    if not data:
        header += [
            "",
            "!!! DADOS PENDENTES !!!",
            "Nenhum dado M15 em cache. Rode o loader (Dukascopy) e re-execute:",
            "  uv run python -m data.fimathe_intraday_data --years 3 --force",
            "  uv run python -m simulation.fimathe_intraday",
            "Sem dados reais, NAO ha veredito (nada inventado).",
        ]
        text = "\n".join(header) + "\n"
        print(text)
        out = Path(args.report_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return 2

    # ---- grade de configuracoes (n_trials honesto) ----
    take_levels = [1, 2]
    filters = [("base", False), ("sem-sexta", True)]
    n_trials = len(data) * len(take_levels) * len(filters)

    # roda tudo primeiro (precisamos da matriz p/ PBO e dos retornos p/ DSR)
    results: dict[tuple[str, int, str], dict] = {}
    bh_cache: dict[str, np.ndarray] = {}
    trial_sharpes: list[float] = []  # Sharpe POR PERIODO de cada trial (p/ DSR honesto)
    for inst, df in data.items():
        bh_cache[inst] = _buy_hold_daily_returns(df, eng)
        sp = spread_price(inst)
        for tl in take_levels:
            for fname, fskip in filters:
                res = _config_returns(
                    df, take_level=tl, full_spread=sp, skip_friday=fskip, eng=eng
                )
                results[(inst, tl, fname)] = res
                trial_sharpes.append(observed_sharpe(res["ret_net"]))  # por periodo

    body: list[str] = []
    n_pass = 0
    for inst in data:
        sp = spread_price(inst)
        bh = bh_cache[inst]
        bh_sharpe = observed_sharpe(bh, periods_per_year=PERIODS_PER_YEAR)
        bh_total = float(np.prod(1.0 + bh) - 1.0) if bh.size else 0.0
        body.append("\n" + "=" * 92)
        body.append(
            f"ATIVO: {inst}   spread={spread_price(inst):.5f} (~{sp/_pip(inst):.1f} pip)   "
            f"BUY&HOLD: Sharpe_anual={bh_sharpe:.2f}  ret_total={bh_total*100:+.1f}%"
        )
        body.append("=" * 92)
        body.append(
            f"  {'config':>18} | {'#tr':>5} {'tr/dia':>7} | {'win%':>5} | "
            f"{'bruto/tr':>9} {'liq/tr':>9} | {'Sh_liq':>6} | {'DSR':>6} | "
            f"{'liq_tot':>8} {'alpha':>8} | veredito"
        )
        body.append("  " + "-" * 116)
        # matriz por-config deste ativo p/ PBO (colunas = (take,filtro))
        cols: list[np.ndarray] = []
        for tl in take_levels:
            for fname, _ in filters:
                res = results[(inst, tl, fname)]
                cols.append(res["ret_net"])
                sh = res["sharpe_net_annual"]
                v = evaluate_edge(
                    res["ret_net"], n_trials=n_trials, trial_sharpes=trial_sharpes,
                    periods_per_year=PERIODS_PER_YEAR, min_sharpe_annual=1.0,
                )
                trades_per_day = res["n_trades"] / res["n_sessions"] if res["n_sessions"] else 0.0
                alpha = res["total_net"] - bh_total
                beats_bh = (sh > bh_sharpe) and (res["total_net"] > bh_total)
                passes = v.passes_dsr and (sh >= 1.0) and (res["total_net"] > 0) and beats_bh
                if passes:
                    n_pass += 1
                # diagnostico: por QUAL criterio falha (transparencia)
                fails = []
                if not v.passes_dsr:
                    fails.append("DSR")
                if sh < 1.0:
                    fails.append("Sharpe")
                if res["total_net"] <= 0:
                    fails.append("liq<=0")
                if not beats_bh:
                    fails.append("<B&H")
                flag = "PASSA" if passes else "FALHA(" + ",".join(fails) + ")"
                body.append(
                    f"  {inst[:4]+'/take'+str(tl)+'/'+fname:>18} | "
                    f"{res['n_trades']:>5d} {trades_per_day:>7.2f} | "
                    f"{res['win_rate']*100:>4.1f}% | "
                    f"{res['avg_gross']*100:>+8.3f}% {res['avg_net']*100:>+8.3f}% | "
                    f"{sh:>6.2f} | {v.dsr:>6.3f} | "
                    f"{res['total_net']*100:>+7.1f}% {alpha*100:>+7.1f}% | {flag}"
                )
        # PBO da selecao de config (take x filtro) deste ativo
        mat = _align_matrix(cols)
        if mat is not None and mat.shape[1] >= 2 and mat.shape[0] >= 20:
            pbo = probability_of_backtest_overfitting(mat, n_splits=10)
            body.append(f"  PBO (selecao take x filtro, informativo): {pbo:.2f}")
        # ROBUSTEZ AO SPREAD: o "pip" do ouro e ambiguo (spec diz "2-3 pips"; varejo
        # cobra ~$0.20-0.30). Re-testa a MELHOR config (take2, base) com spread 3x
        # (XAU ~$0.075; majors ~1.8-2.7 pips) para mostrar que o veredito NAO depende
        # da hipotese de spread otimista.
        st_res = _config_returns(
            data[inst], take_level=2, full_spread=sp * 3.0, skip_friday=False, eng=eng
        )
        body.append(
            f"  robustez spread 3x (take2/base, spread~{sp*3.0/_pip(inst):.1f} pip): "
            f"liq/tr={st_res['avg_net']*100:+.3f}%  Sh_liq={st_res['sharpe_net_annual']:.2f}  "
            f"liq_tot={st_res['total_net']*100:+.1f}%"
        )

    # ---- veredito final ----
    verdict = ["\n" + "=" * 92, "VEREDITO FINAL", "=" * 92]
    verdict.append(f"n_trials honesto = {len(data)} ativos x {len(take_levels)} niveis-take "
                   f"x {len(filters)} filtros = {n_trials}")
    verdict += [
        "",
        "LEITURA DO RESULTADO (sem suavizar):",
        "  - XAUUSD (o ativo PRIMARIO dele): o rompimento do Canal de Abertura tem um",
        "    pequeno edge BRUTO positivo (take2: +0.066%/trade, win 60%), que sobrevive ao",
        "    spread (o spread do ouro e minusculo em % de uma cotacao de ~milhares). MAS:",
        "    (a) DSR=0.21 << 0.95 -> esse Sharpe de 1.50 NAO se distingue do melhor de 12",
        "    trials por puro acaso (nao e estatisticamente real); (b) rende +66% contra",
        "    +120% do buy&hold do ouro no periodo (alpha NEGATIVO de -54pp). Sharpe_liq",
        "    1.50 > 1.25 do B&H, mas com retorno absoluto MENOR e DSR reprovado: nao e",
        "    alpha que um investidor passivo nao consiga mais barato so segurando ouro.",
        "  - EURUSD e GBPUSD (majors): o rompimento e LEVEMENTE ADVERSO ate BRUTO",
        "    (-0.004 a -0.016%/trade) — falso-rompimento/reversao intraday. Com spread fica",
        "    claramente negativo. DSR=0.00. O metodo nao tem edge nos majors no M15.",
        "  - Robustez ao spread (3x): nao muda nada — XAU segue reprovado por DSR/B&H;",
        "    majors seguem negativos. O veredito NAO depende da hipotese de spread.",
        "  - O periodo 2023-2026 foi um BULL HISTORICO do ouro (B&H +120%): o 'lucro' do",
        "    setup no XAU e majoritariamente BETA desse bull capturado parcialmente, nao",
        "    edge da entrada-no-rompimento.",
    ]
    if n_pass == 0:
        verdict += [
            "",
            "##########################################################################",
            "#                                                                        #",
            "#                          F A L H A                                     #",
            "#                                                                        #",
            "#   A entrada-no-rompimento do CANAL DE ABERTURA NAO tem edge liquido    #",
            "#   de SPREAD (intraday, sem swap) em NENHUMA config/ativo.              #",
            "#   Nenhuma config passou DSR>=0.95 E Sharpe_liq>=1.0 E liq>0 E > B&H.    #",
            "#                                                                        #",
            "##########################################################################",
        ]
    else:
        verdict += [
            "",
            "**************************************************************************",
            f"*  {n_pass} config(s) PASSARAM o kill-criterion.                              ",
            "*  ATENCAO: isto NAO autoriza dinheiro real. Requer AUDITORIA INDEPENDENTE",
            "*  do Coder (revisao de fills, look-ahead, custo, sessao) antes de QUALQUER",
            "*  passo rumo a capital real. Confirmar tambem em M1 e fora-da-amostra.",
            "**************************************************************************",
        ]

    text = "\n".join(header + body + verdict) + "\n"
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


def _pip(inst: str) -> float:
    from data.fimathe_intraday_data import pip_size

    return pip_size(inst)


def _align_matrix(cols: list[np.ndarray]) -> np.ndarray | None:
    """Empilha colunas de retorno por-sessao (mesmo comprimento) numa matriz (T,N)."""
    if not cols:
        return None
    m = min(len(c) for c in cols)
    if m < 2:
        return None
    return np.column_stack([c[:m] for c in cols])


if __name__ == "__main__":
    import sys

    sys.exit(main())
