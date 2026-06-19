"""Tribunal do BREAKOUT (FimatheEngine) — o teste que faltava.

O `simulation/verdict.py` julga trailing_stop e ladder_buys (que sao, na
pratica, buy-and-hold+stop e dip-buyer) — NUNCA o setup de breakout da
FimatheEngine, que e o unico com edge por-trade demonstrado (PF~1.8). Este
modulo corrige isso: monta a serie de retornos DIARIA de uma carteira de
breakouts e a submete ao MESMO tribunal (Sharpe, PSR, Deflated Sharpe), com:

  - FILLS REALISTAS (gap-aware): a saida no stop/alvo usa o OPEN real da barra
    quando o preco abre alem do nivel (gap-through) — sem assumir fill exato.
    Empate intrabar (low<=stop E high>=alvo) resolve PESSIMISTA (stop primeiro).
  - SIZING comparavel: equal-dolar (w=1) vs equal-risco (w=1/risco%). Sharpe e
    invariante a escala, entao a diferenca isola o EFEITO DA PESAGEM (stops de
    14.8% em acoes vs 35% em cripto fazem equal-dolar superexpor cripto).
  - GATE de regime opcional (so opera em trend_up — onde o bootstrap mostra
    que mora o edge).
  - CUSTOS reais (simulation.costs) deduzidos na entrada e na saida.
  - n_trials HONESTO no Deflated Sharpe (a busca real, nao o espantalho 200).

Uso:
    uv run python -m simulation.breakout_tribunal
    uv run python -m simulation.breakout_tribunal --report-file data/breakout_verdict.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feedback.regime import classify_regime
from fimathe.engine import FimatheEngine
from simulation.costs import CRYPTO_BASE, EQUITY_BASE, CostModel
from simulation.statistics import (
    CRYPTO_PERIODS,
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE = Path("data/cache")
WARMUP = 35
MAX_HOLD = 20
REGIME_LOOKBACK = 60

CRYPTO = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "DOGE/USD", "AVAX/USD"]
EQUITIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "JPM",
    "V", "MA", "XOM", "KO", "PG", "JNJ", "UNH", "HD", "WMT", "DIS",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "GLD",
]


@dataclass
class Trade:
    entry_day: int          # indice da barra de entrada (open da i+1)
    exit_day: int           # indice da barra de saida
    risk_frac: float        # (entry - stop)/entry  -> 1R em fracao de preco
    regime: str
    daily_rets: np.ndarray  # retorno do ATIVO em cada dia mantido (entry..exit), ja com fills nas pontas


def _load(symbol: str) -> pd.DataFrame | None:
    p = CACHE / f"{symbol.replace('/', '_')}_1d.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df[~df.index.isna()]


def generate_trades(df: pd.DataFrame, engine: FimatheEngine, cost: CostModel) -> list[Trade]:
    """Setups LONG da FimatheEngine com saida gap-aware. Retornos diarios do
    ativo durante o hold, ja liquidos de custo nas pontas."""
    if len(df) < engine.swing_period + 40:
        return []
    out = engine.process(df)
    o = out["open"].to_numpy(float); h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float); c = out["close"].to_numpy(float)
    sig = out["signal"].to_numpy(float); val = out["setup_valid"].to_numpy(bool)
    sl = out["stop_loss"].to_numpy(float); tp = out["take_profit_2"].to_numpy(float)
    n = len(c)
    cf = cost.commission_bps / 10000.0 + cost.slippage_bps / 10000.0  # custo por lado (fracao)
    trades: list[Trade] = []

    i = engine.swing_period
    while i < n - 1:
        if sig[i] != 1 or not val[i]:
            i += 1
            continue
        entry = o[i + 1]
        stop = sl[i]; target = tp[i]
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(stop) and 0 < stop < entry):
            i += 1
            continue
        risk_frac = (entry - stop) / entry
        # ---- caminha ate a saida (gap-aware, empate => stop) ----
        end = min(i + 1 + MAX_HOLD, n - 1)
        exit_day = end
        exit_fill = c[end]
        for j in range(i + 1, end + 1):
            if o[j] <= stop:                     # gap-through do stop: fill no open (pior)
                exit_day = j; exit_fill = o[j]; break
            if l[j] <= stop:                     # tocou o stop intrabar
                exit_day = j; exit_fill = stop; break
            if o[j] >= target:                   # gap-through do alvo: fill no open
                exit_day = j; exit_fill = o[j]; break
            if h[j] >= target:                   # tocou o alvo intrabar
                exit_day = j; exit_fill = target; break
        # ---- retornos diarios do ativo durante o hold, com fills nas pontas ----
        rets: list[float] = []
        prev = entry  # entrada efetiva
        for j in range(i + 1, exit_day + 1):
            px = exit_fill if j == exit_day else c[j]
            rets.append(px / prev - 1.0)
            prev = c[j]
        # custo de ida (entrada) e volta (saida) como hit de retorno nas pontas
        if rets:
            rets[0] -= cf
            rets[-1] -= cf
        regime = classify_regime(c[max(0, i - REGIME_LOOKBACK): i + 1].tolist()).value
        trades.append(Trade(i + 1, exit_day, risk_frac, regime, np.asarray(rets, float)))
        i = exit_day + 1  # 1 posicao por simbolo por vez (sem empilhar no mesmo ativo)
    return trades


def portfolio_series(
    symbols: list[str], engine: FimatheEngine, cost: CostModel,
    *, weighting: str = "risk", gate_trend_up: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Serie de retorno DIARIA da carteira de breakouts + matriz por-simbolo (p/ PBO).

    weighting: 'dollar' (w=1) ou 'risk' (w=1/risco%). Sharpe e invariante a escala,
    entao isto isola o efeito da PESAGEM relativa entre posicoes."""
    # Acumula, por dia: numerador sum(w*r) e denominador sum(w). A carteira diaria
    # e a MEDIA PONDERADA das posicoes ATIVAS (sempre 100% investida no que esta
    # ativo) -> sem o artefato de "alavancagem por contagem" de uma soma crua.
    per_symbol_wr: list[pd.Series] = []   # w*r por simbolo (p/ PBO)
    num: dict | None = None
    den: dict | None = None
    union_num = None
    union_den = None
    for sym in symbols:
        df = _load(sym)
        if df is None or len(df) < WARMUP + 50:
            continue
        idx = df.index
        wr = pd.Series(0.0, index=idx)    # w*r
        w_act = pd.Series(0.0, index=idx)  # w nos dias ativos
        for tr in generate_trades(df, engine, cost):
            if gate_trend_up and tr.regime != "trend_up":
                continue
            w = 1.0 if weighting == "dollar" else (1.0 / tr.risk_frac if tr.risk_frac > 0 else 0.0)
            days = idx[tr.entry_day: tr.exit_day + 1]
            if len(days) != len(tr.daily_rets):
                continue
            wr.loc[days] += w * tr.daily_rets
            w_act.loc[days] += w
        if w_act.sum() > 0:
            per_symbol_wr.append(wr.rename(sym))
            union_num = wr if union_num is None else union_num.add(wr, fill_value=0.0)
            union_den = w_act if union_den is None else union_den.add(w_act, fill_value=0.0)
    if union_num is None:
        return pd.Series(dtype=float), pd.DataFrame()
    union_num = union_num.sort_index(); union_den = union_den.sort_index()
    port = (union_num / union_den.replace(0.0, np.nan)).fillna(0.0)  # media ponderada das ativas
    mat = pd.concat(per_symbol_wr, axis=1).sort_index().fillna(0.0)  # p/ PBO informativo
    return port[port.index.notna()], mat


def verdict_block(label: str, port: pd.Series, mat: pd.DataFrame, periods: int, n_trials: int) -> str:
    if port.size < 30 or port.std() == 0:
        return f"\n{label}: amostra insuficiente ({port.size} obs)\n"
    ret = port.to_numpy()
    active = float((ret != 0).mean())
    sharpe_ann = observed_sharpe(ret, periods_per_year=periods)
    lines = [f"\n{'=' * 92}", label, "=" * 92]
    lines.append(
        f"  obs={port.size} dias_ativos={active*100:4.1f}%  Sharpe_anual(liq)={sharpe_ann:6.2f}  "
        f"ret_diario_medio={ret.mean()*100:6.3f}%  vol_diaria={ret.std()*100:5.2f}%"
    )
    lines.append(f"  {'-' * 88}")
    lines.append(f"  {'n_trials':>9} | {'obstaculo_SR':>12} | {'PSR(>0)':>8} | {'DSR':>7} | veredito")
    for nt in sorted({1, n_trials}):
        v = evaluate_edge(ret, n_trials=nt, periods_per_year=periods, min_sharpe_annual=0.8)
        flag = "PASSA" if v.passes else "FALHA"
        lines.append(f"  {nt:>9} | {v.sr_benchmark_annual:>12.2f} | {v.psr:>8.3f} | {v.dsr:>7.3f} | {flag}")
    if mat.shape[1] >= 2 and mat.shape[0] >= 20:
        pbo = probability_of_backtest_overfitting(mat.to_numpy(), n_splits=10)
        lines.append(f"  PBO (selecao de simbolo, informativo): {pbo:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tribunal do breakout FimatheEngine (portfolio, gap-aware)")
    parser.add_argument("--report-file", default="data/breakout_verdict.txt")
    parser.add_argument("--n-trials", type=int, default=12, help="n_trials HONESTO no Deflated Sharpe (busca real)")
    args = parser.parse_args(argv)

    engine = FimatheEngine()
    blocks = [
        "TRIBUNAL DO BREAKOUT (FimatheEngine) — a estrategia que o veredito original nunca testou.",
        "Fills gap-aware (open real em gap-through; empate=>stop). Custos reais. Carteira diaria.",
        f"Barra: DSR>=0.95 E Sharpe_anual_liq>=0.8. n_trials honesto={args.n_trials}.",
    ]
    matrix = [
        ("CRIPTO", CRYPTO, CRYPTO_BASE, CRYPTO_PERIODS),
        ("EQUITIES", EQUITIES, EQUITY_BASE, EQUITY_PERIODS),
    ]
    for aclass, syms, cost, periods in matrix:
        for weighting in ("dollar", "risk"):
            for gate in (False, True):
                gtag = "gate:trend_up" if gate else "gate:off"
                port, mat = portfolio_series(syms, engine, cost, weighting=weighting, gate_trend_up=gate)
                label = f"{aclass} · breakout · peso:{weighting} · {gtag} · custo base"
                blocks.append(verdict_block(label, port, mat, periods, args.n_trials))

    text = "\n".join(blocks) + "\n"
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
