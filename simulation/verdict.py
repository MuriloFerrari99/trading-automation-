"""Veredito honesto: as estrategias ATUAIS tem edge ou e ruido?

Roda cada estrategia (trailing_stop, ladder_buys) sobre o historico completo de cada
simbolo (cache em data/cache), monta uma carteira equal-weight diaria, aplica CUSTOS REAIS
(cripto 70bps round-trip; equities ~spread) e submete a serie ao tribunal estatistico:
Sharpe anualizado, PSR, Deflated Sharpe (a 1/50/200 trials) e PBO entre simbolos.

Uso:
    uv run python -m simulation.verdict
    uv run python -m simulation.verdict --report-file data/edge_verdict.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import CRYPTO_BASE, EQUITY_BASE, CostModel
from simulation.engine import run_backtest
from simulation.statistics import (
    CRYPTO_PERIODS,
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE = Path("data/cache")
WARMUP = 35
STRATEGIES = ("trailing_stop", "ladder_buys")

CRYPTO = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "DOGE/USD", "AVAX/USD"]
EQUITIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "JPM",
    "V", "MA", "XOM", "KO", "PG", "JNJ", "UNH", "HD", "WMT", "DIS",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "GLD",
]


def _load(symbol: str) -> pd.DataFrame | None:
    p = CACHE / f"{symbol.replace('/', '_')}_1d.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df[~df.index.isna()]


def _strategy_returns(symbol: str, df: pd.DataFrame, strat: str, cost: CostModel) -> pd.Series | None:
    """Retornos diarios da estrategia no simbolo, ja liquidos de custo. Indexados por data."""
    ohlc = {
        "open": df["open"].astype(float).tolist(),
        "high": df["high"].astype(float).tolist(),
        "low": df["low"].astype(float).tolist(),
        "close": df["close"].astype(float).tolist(),
    }
    r = run_backtest(
        symbol, ohlc, strat,
        commission_bps=cost.commission_bps, slippage_bps=cost.slippage_bps,
        warmup=WARMUP, use_risk=False,
    )
    if r is None or len(r.equity) < WARMUP + 5:
        return None
    eq = np.asarray(r.equity, dtype=float)
    prev = np.where(eq[:-1] == 0, np.nan, eq[:-1])
    rets = np.diff(eq) / prev
    idx = df.index[1 : len(eq)]
    s = pd.Series(rets, index=idx).iloc[WARMUP:]  # so depois do warmup (estrategia ativa)
    return s.dropna()


def _portfolio(symbols: list[str], strat: str, cost: CostModel) -> tuple[pd.Series, pd.DataFrame]:
    cols: list[pd.Series] = []
    for sym in symbols:
        df = _load(sym)
        if df is None or len(df) < WARMUP + 50:
            continue
        s = _strategy_returns(sym, df, strat, cost)
        if s is not None and s.size > 0:
            cols.append(s.rename(sym))
    if not cols:
        return pd.Series(dtype=float), pd.DataFrame()
    mat = pd.concat(cols, axis=1).sort_index()
    port = mat.mean(axis=1, skipna=True).dropna()  # equal-weight diario
    return port, mat


def _verdict_block(label: str, port: pd.Series, mat: pd.DataFrame, periods: int) -> str:
    if port.size < 30:
        return f"\n{label}: amostra insuficiente ({port.size} obs)\n"
    ret = port.to_numpy()
    sharpe_ann = observed_sharpe(ret, periods_per_year=periods)

    lines = [f"\n{'=' * 92}", f"{label}", "=" * 92]
    lines.append(
        f"  obs={port.size}  Sharpe_anual(liq)={sharpe_ann:6.2f}  "
        f"retorno_medio_diario={ret.mean() * 100:6.3f}%  vol_diaria={ret.std() * 100:5.2f}%"
    )
    lines.append(f"  {'-' * 88}")
    lines.append(f"  {'n_trials':>10} | {'obstaculo_SR':>12} | {'PSR(>0)':>9} | {'DSR':>7} | veredito")
    for nt in (1, 50, 200):
        v = evaluate_edge(ret, n_trials=nt, periods_per_year=periods, min_sharpe_annual=0.8)
        flag = "PASSA" if v.passes else "FALHA"
        lines.append(
            f"  {nt:>10} | {v.sr_benchmark_annual:>12.2f} | {v.psr:>9.3f} | {v.dsr:>7.3f} | {flag}"
        )

    # PBO: a melhor escolha de simbolo in-sample se sustenta out-of-sample?
    clean = mat.fillna(0.0)
    if clean.shape[1] >= 2 and clean.shape[0] >= 20:
        pbo = probability_of_backtest_overfitting(clean.to_numpy(), n_splits=10)
        lines.append(f"  PBO (overfitting de selecao de simbolo): {pbo:.2f}  "
                     f"({'OK <0.5' if pbo < 0.5 else 'ALTO >=0.5'})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Veredito do tribunal sobre as estrategias atuais")
    parser.add_argument("--report-file", default="data/edge_verdict.txt")
    args = parser.parse_args(argv)

    blocks = [
        "TRIBUNAL ESTATISTICO — as estrategias atuais tem edge ou e ruido?",
        "Custos reais (cripto 70bps round-trip / equities spread). Serie = carteira equal-weight diaria.",
        "Barra de aceitacao: DSR>=0.95 (p<0.05) E Sharpe_anual_liq>=0.8.",
    ]

    for strat in STRATEGIES:
        cp, cm = _portfolio(CRYPTO, strat, CRYPTO_BASE)
        blocks.append(_verdict_block(f"CRIPTO · {strat} · custo base (70bps)", cp, cm, CRYPTO_PERIODS))
        cps, cms = _portfolio(CRYPTO, strat, CRYPTO_BASE.stressed(2.0))
        blocks.append(_verdict_block(f"CRIPTO · {strat} · custo ESTRESSADO 2x (140bps)", cps, cms, CRYPTO_PERIODS))
        ep, em = _portfolio(EQUITIES, strat, EQUITY_BASE)
        blocks.append(_verdict_block(f"EQUITIES · {strat} · custo base", ep, em, EQUITY_PERIODS))

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
