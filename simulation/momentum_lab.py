"""Primeiro candidato a edge REAL: time-series momentum com volatility-management.

Hipotese (literatura): cripto TENDE e momentum de serie temporal e robusto OOS; gerenciar
exposicao pela volatilidade recente reduz os crashes do momentum e eleva o Sharpe
(Springer 2025). Adaptado a Alpaca cripto: SPOT LONG-ONLY, sem alavancagem (cap=1.0).

Mecanica (sem look-ahead):
  - sinal no fechamento t: long se retorno acumulado dos ultimos `lookback` dias > 0, senao FLAT.
  - peso alvo: w_t = sinal * min(1, target_vol / vol_realizada_anualizada)  (vol-targeting).
  - posicao w_t e mantida de t -> t+1; ganha w_t * r_{t+1}.
  - custo de transacao incide sobre o TURNOVER |w_t - w_{t-1}| (onde o momentum costuma morrer).

Honestidade anti-snooping: testamos uma GRADE de parametros e reportamos o DSR do melhor
config penalizado por n_trials = tamanho da grade. Um edge so "passa" se vencer esse obstaculo.

Uso:
    uv run python -m simulation.momentum_lab
    uv run python -m simulation.momentum_lab --report-file data/momentum_verdict.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import CRYPTO_BASE, CostModel
from simulation.statistics import (
    CRYPTO_PERIODS,
    evaluate_edge,
    observed_sharpe,
)

CACHE = Path("data/cache")
CRYPTO = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "DOGE/USD", "AVAX/USD"]

# Grade de parametros (define n_trials para o DSR).
LOOKBACKS = (20, 30, 60, 90, 120)
VOL_WINDOWS = (20, 30)
TARGET_VOLS = (0.40, 0.60)  # vol anual alvo; None = momentum binario (sem vol-mgmt)


@dataclass(frozen=True)
class Params:
    lookback: int
    vol_window: int
    target_vol: float | None  # None = binario (peso 0/1)
    rebalance_days: int = 1  # 1=diario; 7=semanal (sample-and-hold entre rebalances)

    def label(self) -> str:
        tv = "bin" if self.target_vol is None else f"{self.target_vol:.2f}"
        rb = "" if self.rebalance_days == 1 else f"/rb{self.rebalance_days}"
        return f"L{self.lookback}/V{self.vol_window}/tv{tv}{rb}"


def _load_close(symbol: str) -> pd.Series | None:
    p = CACHE / f"{symbol.replace('/', '_')}_1d.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()]
    return df["close"].astype(float)


def momentum_weights(close: np.ndarray, p: Params, *, cap: float = 1.0,
                     periods: int = CRYPTO_PERIODS) -> np.ndarray:
    """w[t] = peso a manter de t->t+1, usando SO informacao ate o fechamento t (sem leak)."""
    n = close.size
    w = np.zeros(n)
    rets = np.empty(n)
    rets[0] = 0.0
    rets[1:] = close[1:] / close[:-1] - 1.0
    ann = np.sqrt(periods)
    start = max(p.lookback, p.vol_window) + 1
    last_w = 0.0
    for t in range(start, n):
        # sample-and-hold: so recalcula o alvo em dias de rebalance; senao carrega o peso.
        if (t - start) % p.rebalance_days == 0:
            mom = close[t] / close[t - p.lookback] - 1.0  # retorno acumulado ate t
            if mom <= 0:
                last_w = 0.0  # long-only: fora do mercado
            elif p.target_vol is None:
                last_w = 1.0  # momentum binario
            else:
                sigma = rets[t - p.vol_window + 1 : t + 1].std(ddof=1) * ann
                last_w = min(cap, p.target_vol / sigma) if sigma > 0 else 0.0
        w[t] = last_w
    return w


def _asset_net_returns(close: pd.Series, p: Params, cost: CostModel) -> pd.Series:
    """Retornos diarios liquidos do ativo: w[t]*r[t+1] - custo(turnover em t). Indexado em t+1."""
    c = close.to_numpy()
    n = c.size
    w = momentum_weights(c, p)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = c[1:] / c[:-1] - 1.0
    per_side = cost.per_side_bps / 10000.0
    turnover = np.abs(np.diff(np.concatenate([[0.0], w])))  # |w[t]-w[t-1]|
    cost_t = turnover * per_side
    # retorno realizado em t+1 = w[t]*r[t+1]; custo pago em t reduz o retorno de t+1
    net = np.full(n, np.nan)
    net[1:] = w[:-1] * r[1:] - cost_t[:-1]
    return pd.Series(net, index=close.index).dropna()


def portfolio(symbols: list[str], p: Params, cost: CostModel) -> pd.Series:
    cols = []
    for s in symbols:
        c = _load_close(s)
        if c is None or c.size < 200:
            continue
        cols.append(_asset_net_returns(c, p, cost).rename(s))
    if not cols:
        return pd.Series(dtype=float)
    mat = pd.concat(cols, axis=1).sort_index()
    return mat.mean(axis=1, skipna=True).dropna()  # equal-weight de capital (sem alavancagem)


def _asset_gross_net(close: pd.Series, p: Params, cost: CostModel) -> tuple[pd.Series, pd.Series, float]:
    """Retornos BRUTOS e LIQUIDOS do ativo + turnover anualizado (p/ separar sinal de custo)."""
    c = close.to_numpy()
    n = c.size
    w = momentum_weights(c, p)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = c[1:] / c[:-1] - 1.0
    per_side = cost.per_side_bps / 10000.0
    turnover = np.abs(np.diff(np.concatenate([[0.0], w])))
    gross = np.full(n, np.nan)
    net = np.full(n, np.nan)
    gross[1:] = w[:-1] * r[1:]
    net[1:] = w[:-1] * r[1:] - turnover[:-1] * per_side
    years = max(n / CRYPTO_PERIODS, 1e-9)
    ann_turnover = float(turnover.sum() / years)  # ~nº de trocas de posicao por ano
    gs = pd.Series(gross, index=close.index).dropna()
    ns = pd.Series(net, index=close.index).dropna()
    return gs, ns, ann_turnover


def diagnose(symbols: list[str], cost: CostModel) -> str:
    """Diagnostico: o custo/turnover diario matou o sinal, ou o sinal e fraco?"""
    signals = [
        Params(30, 20, None),   # vencedor da grade (binario)
        Params(30, 20, 0.40),   # vol-managed (a tese)
        Params(90, 30, 0.40),   # lookback mais longo
    ]
    freqs = (1, 3, 7, 14, 30)
    lines = [
        "=" * 100,
        "DIAGNOSTICO — Sharpe BRUTO vs LIQUIDO por frequencia de rebalance",
        "(bruto alto + liquido sobe com rebalance mais longo => o CUSTO matou; ambos baixos => SINAL fraco)",
        "=" * 100,
    ]
    bh = buy_and_hold(symbols, cost)
    lines.append(f"Referencia buy&hold: Sharpe_anual_liq={observed_sharpe(bh.to_numpy(), periods_per_year=CRYPTO_PERIODS):.2f}")
    for base in signals:
        lines.append("")
        lines.append(f"Sinal {base.label()}:")
        lines.append(f"  {'rebalance':>10} | {'Sharpe_BRUTO':>12} | {'Sharpe_LIQ':>10} | {'turnover/ano':>12}")
        for f in freqs:
            p = Params(base.lookback, base.vol_window, base.target_vol, rebalance_days=f)
            gcols, ncols, turns = [], [], []
            for s in symbols:
                c = _load_close(s)
                if c is None or c.size < 200:
                    continue
                g, nn, tn = _asset_gross_net(c, p, cost)
                gcols.append(g.rename(s))
                ncols.append(nn.rename(s))
                turns.append(tn)
            if not gcols:
                continue
            gp = pd.concat(gcols, axis=1).mean(axis=1, skipna=True).dropna()
            npf = pd.concat(ncols, axis=1).mean(axis=1, skipna=True).dropna()
            g_sr = observed_sharpe(gp.to_numpy(), periods_per_year=CRYPTO_PERIODS)
            n_sr = observed_sharpe(npf.to_numpy(), periods_per_year=CRYPTO_PERIODS)
            lines.append(f"  {f:>9}d | {g_sr:>12.2f} | {n_sr:>10.2f} | {np.mean(turns):>12.1f}")
    return "\n".join(lines)


def buy_and_hold(symbols: list[str], cost: CostModel) -> pd.Series:
    cols = []
    for s in symbols:
        c = _load_close(s)
        if c is None or c.size < 200:
            continue
        r = c.pct_change().dropna()
        r.iloc[0] -= cost.per_side_bps / 10000.0  # custo de entrada unico
        cols.append(r.rename(s))
    mat = pd.concat(cols, axis=1).sort_index()
    return mat.mean(axis=1, skipna=True).dropna()


def run_grid(symbols: list[str], cost: CostModel) -> list[tuple[Params, float, pd.Series]]:
    grid = [Params(lb, vw, tv) for lb, vw, tv in product(LOOKBACKS, VOL_WINDOWS, TARGET_VOLS)]
    grid += [Params(lb, VOL_WINDOWS[0], None) for lb in LOOKBACKS]  # variantes binarias
    out = []
    for p in grid:
        port = portfolio(symbols, p, cost)
        if port.size < 60:
            continue
        sr = observed_sharpe(port.to_numpy(), periods_per_year=CRYPTO_PERIODS)
        out.append((p, sr, port))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lab de momentum cripto vs tribunal")
    parser.add_argument("--report-file", default="data/momentum_verdict.txt")
    parser.add_argument("--diagnose", action="store_true", help="diagnostico bruto vs liquido por rebalance")
    args = parser.parse_args(argv)

    if args.diagnose:
        text = diagnose(CRYPTO, CRYPTO_BASE) + "\n"
        print(text)
        out = Path("data/momentum_diagnose.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Relatorio salvo em {out}")
        return 0

    results = run_grid(CRYPTO, CRYPTO_BASE)
    n_trials = len(results)
    lines = [
        "=" * 92,
        "EDGE #1 — TIME-SERIES MOMENTUM + VOL-MANAGEMENT (cripto, long-only, custo real)",
        "=" * 92,
        f"Grade testada: {n_trials} configs (= n_trials do DSR).  periods/ano=365.",
        "",
        "Top 10 configs por Sharpe liquido:",
        f"  {'config':<18} {'Sharpe_anual_liq':>16}",
    ]
    for p, sr, _ in results[:10]:
        lines.append(f"  {p.label():<18} {sr:>16.2f}")

    bh = buy_and_hold(CRYPTO, CRYPTO_BASE)
    bh_sr = observed_sharpe(bh.to_numpy(), periods_per_year=CRYPTO_PERIODS)
    lines += ["", f"Benchmark buy&hold (equal-weight cripto): Sharpe_anual={bh_sr:.2f}", ""]

    # Tribunal sobre o MELHOR config, penalizado por n_trials = tamanho da grade.
    best_p, best_sr, best_port = results[0]
    lines += ["-" * 92, f"TRIBUNAL — melhor config ({best_p.label()}), n_trials={n_trials}:", "-" * 92]
    v = evaluate_edge(best_port.to_numpy(), n_trials=n_trials, periods_per_year=CRYPTO_PERIODS)
    lines.append("  custo BASE (70bps):    " + v.summary())
    vs = evaluate_edge(
        portfolio(CRYPTO, best_p, CRYPTO_BASE.stressed(2.0)).to_numpy(),
        n_trials=n_trials, periods_per_year=CRYPTO_PERIODS,
    )
    lines.append("  custo ESTRESSADO 2x:   " + vs.summary())

    # Config robusto (mediana da grade) — evita cherry-picking do topo.
    mid = results[len(results) // 2]
    vm = evaluate_edge(mid[2].to_numpy(), n_trials=n_trials, periods_per_year=CRYPTO_PERIODS)
    lines += ["", f"Config MEDIANO da grade ({mid[0].label()}, Sharpe={mid[1]:.2f}) — sanidade:"]
    lines.append("  " + vm.summary())

    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
