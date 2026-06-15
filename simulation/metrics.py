"""Metricas de performance (doc 06 §3).

Leia em conjunto — nenhuma isolada conta a historia toda. Tudo em numpy, sobre
a curva de equity e os retornos por trade de UM backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class PerfMetrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float  # <= 0
    win_rate: float
    profit_factor: float
    exposure: float
    n_trades: int

    def as_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "exposure": self.exposure,
            "n_trades": self.n_trades,
        }


def sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - rf
    sd = excess.std(ddof=1)
    return float(np.sqrt(periods) * excess.mean() / sd) if sd > 0 else 0.0


def sortino(returns: np.ndarray, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - rf
    downside = excess[excess < 0]
    dd = downside.std(ddof=1) if downside.size >= 2 else 0.0
    return float(np.sqrt(periods) * excess.mean() / dd) if dd > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def compute_metrics(
    equity: np.ndarray,
    trade_pnls: list[float],
    *,
    n_bars: int,
    bars_in_market: int,
    periods: int = TRADING_DAYS,
) -> PerfMetrics:
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2 or equity[0] <= 0:
        return PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0, len(trade_pnls))

    pnls_arr = np.asarray(trade_pnls, dtype=float)
    years = max(n_bars / periods, 1e-9)

    # RUINA: se o equity zerou/ficou negativo, a conta foi estourada. Retornos/
    # CAGR/MDD ficam definidos como perda total (-100%) — sem numeros absurdos.
    if (equity <= 0).any():
        n = int(pnls_arr.size)
        win_rate = float((pnls_arr > 0).sum() / n) if n else 0.0
        exposure = float(bars_in_market / n_bars) if n_bars else 0.0
        return PerfMetrics(-1.0, -1.0, 0.0, 0.0, -1.0, win_rate, 0.0, exposure, n)

    rets = np.diff(equity) / equity[:-1]
    total_return = float(equity[-1] / equity[0] - 1.0)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)

    pnls = np.asarray(trade_pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(wins.size / pnls.size) if pnls.size else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    exposure = float(bars_in_market / n_bars) if n_bars else 0.0

    return PerfMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe(rets, periods=periods),
        sortino=sortino(rets, periods=periods),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        profit_factor=profit_factor,
        exposure=exposure,
        n_trades=int(pnls.size),
    )
