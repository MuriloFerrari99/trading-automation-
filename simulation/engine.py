"""Engine de backtest: roda UMA estrategia sobre UMA serie de precos.

Dirige o SimBroker bar-a-bar (sem look-ahead) e devolve metricas + o regime em
que a janela operou (rotulado por feedback/regime.py), para o breakdown por
estrategia@regime.

Notas por estrategia:
- Trailing stop e PROTETIVO: precisa de uma posicao. O backtest entra long no
  inicio da janela e deixa o trailing proteger (mede preservacao de capital).
- Ladder e de ENTRADA: comeca flat e compra nas quedas; liquida no fim.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from config.watchlist import LadderConfig, LadderRung, WatchlistItem, Watchlist
from feedback.regime import classify_regime
from simulation.metrics import PerfMetrics, compute_metrics
from simulation.sim_broker import SimBroker
from strategies.base import StrategyContext
from strategies.ladder_buys import LadderBuysStrategy
from strategies.trailing_stop import TrailingStopStrategy


class _SimState:
    """Estado chave/valor em memoria (mesma interface do StateRepository)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def set(self, key: str, value: str) -> None:
        self._d[key] = value

    def delete(self, key: str) -> None:
        self._d.pop(key, None)

    def get_decimal(self, key: str) -> Decimal | None:
        v = self._d.get(key)
        return Decimal(v) if v is not None else None

    def set_decimal(self, key: str, value: Decimal) -> None:
        self._d[key] = str(value)


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    regime: str
    metrics: PerfMetrics


def _make_sim_risk(cash: float):
    from config.risk import RiskSettings
    from risk.manager import RiskManager
    from risk.portfolio_guard import PortfolioRiskGuard

    return RiskManager(RiskSettings(_env_file=None), PortfolioRiskGuard(Decimal(str(cash))))


def _route_risk(intents, sim, risk):
    equity = sim.get_account().equity
    positions = sim.get_positions()
    out = []
    for it in intents:
        d = risk.assess(it, equity, positions, sim.get_last_price(it.symbol))
        if d.approved and d.intent is not None:
            out.append(d.intent)
    return out


def _watchlist_for(symbol: str, strategy_name: str) -> Watchlist:
    if strategy_name == "trailing_stop":
        return Watchlist(items=[WatchlistItem(symbol=symbol, trailing_stop_pct=Decimal("0.10"))])
    # ladder_buys
    return Watchlist(items=[WatchlistItem(symbol=symbol, ladder=LadderConfig(
        rungs=[
            LadderRung(drop_pct=Decimal("0.10"), qty=Decimal("10")),
            LadderRung(drop_pct=Decimal("0.20"), qty=Decimal("20")),
            LadderRung(drop_pct=Decimal("0.30"), qty=Decimal("30")),
        ],
    ))])


def run_backtest(
    symbol: str,
    ohlc: dict[str, list[float]],
    strategy_name: str,
    *,
    cash: float = 100_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    warmup: int = 35,
    use_risk: bool = False,
    gate=None,
) -> BacktestResult | None:
    closes = ohlc["close"]
    n = len(closes)
    if n < warmup + 5:
        return None

    regime = classify_regime([float(c) for c in closes]).value

    # Gate de regime (config C): se o combo estrategia@regime foi vetado no
    # treino, a estrategia NAO opera nesta janela — fica flat (retorno 0).
    if gate is not None and not gate.allow(strategy_name, regime):
        return BacktestResult(
            symbol=symbol, strategy=strategy_name, regime=regime,
            metrics=compute_metrics(np.asarray([cash, cash], dtype=float), [], n_bars=1, bars_in_market=0),
        )

    sim = SimBroker(
        symbol, ohlc["open"], ohlc["high"], ohlc["low"], closes,
        cash=cash, commission_bps=commission_bps, slippage_bps=slippage_bps,
    )
    state = _SimState()
    wl = _watchlist_for(symbol, strategy_name)
    strategy = TrailingStopStrategy() if strategy_name == "trailing_stop" else LadderBuysStrategy()
    risk = _make_sim_risk(cash) if use_risk else None

    for i in range(n):
        sim.set_index(i)
        sim.fill_pending_at_open(i)

        if i < warmup:
            sim.mark_equity()
            continue
        if i == warmup and strategy_name == "trailing_stop":
            # entra long (posicao a proteger): ~50% do caixa.
            price0 = Decimal(str(closes[i]))
            qty = (Decimal(str(cash)) * Decimal("0.5") / price0).to_integral_value()
            if qty > 0:
                sim._fill_buy(price0, qty)  # entrada no fechamento de warmup

        sim.update_trailing_and_maybe_trigger()
        ctx = StrategyContext(broker=sim, state=state, watchlist=wl)
        intents = strategy.evaluate(ctx)
        if risk is not None and intents:
            risk.begin_cycle(sim.get_account().equity)
            intents = _route_risk(intents, sim, risk)
        for intent in intents:
            sim.submit_order(intent)
        sim.mark_equity()

    sim.set_index(n - 1)
    sim.liquidate_final()

    metrics = compute_metrics(
        np.asarray(sim.equity_curve, dtype=float),
        sim.trade_pnls,
        n_bars=n - warmup,
        bars_in_market=sim.bars_in_market,
    )
    return BacktestResult(symbol=symbol, strategy=strategy_name, regime=regime, metrics=metrics)
