"""Experimento controlado: quanto as camadas de inteligencia melhoram o modelo.

Roda o LADDER (a estrategia que o sweep mostrou frágil) em backtest CONTINUO por
simbolo, em 3 configuracoes, e mede no trecho OUT-OF-SAMPLE (anti-overfitting):

  A) baseline      — ladder cru (esperado: ruina em downtrend)
  B) +risco        — RiskManager (cap por simbolo + buying power => sem caixa
                     negativo) e stop de invalidacao global do ladder
  C) +risco+decisao— B + DecisionIntelligence aprendendo ONLINE: acumula track
                     record e VETA combos estrategia@regime de edge negativo

Comparacao pareada (mesmos simbolos e mesmo trecho OOS), com estresse de custos
(5/10/15 bps). Metricas medidas SO no OOS (equity do trecho de validacao).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from agents.feedback_agent import FeedbackAgent
from config.risk import RiskSettings
from config.watchlist import LadderConfig, LadderRung, Watchlist, WatchlistItem
from data.db import Database
from feedback.decision_log import DecisionLog
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence
from orchestration.bus import InMemoryBus
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard
from simulation.engine import _SimState
from simulation.metrics import PerfMetrics, compute_metrics
from simulation.sim_broker import SimBroker
from strategies.base import StrategyContext
from strategies.ladder_buys import LadderBuysStrategy

logger = logging.getLogger("simulation.experiment")

CASH = 100_000.0
WARMUP = 35
LADDER_STOP_PCT = Decimal("0.40")  # tese quebrada a -40% da ancora


@dataclass
class RunConfig:
    name: str
    use_risk: bool = False
    use_decision: bool = False
    use_stop: bool = False
    commission_bps: float = 5.0
    slippage_bps: float = 5.0


def _ladder_watchlist(symbol: str, use_stop: bool) -> Watchlist:
    return Watchlist(items=[WatchlistItem(symbol=symbol, ladder=LadderConfig(
        rungs=[
            LadderRung(drop_pct=Decimal("0.10"), qty=Decimal("10")),
            LadderRung(drop_pct=Decimal("0.20"), qty=Decimal("20")),
            LadderRung(drop_pct=Decimal("0.30"), qty=Decimal("30")),
        ],
        stop_loss_pct=LADDER_STOP_PCT if use_stop else None,
    ))])


def _make_risk() -> RiskManager:
    settings = RiskSettings(_env_file=None)  # type: ignore[call-arg]
    return RiskManager(settings, PortfolioRiskGuard(Decimal(str(CASH))))


def run_continuous(
    symbol: str,
    ohlc: dict[str, list[float]],
    cfg: RunConfig,
    *,
    decision: DecisionIntelligence | None = None,
    feedback: FeedbackAgent | None = None,
    is_frac: float = 0.6,
) -> PerfMetrics | None:
    closes = ohlc["close"]
    n = len(closes)
    if n < WARMUP + 30:
        return None

    sim = SimBroker(
        symbol, ohlc["open"], ohlc["high"], ohlc["low"], closes,
        cash=CASH, commission_bps=cfg.commission_bps, slippage_bps=cfg.slippage_bps,
    )
    state = _SimState()
    wl = _ladder_watchlist(symbol, cfg.use_stop)
    strat = LadderBuysStrategy()
    risk = _make_risk() if cfg.use_risk else None
    bus = InMemoryBus()
    if decision is not None:
        decision._price_provider = sim.get_last_price  # aponta p/ o sim atual

    for i in range(n):
        sim.set_index(i)
        fills = sim.fill_pending_at_open(i)
        if feedback is not None and fills:
            for f in fills:
                bus.publish("fills", f)
            feedback.step(bus)

        if i >= WARMUP:
            ctx = StrategyContext(broker=sim, state=state, watchlist=wl)
            intents = strat.evaluate(ctx)
            if risk is not None and intents:
                risk.begin_cycle(sim.get_account().equity)
                intents = _apply_risk(intents, sim, risk)
            if decision is not None and intents:
                # Regime a partir de BARRAS REAIS ate o bar atual (sem look-ahead).
                market_data = {symbol: sim.get_bars(symbol, 60)}
                intents = decision.process(intents, [], market_data=market_data).allowed
            for it in intents:
                sim.submit_order(it)
        sim.mark_equity()

    sim.set_index(n - 1)
    final_fill = sim.liquidate_final()
    if feedback is not None and final_fill:
        bus.publish("fills", final_fill)
        feedback.step(bus)

    # Metricas SO no trecho OOS (anti-overfitting).
    equity = np.asarray(sim.equity_curve, dtype=float)
    is_idx = int(len(equity) * is_frac)
    oos_equity = equity[is_idx:]
    if oos_equity.size < 2:
        return None
    return compute_metrics(
        oos_equity, sim.trade_pnls,
        n_bars=oos_equity.size, bars_in_market=sim.bars_in_market,
    )


def _apply_risk(intents, sim, risk):
    equity = sim.get_account().equity
    positions = sim.get_positions()
    out = []
    for it in intents:
        price = sim.get_last_price(it.symbol)
        d = risk.assess(it, equity, positions, price)
        if d.approved and d.intent is not None:
            out.append(d.intent)
    return out


@dataclass
class ConfigResult:
    name: str
    cost_bps: float
    n_symbols: int = 0
    n_ruined: int = 0
    pct_profitable: float = 0.0
    median_return: float = 0.0
    mean_return: float = 0.0
    mean_sharpe: float = 0.0
    mean_max_dd: float = 0.0
    worst_max_dd: float = 0.0
    _rets: list[float] = field(default_factory=list, repr=False)


def _aggregate(name: str, cost: float, metrics: list[PerfMetrics]) -> ConfigResult:
    import statistics

    rets = [m.total_return for m in metrics]
    dds = [m.max_drawdown for m in metrics]
    return ConfigResult(
        name=name, cost_bps=cost, n_symbols=len(metrics),
        n_ruined=sum(1 for m in metrics if m.max_drawdown <= -0.999),
        pct_profitable=sum(1 for r in rets if r > 0) / len(rets) if rets else 0.0,
        median_return=statistics.median(rets) if rets else 0.0,
        mean_return=statistics.fmean(rets) if rets else 0.0,
        mean_sharpe=statistics.fmean([m.sharpe for m in metrics]) if metrics else 0.0,
        mean_max_dd=statistics.fmean(dds) if dds else 0.0,
        worst_max_dd=min(dds) if dds else 0.0,
    )


def run_experiment(
    data: dict[str, dict[str, list[float]]],
    *,
    costs: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[ConfigResult]:
    results: list[ConfigResult] = []
    for cost in costs:
        # A) baseline cru
        metrics_a = [
            m for sym, ohlc in data.items()
            if (m := run_continuous(sym, ohlc, RunConfig("baseline", commission_bps=cost, slippage_bps=cost))) is not None
        ]
        results.append(_aggregate("A_baseline", cost, metrics_a))

        # B) +risco +stop de invalidacao
        cfg_b = RunConfig("risk", use_risk=True, use_stop=True, commission_bps=cost, slippage_bps=cost)
        metrics_b = [
            m for sym, ohlc in data.items()
            if (m := run_continuous(sym, ohlc, cfg_b)) is not None
        ]
        results.append(_aggregate("B_risk+stop", cost, metrics_b))

        # C) +risco +stop +decisao. DUAS PASSADAS:
        #   1) treino: roda a serie inteira p/ a DecisionPolicy acumular track
        #      record (regime por BARRAS reais) — fecha o loop via FeedbackAgent.
        #   2) avaliacao: roda de novo; agora a policy VETA combos negativos.
        # Politica mais sensivel (min_samples menor) p/ agir com a amostra que ha.
        db = Database(":memory:")
        dlog = DecisionLog(connection=db.conn)
        decision = DecisionIntelligence(dlog, DecisionPolicy(min_samples=6))
        feedback = FeedbackAgent(dlog)
        cfg_c = RunConfig("risk+decision", use_risk=True, use_stop=True, commission_bps=cost, slippage_bps=cost)

        for sym, ohlc in data.items():  # passada 1 — treino (descarta metricas)
            run_continuous(sym, ohlc, cfg_c, decision=decision, feedback=feedback)

        vetoed = _vetoed_combos(dlog)
        logger.info("Custo %.0fbps: policy aprendeu a vetar %d combo(s): %s", cost, len(vetoed), vetoed)

        metrics_c = [  # passada 2 — avaliacao OOS com o gate ativo
            m for sym, ohlc in data.items()
            if (m := run_continuous(sym, ohlc, cfg_c, decision=decision, feedback=feedback)) is not None
        ]
        results.append(_aggregate("C_risk+decision", cost, metrics_c))
        db.close()

    return results


def _vetoed_combos(dlog: DecisionLog) -> list[str]:
    """Combos estrategia@regime que a policy aprendeu a vetar (expectancy < 0)."""
    from feedback.evaluation import evaluate

    report = evaluate(dlog.all_records())
    out = []
    for (strat, regime), stats in report.by_strategy_regime.items():
        if stats.n_trades >= 6 and stats.expectancy < 0:
            out.append(f"{strat}@{regime}(n={stats.n_trades},exp={stats.expectancy:.1f})")
    return out


def format_experiment(results: list[ConfigResult]) -> str:
    lines = ["=" * 110, "EXPERIMENTO — ganho das camadas (ladder, OOS, dados reais Alpaca)", "=" * 110]
    lines.append(
        f"{'config':<18} {'custo':>5} {'n':>4} {'ruina':>6} {'prof%':>6} "
        f"{'retMed':>8} {'retMed_avg':>10} {'shpe':>6} {'mddMed':>8} {'mddPior':>8}"
    )
    lines.append("-" * 110)
    for r in results:
        lines.append(
            f"{r.name:<18} {r.cost_bps:>4.0f}b {r.n_symbols:>4} {r.n_ruined:>6} "
            f"{r.pct_profitable * 100:>5.1f}% {r.median_return * 100:>7.2f}% "
            f"{r.mean_return * 100:>9.2f}% {r.mean_sharpe:>6.2f} "
            f"{r.mean_max_dd * 100:>7.1f}% {r.worst_max_dd * 100:>7.1f}%"
        )
    lines.append("=" * 110)
    lines.append("ruina=contas estouradas (MDD<=-100%) | prof%=simbolos lucrativos no OOS | retMed=retorno mediano OOS")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Experimento de melhoria do modelo (ladder)")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--no-crypto", action="store_true")
    parser.add_argument("--report-file", default="data/experiment_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    from simulation.data import fetch_daily, to_ohlc_lists
    from simulation.run import DEFAULT_CRYPTO, DEFAULT_STOCKS

    crypto = [] if args.no_crypto else DEFAULT_CRYPTO
    frames = fetch_daily(DEFAULT_STOCKS, crypto, years=args.years)
    data = {sym: to_ohlc_lists(df) for sym, df in frames.items()}
    logging.getLogger("simulation.experiment").info("Dados: %d simbolos.", len(data))

    results = run_experiment(data)
    text = format_experiment(results)
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
