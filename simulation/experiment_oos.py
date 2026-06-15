"""Experimento OOS em escala (>=50k janelas, dados desde 2020).

Pipeline rigoroso (anti-overfitting, doc 06):
  1. Split temporal por simbolo: IN-SAMPLE (treino) vs OUT-OF-SAMPLE (avaliacao).
  2. Treina o FrozenRegimeGate no IS (camada de decisao aprende quais combos
     estrategia@regime tem edge NEGATIVO — agora com COVID-2020 e bear-2022 na
     amostra).
  3. Avalia 3 configs em DEZENAS DE MILHARES de janelas OOS:
       A baseline | B +risco | C +risco+gate (veta combos negativos)
  4. Agrega por config e POR REGIME — onde o gate evita perdas (bear) aparece.

A pergunta que isto responde: a camada de decisao se paga quando ha bear de
verdade na base?
"""

from __future__ import annotations

import logging

from data.db import Database
from feedback.decision_log import DecisionLog
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence
from agents.feedback_agent import FeedbackAgent
from simulation.experiment import RunConfig, run_continuous
from simulation.gate import FrozenRegimeGate
from simulation.montecarlo import GroupSummary, _summarize_group, run_sweep

logger = logging.getLogger("simulation.experiment_oos")


def split_is_oos(ohlc: dict[str, list[float]], is_frac: float):
    n = len(ohlc["close"])
    k = int(n * is_frac)
    is_ohlc = {key: v[:k] for key, v in ohlc.items()}
    oos_ohlc = {key: v[k:] for key, v in ohlc.items()}
    return is_ohlc, oos_ohlc


def train_gate(data, *, is_frac: float = 0.6, min_samples: int = 12) -> tuple[FrozenRegimeGate, DecisionLog]:
    db = Database(":memory:")
    dlog = DecisionLog(connection=db.conn)
    decision = DecisionIntelligence(dlog, DecisionPolicy(min_samples=min_samples))
    fb = FeedbackAgent(dlog)
    cfg = RunConfig("train", use_risk=True, use_stop=True)
    for sym, ohlc in data.items():
        is_ohlc, _ = split_is_oos(ohlc, is_frac)
        if len(is_ohlc["close"]) < 80:
            continue
        # is_frac=1.0: roda a serie IS inteira (so popular o track record).
        run_continuous(sym, is_ohlc, cfg, decision=decision, feedback=fb, is_frac=1.0)
    gate = FrozenRegimeGate.from_decision_log(dlog, min_samples=min_samples)
    return gate, dlog


def run_oos_experiment(
    data,
    *,
    is_frac: float = 0.6,
    window_len: int = 120,
    step: int = 3,
    cost_bps: float = 5.0,
    min_samples: int = 12,
):
    gate, dlog = train_gate(data, is_frac=is_frac, min_samples=min_samples)
    logger.info("Gate treinado. Combos vetados: %s", gate.describe())

    oos = {sym: split_is_oos(ohlc, is_frac)[1] for sym, ohlc in data.items()}

    configs = {
        "A_baseline": dict(use_risk=False, gate=None),
        "B_risk": dict(use_risk=True, gate=None),
        "C_risk+gate": dict(use_risk=True, gate=gate),
    }
    results = {}
    for name, kw in configs.items():
        logger.info("Avaliando config %s ...", name)
        results[name] = run_sweep(
            oos, window_len=window_len, step=step,
            commission_bps=cost_bps, slippage_bps=cost_bps, **kw,
        )
        logger.info("  %s: %d backtests.", name, len(results[name]))
    return gate, results


def _by_regime(results) -> dict[str, GroupSummary]:
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[r.regime].append(r)
    return {reg: _summarize_group(reg, items) for reg, items in groups.items()}


def _row(label: str, s: GroupSummary) -> str:
    return (
        f"{label:<26} n={s.n:<6} prof%={s.pct_profitable * 100:5.1f} "
        f"retMed={s.median_return * 100:6.2f}% retAvg={s.mean_return * 100:6.2f}% "
        f"shpe={s.mean_sharpe:5.2f} mddMed={s.mean_max_dd * 100:6.1f}% mddPior={s.worst_max_dd * 100:7.1f}%"
    )


def format_oos(gate: FrozenRegimeGate, results) -> str:
    total = sum(len(v) for v in results.values())
    lines = ["=" * 124, f"EXPERIMENTO OOS — {total} backtests reais (desde 2020; treino IS, avaliacao OOS)", "=" * 124]
    lines.append(f"Gate aprendeu a vetar: {gate.describe()}")
    lines.append("-" * 124)
    lines.append("POR CONFIG (agregado OOS):")
    for name, res in results.items():
        lines.append("  " + _row(name, _summarize_group(name, res)))
    lines.append("-" * 124)
    lines.append("POR CONFIG x REGIME (onde o gate evita perdas no bear):")
    for name, res in results.items():
        lines.append(f"  [{name}]")
        for reg, s in sorted(_by_regime(res).items()):
            lines.append("    " + _row(reg, s))
    lines.append("=" * 124)
    lines.append("prof%=janelas lucrativas retMed=retorno mediano shpe=Sharpe medio mddPior=pior drawdown")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Experimento OOS em escala (desde 2020)")
    parser.add_argument("--years", type=int, default=7)  # ~2019.5 -> inclui COVID + bear 2022
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--no-crypto", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--report-file", default="data/experiment_oos_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    from simulation.data import fetch_daily, to_ohlc_lists
    from simulation.run import DEFAULT_CRYPTO, DEFAULT_STOCKS

    crypto = [] if args.no_crypto else DEFAULT_CRYPTO
    frames = fetch_daily(DEFAULT_STOCKS, crypto, years=args.years, force=args.force_fetch)
    data = {sym: to_ohlc_lists(df) for sym, df in frames.items()}
    logger.info("Dados: %d simbolos (%d anos).", len(data), args.years)

    gate, results = run_oos_experiment(
        data, window_len=args.window, step=args.step,
        cost_bps=args.cost_bps, min_samples=args.min_samples,
    )
    text = format_oos(gate, results)
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
