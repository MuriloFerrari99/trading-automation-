"""Testes do FrozenRegimeGate e do roteamento risco/gate no engine (offline)."""

from __future__ import annotations

from decimal import Decimal

from data.db import Database
from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, Outcome, OutcomeStatus
from simulation.engine import run_backtest
from simulation.experiment_oos import split_is_oos
from simulation.gate import FrozenRegimeGate


def _series(closes):
    opens = [closes[0]] + closes[:-1]
    return {"open": opens, "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes], "close": closes}


def test_gate_vetoes_negative_edge_combo():
    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    # 8 trades perdedores de ladder@trend_down
    for _ in range(8):
        did = log.record(Decision(strategy="ladder_buys", symbol="X",
                                   action=DecisionAction.BUY, regime=MarketRegime.TREND_DOWN,
                                   reference_price=Decimal("100")))
        log.attach_outcome(did, Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-50"),
                                        return_pct=-0.05))
    gate = FrozenRegimeGate.from_decision_log(log, min_samples=6)
    assert gate.allow("ladder_buys", "trend_down") is False  # vetado
    assert gate.allow("ladder_buys", "trend_up") is True      # sem historico => permite
    assert gate.allow("trailing_stop", "trend_down") is True  # outra estrategia


def test_gate_needs_min_samples():
    db = Database(":memory:")
    log = DecisionLog(connection=db.conn)
    for _ in range(3):  # poucas amostras
        did = log.record(Decision(strategy="ladder_buys", symbol="X",
                                   action=DecisionAction.BUY, regime=MarketRegime.TREND_DOWN,
                                   reference_price=Decimal("100")))
        log.attach_outcome(did, Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-50"), return_pct=-0.05))
    gate = FrozenRegimeGate.from_decision_log(log, min_samples=6)
    assert gate.allow("ladder_buys", "trend_down") is True  # amostra insuficiente => nao veta


def test_engine_gate_makes_strategy_flat():
    ohlc = _series([100 + (i % 5) for i in range(80)])
    r_open = run_backtest("X", ohlc, "ladder_buys", warmup=35)
    veto_all = FrozenRegimeGate({("ladder_buys", r_open.regime)})  # veta o regime desta janela
    r_vetoed = run_backtest("X", ohlc, "ladder_buys", warmup=35, gate=veto_all)
    assert r_vetoed.metrics.n_trades == 0       # vetado => nao operou
    assert r_vetoed.metrics.total_return == 0.0  # ficou flat


def test_split_is_oos():
    ohlc = _series(list(range(100, 300)))
    is_ohlc, oos = split_is_oos(ohlc, 0.6)
    assert len(is_ohlc["close"]) == 120
    assert len(oos["close"]) == 80


def test_engine_use_risk_runs():
    ohlc = _series([100 * 0.99 ** i for i in range(120)])  # downtrend
    r = run_backtest("X", ohlc, "ladder_buys", warmup=35, use_risk=True)
    assert r is not None  # roda com risco sem erro
