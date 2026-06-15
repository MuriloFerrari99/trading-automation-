"""Testes da avaliacao (metricas por estrategia e por regime)."""

from __future__ import annotations

import math
from decimal import Decimal

from feedback.decision_log import DecisionLog
from feedback.evaluation import evaluate, format_report
from feedback.models import (
    Decision,
    DecisionAction,
    MarketRegime,
    Outcome,
    OutcomeStatus,
)


def _seed(log: DecisionLog) -> None:
    # 3 trades trailing_stop em TREND_UP: +100, +200, -50
    for coid, pnl, ret, status in [
        ("a", "100", 0.10, OutcomeStatus.WIN),
        ("b", "200", 0.20, OutcomeStatus.WIN),
        ("c", "-50", -0.05, OutcomeStatus.LOSS),
    ]:
        did = log.record(
            Decision(
                strategy="trailing_stop",
                symbol="AAPL",
                action=DecisionAction.BUY,
                regime=MarketRegime.TREND_UP,
                client_order_id=coid,
            )
        )
        log.attach_outcome(
            did,
            Outcome(status=status, realized_pnl=Decimal(pnl), return_pct=ret),
        )
    # 1 trade wheel em RANGE: -30
    did = log.record(
        Decision(
            strategy="wheel",
            symbol="MSFT",
            action=DecisionAction.SELL,
            regime=MarketRegime.RANGE,
        )
    )
    log.attach_outcome(
        did, Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-30"), return_pct=-0.03)
    )
    # 1 skip (nao opera)
    log.record(Decision(strategy="wheel", symbol="MSFT", action=DecisionAction.SKIP))
    # 1 trade aberto
    log.record(
        Decision(strategy="trailing_stop", symbol="NVDA", action=DecisionAction.BUY)
    )


def test_metricas_globais(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    _seed(log)
    rep = evaluate(log.all_records())
    o = rep.overall
    assert o.n_decisions == 6
    assert o.n_trades == 4  # 4 fechados
    assert o.n_skips == 1
    assert o.n_open == 1
    assert o.wins == 2
    assert o.losses == 2
    assert o.total_pnl == 220.0  # 100+200-50-30
    assert o.gross_profit == 300.0
    assert o.gross_loss == -80.0
    assert math.isclose(o.profit_factor, 300.0 / 80.0)
    assert math.isclose(o.expectancy, 220.0 / 4)
    log.close()


def test_quebra_por_estrategia_e_regime(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    _seed(log)
    rep = evaluate(log.all_records())

    ts = rep.by_strategy["trailing_stop"]
    assert ts.n_trades == 3
    assert ts.total_pnl == 250.0
    assert ts.wins == 2 and ts.losses == 1

    wheel = rep.by_strategy["wheel"]
    assert wheel.n_trades == 1
    assert wheel.total_pnl == -30.0
    assert wheel.n_skips == 1

    trend = rep.by_regime[MarketRegime.TREND_UP.value]
    assert trend.n_trades == 3
    assert trend.total_pnl == 250.0
    log.close()


def test_max_drawdown(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    # sequencia: +100, +200, -50 -> curva 100,300,250 -> pico 300, DD = -50
    _seed(log)
    rep = evaluate(log.all_records())
    assert rep.by_strategy["trailing_stop"].max_drawdown == -50.0
    log.close()


def test_profit_factor_infinito_sem_perdas(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    did = log.record(Decision(strategy="s", symbol="X", action=DecisionAction.BUY))
    log.attach_outcome(did, Outcome(status=OutcomeStatus.WIN, realized_pnl=Decimal("10")))
    rep = evaluate(log.all_records())
    assert rep.overall.profit_factor == math.inf
    log.close()


def test_format_report_roda(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    _seed(log)
    txt = format_report(evaluate(log.all_records()))
    assert "LOOP DE FEEDBACK" in txt
    assert "trailing_stop" in txt
    assert "trend_up" in txt
    log.close()
