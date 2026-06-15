"""Testes do DecisionLog (persistencia da Camada 0)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from feedback.decision_log import DecisionLog
from feedback.models import (
    Decision,
    DecisionAction,
    MarketRegime,
    Outcome,
    OutcomeStatus,
)


@pytest.fixture()
def log(tmp_path):
    lg = DecisionLog(db_path=tmp_path / "fb.sqlite")
    yield lg
    lg.close()


def test_registra_decisao_de_trade_nasce_open(log):
    did = log.record(
        Decision(
            strategy="trailing_stop",
            symbol="aapl",
            action=DecisionAction.BUY,
            regime=MarketRegime.TREND_UP,
            reference_price=Decimal("190.5"),
            signal_strength=0.7,
            context={"rsi": 55},
            client_order_id="coid-1",
        )
    )
    rows = log.all_records()
    assert len(rows) == 1
    assert rows[0]["id"] == did
    assert rows[0]["symbol"] == "aapl"
    assert rows[0]["outcome_status"] == OutcomeStatus.OPEN.value
    assert log.open_decisions()[0]["client_order_id"] == "coid-1"


def test_skip_nasce_skipped(log):
    log.record(
        Decision(strategy="wheel", symbol="MSFT", action=DecisionAction.SKIP)
    )
    rows = log.all_records()
    assert rows[0]["outcome_status"] == OutcomeStatus.SKIPPED.value
    assert log.open_decisions() == []  # skip nao e trade aberto


def test_anexa_resultado_por_id(log):
    did = log.record(
        Decision(strategy="s", symbol="X", action=DecisionAction.BUY)
    )
    log.attach_outcome(
        did,
        Outcome(
            status=OutcomeStatus.WIN,
            entry_price=Decimal("10"),
            exit_price=Decimal("11"),
            realized_pnl=Decimal("100"),
            return_pct=0.1,
        ),
    )
    row = log.all_records()[0]
    assert row["outcome_status"] == OutcomeStatus.WIN.value
    assert row["realized_pnl"] == "100"
    assert row["closed_at"] is not None
    assert log.open_decisions() == []


def test_anexa_resultado_por_client_order_id(log):
    log.record(
        Decision(
            strategy="s", symbol="X", action=DecisionAction.BUY, client_order_id="abc"
        )
    )
    n = log.attach_outcome_by_client_order_id(
        "abc", Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-50"))
    )
    assert n == 1
    assert log.all_records()[0]["outcome_status"] == OutcomeStatus.LOSS.value


def test_persiste_entre_conexoes(tmp_path):
    p = tmp_path / "fb.sqlite"
    lg1 = DecisionLog(db_path=p)
    lg1.record(Decision(strategy="s", symbol="X", action=DecisionAction.BUY))
    lg1.close()
    lg2 = DecisionLog(db_path=p)
    assert len(lg2.all_records()) == 1
    lg2.close()
