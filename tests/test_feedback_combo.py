"""Testa o cruzamento estrategia x regime na avaliacao."""

from __future__ import annotations

from decimal import Decimal

from feedback.decision_log import DecisionLog
from feedback.evaluation import evaluate, format_report
from feedback.models import Decision, DecisionAction, MarketRegime, Outcome, OutcomeStatus


def test_by_strategy_regime(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    # trailing_stop ganha em TREND_UP, perde em RANGE
    for reg, pnl, st in [
        (MarketRegime.TREND_UP, "100", OutcomeStatus.WIN),
        (MarketRegime.TREND_UP, "80", OutcomeStatus.WIN),
        (MarketRegime.RANGE, "-60", OutcomeStatus.LOSS),
        (MarketRegime.RANGE, "-40", OutcomeStatus.LOSS),
    ]:
        did = log.record(
            Decision(strategy="trailing_stop", symbol="AAPL",
                     action=DecisionAction.BUY, regime=reg)
        )
        log.attach_outcome(did, Outcome(status=st, realized_pnl=Decimal(pnl)))

    rep = evaluate(log.all_records())
    up = rep.combo("trailing_stop", MarketRegime.TREND_UP.value)
    rng = rep.combo("trailing_stop", MarketRegime.RANGE.value)
    assert up.n_trades == 2 and up.total_pnl == 180.0
    assert rng.n_trades == 2 and rng.total_pnl == -100.0
    assert rng.expectancy < 0  # combo perdedor

    txt = format_report(rep)
    assert "expectancy NEGATIVA" in txt
    assert "trailing_stop@range" in txt
    log.close()
