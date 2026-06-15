"""Testes do DecisionIntelligence (registro + gate por regime)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.models import OrderIntent, OrderSide, Signal
from feedback.decision_log import DecisionLog
from feedback.models import (
    Decision,
    DecisionAction,
    MarketRegime,
    Outcome,
    OutcomeStatus,
)
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence


@pytest.fixture()
def log(tmp_path):
    lg = DecisionLog(db_path=tmp_path / "fb.sqlite")
    yield lg
    lg.close()


def _price(_symbol: str) -> Decimal:
    return Decimal("100")  # 1 amostra -> regime UNKNOWN (suficiente para o gate)


def _buy(symbol="AAPL", strategy="trailing_stop") -> OrderIntent:
    return OrderIntent(
        symbol=symbol, side=OrderSide.BUY, qty=Decimal("1"), strategy=strategy
    )


def _seed_losers(log: DecisionLog, n: int, *, strategy: str, regime: MarketRegime):
    for _ in range(n):
        did = log.record(
            Decision(strategy=strategy, symbol="AAPL", action=DecisionAction.BUY, regime=regime)
        )
        log.attach_outcome(
            did, Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-10"), return_pct=-0.01)
        )


def test_sem_historico_permite_e_registra_open(log):
    eng = DecisionIntelligence(log, DecisionPolicy(), price_provider=_price)
    res = eng.process([_buy()])
    assert len(res.allowed) == 1
    assert res.blocked_count == 0
    rows = log.all_records()
    assert rows[-1]["action"] == "buy"
    assert rows[-1]["outcome_status"] == OutcomeStatus.OPEN.value
    assert rows[-1]["regime"] == MarketRegime.UNKNOWN.value


def test_veta_combo_perdedor_e_registra_skip(log):
    # historico ruim para (trailing_stop, unknown): 15 perdas
    _seed_losers(log, 15, strategy="trailing_stop", regime=MarketRegime.UNKNOWN)
    eng = DecisionIntelligence(log, DecisionPolicy(min_samples=12), price_provider=_price)
    res = eng.process([_buy(strategy="trailing_stop")])
    assert res.allowed == []
    assert res.blocked_count == 1
    assert log.all_records()[-1]["action"] == "skip"


def test_combo_diferente_nao_e_afetado(log):
    # historico ruim so para trailing_stop; ladder_buys deve passar
    _seed_losers(log, 15, strategy="trailing_stop", regime=MarketRegime.UNKNOWN)
    eng = DecisionIntelligence(log, DecisionPolicy(min_samples=12), price_provider=_price)
    res = eng.process([_buy(strategy="ladder_buys")])
    assert len(res.allowed) == 1


def test_sinal_alinhado_registrado_no_contexto(log):
    eng = DecisionIntelligence(log, DecisionPolicy(), price_provider=_price)
    sig = Signal(symbol="AAPL", side=OrderSide.BUY, source="congress", confidence=0.9)
    eng.process([_buy(symbol="AAPL")], signals=[sig])
    row = log.all_records()[-1]
    assert row["signal_strength"] == pytest.approx(0.9)


def test_sem_price_provider_regime_unknown_e_permite(log):
    eng = DecisionIntelligence(log, DecisionPolicy())  # sem price_provider
    res = eng.process([_buy()])
    assert len(res.allowed) == 1
    assert log.all_records()[-1]["reference_price"] is None
