"""Testa a integracao opcional da camada de decisao no LocalOrchestrator."""

from __future__ import annotations

from decimal import Decimal

from core.models import OrderIntent, OrderSide
from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, Outcome, OutcomeStatus
from intelligence.decision_policy import DecisionPolicy
from intelligence.engine import DecisionIntelligence
from orchestration.local_orchestrator import LocalOrchestrator


class _StubPlanner:
    def __init__(self, intents):
        self._intents = intents

    def gather_signals(self):
        return []

    def plan(self):
        return list(self._intents)


class _StubExecutor:
    def __init__(self):
        self.seen = None

    def execute_many(self, intents):
        self.seen = intents
        return [object() for _ in intents]  # 1 "resultado" por intent


def _buy(strategy="trailing_stop"):
    return OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"), strategy=strategy
    )


def test_sem_inteligencia_executa_tudo():
    ex = _StubExecutor()
    orch = LocalOrchestrator(_StubPlanner([_buy()]), ex)
    res = orch.run_cycle()
    assert len(ex.seen) == 1
    assert res.executed_count == 1


def test_inteligencia_veta_e_nao_executa(tmp_path):
    log = DecisionLog(db_path=tmp_path / "fb.sqlite")
    for _ in range(15):  # historico perdedor p/ (trailing_stop, unknown)
        did = log.record(
            Decision(strategy="trailing_stop", symbol="AAPL",
                     action=DecisionAction.BUY, regime=MarketRegime.UNKNOWN)
        )
        log.attach_outcome(did, Outcome(status=OutcomeStatus.LOSS, realized_pnl=Decimal("-5")))

    intel = DecisionIntelligence(
        log, DecisionPolicy(min_samples=12), price_provider=lambda s: Decimal("100")
    )
    ex = _StubExecutor()
    orch = LocalOrchestrator(_StubPlanner([_buy()]), ex, intelligence=intel)
    res = orch.run_cycle()
    # tudo vetado -> curto-circuito: o Executor nem e chamado
    assert ex.seen is None
    assert res.executed_count == 0
    assert res.intents_count == 0
    log.close()
