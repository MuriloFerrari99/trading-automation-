"""Garantia estrutural: sinais sao apenas sugestoes e NUNCA viram ordens.

Mesmo com um sinal de COMPRA forte presente, nenhum ativo deve ser comprado
por causa dele. So estrategias de mercado (Nivel 1) chegam ao Executor.
"""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from agents.planner import Planner
from config.watchlist import Watchlist, WatchlistItem
from core.models import OrderSide
from data.signal_repo import SignalRepository
from orchestration.local_orchestrator import LocalOrchestrator
from strategies.signals.base import SignalService
from strategies.signals.smart_money import (
    FundPositionChange,
    SmartMoneyProvider,
    StaticSmartMoneySource,
)
from strategies.trailing_stop import TrailingStopStrategy


def test_signal_does_not_become_order(db, broker, state, trade_logger, kill_switch):
    # Sinal forte de COMPRA em NVDA.
    src = StaticSmartMoneySource(
        [FundPositionChange(symbol="NVDA", fund="BigFund", side=OrderSide.BUY, change_ratio=Decimal("0.9"))]
    )
    signal_service = SignalService([SmartMoneyProvider(src)])
    signal_repo = SignalRepository(db)

    # Watchlist sem nada que dispare (AAPL trailing, sem posicao).
    watchlist = Watchlist(
        items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))]
    )
    broker.set_price("AAPL", "100")

    planner = Planner(
        broker, state, watchlist, [TrailingStopStrategy()],
        signal_service=signal_service, signal_repo=signal_repo,
    )
    executor = Executor(broker, trade_logger, kill_switch)
    orch = LocalOrchestrator(planner, executor)

    result = orch.run_cycle()

    # O sinal foi coletado e registrado como sugestao...
    assert result.signals_count == 1
    assert result.signals[0].symbol == "NVDA"
    assert len(signal_repo.recent()) == 1

    # ...mas NENHUMA ordem foi gerada/executada por causa dele.
    assert result.intents_count == 0
    assert result.executed_count == 0
    assert broker.submitted == []
    assert trade_logger.recent() == []
    assert broker.get_position("NVDA") is None
