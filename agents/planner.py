"""Agente Planejador.

Consome dados de mercado (via estrategias) e produz intencoes de ordem.
No MVP roda a(s) estrategia(s) configurada(s) e agrega as OrderIntent.
Nao executa nada — apenas decide.
"""

from __future__ import annotations

import logging

from broker.base import BrokerClient
from config.watchlist import Watchlist
from core.models import Signal
from data.audit_log import PLANNER, AuditLog
from data.signal_repo import SignalRepository
from data.state_repo import StateRepository
from risk.manager import RiskManager
from strategies.base import Strategy, StrategyContext, TradeIntent
from strategies.signals.base import SignalService

logger = logging.getLogger("agent.planner")


class Planner:
    def __init__(
        self,
        broker: BrokerClient,
        state: StateRepository,
        watchlist: Watchlist,
        strategies: list[Strategy],
        *,
        signal_service: SignalService | None = None,
        signal_repo: SignalRepository | None = None,
        risk_manager: RiskManager | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._broker = broker
        self._state = state
        self._watchlist = watchlist
        self._strategies = strategies
        self._signal_service = signal_service
        self._signal_repo = signal_repo
        self._risk = risk_manager
        self._audit = audit

    def gather_signals(self) -> list[Signal]:
        """Coleta sinais dos providers e os registra como SUGESTOES.

        IMPORTANTE: sinais NUNCA sao convertidos em OrderIntent aqui. Eles sao
        apenas logados e persistidos para revisao humana. Por construcao, o
        retorno deste metodo nao alimenta o Executor — a separacao entre
        `gather_signals()` (sugestao) e `plan()` (execucao) e proposital.
        """
        if self._signal_service is None:
            return []
        signals = self._signal_service.collect()
        for s in signals:
            logger.info(
                "SUGESTAO [%s] %s %s conf=%.2f — %s",
                s.source, s.side.value, s.symbol, s.confidence, s.note,
            )
            if self._signal_repo is not None:
                self._signal_repo.record(s)
        return signals

    def plan(self) -> list[TradeIntent]:
        ctx = StrategyContext(
            broker=self._broker, state=self._state, watchlist=self._watchlist
        )
        intents: list[TradeIntent] = []
        for strategy in self._strategies:
            try:
                produced = strategy.evaluate(ctx)
            except Exception:  # uma estrategia com erro nao derruba as demais
                logger.exception("Estrategia '%s' falhou ao avaliar", strategy.name)
                continue
            if produced:
                logger.info("Estrategia '%s' gerou %d intencao(oes)", strategy.name, len(produced))
            intents.extend(produced)

        # Camada de decisao/risco: veta/ajusta cada intencao antes de sair.
        if self._risk is not None:
            intents = self._apply_risk(intents)
        return intents

    def _apply_risk(self, intents: list[TradeIntent]) -> list[TradeIntent]:
        account = self._broker.get_account()
        positions = self._broker.get_positions()

        halt = self._risk.begin_cycle(account.equity)
        if halt:
            self._audit_risk("risk_halt", None, {"reason": halt})

        approved: list[TradeIntent] = []
        for intent in intents:
            symbol = intent.symbol if hasattr(intent, "symbol") else intent.underlying
            price = self._safe_price(symbol)
            decision = self._risk.assess(intent, account.equity, positions, price)
            if decision.approved and decision.intent is not None:
                if decision.intent is not intent:
                    self._audit_risk("risk_adjusted", symbol, {"reason": decision.reason})
                approved.append(decision.intent)
            else:
                logger.info("Risco vetou %s: %s", symbol, decision.reason)
                self._audit_risk("risk_vetoed", symbol, {"reason": decision.reason})
        return approved

    def _safe_price(self, symbol: str):
        try:
            return self._broker.get_last_price(symbol)
        except Exception:
            return None

    def _audit_risk(self, event: str, symbol: str | None, payload: dict) -> None:
        if self._audit is not None:
            self._audit.write(PLANNER, event, symbol=symbol, payload=payload)
