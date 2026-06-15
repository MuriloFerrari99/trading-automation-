"""Wrappers que expoem Planner/DecisionIntelligence/Executor como agentes do bus.

Cada wrapper reaproveita a logica existente — so muda a COMUNICACAO para o bus
(consome de um topico, publica em outro). Sao finos de proposito; a inteligencia
real continua em Planner, DecisionIntelligence e Executor.

Fluxo de um ciclo (orquestrado em ordem pelo BusOrchestrator):
  ingestion -> [market_data]
  planner   -> [signals], [intents]
  decision  -> consome intents(+signals) -> [approved]
  executor  -> consome approved -> [fills]
  feedback  -> consome fills -> anexa Outcome
"""

from __future__ import annotations

from agents.base import BaseAgent
from agents.executor import Executor
from agents.planner import Planner
from core.models import OptionOrderIntent
from orchestration.bus import MessageBus

SIGNALS = "signals"
INTENTS = "intents"
APPROVED = "approved"
FILLS = "fills"
MARKET_DATA = "market_data"


def _drain_concat(bus: MessageBus, topic: str) -> list:
    """Drena um topico e concatena payloads (cada payload pode ser lista)."""
    out: list = []
    for msg in bus.drain(topic):
        payload = msg.payload
        out.extend(payload if isinstance(payload, list) else [payload])
    return out


class PlannerAgent(BaseAgent):
    """Roda estrategias + risco (Planner.plan) e coleta sinais. Sem inbox."""

    def __init__(self, planner: Planner) -> None:
        super().__init__("planner")
        self._planner = planner
        self.last_signals: list = []
        self.last_intents: list = []

    def step(self, bus: MessageBus) -> None:
        self.last_signals = self._planner.gather_signals()
        self.last_intents = self._planner.plan()
        bus.publish(SIGNALS, self.last_signals)
        bus.publish(INTENTS, self.last_intents)


class DecisionAgent(BaseAgent):
    """Embrulha o DecisionIntelligence: regime + track-record -> approved."""

    def __init__(self, intelligence) -> None:
        super().__init__("decision")
        self._intel = intelligence
        self.last_approved: list = []

    def step(self, bus: MessageBus) -> None:
        signals = _drain_concat(bus, SIGNALS)
        intents = _drain_concat(bus, INTENTS)
        # market_data alimenta o regime por barras reais. Pegamos o snapshot mais
        # recente do ciclo (o IngestionAgent publica um dict por step).
        md_msgs = bus.drain(MARKET_DATA)
        market_data = md_msgs[-1].payload if md_msgs else None
        if not intents:
            self.last_approved = []
            return
        result = self._intel.process(intents, signals, market_data=market_data)
        self.last_approved = result.allowed
        bus.publish(APPROVED, result.allowed)


class ExecutorAgent(BaseAgent):
    """Executa intents aprovadas e publica fills (p/ o FeedbackAgent)."""

    def __init__(self, executor: Executor) -> None:
        super().__init__("executor")
        self._executor = executor
        self.last_results: list = []

    def step(self, bus: MessageBus) -> None:
        approved = _drain_concat(bus, APPROVED)
        self.last_results = self._executor.execute_many(approved) if approved else []
        for r in self.last_results:
            if r.filled_qty and r.filled_qty > 0 and r.filled_avg_price is not None:
                bus.publish(
                    FILLS,
                    {
                        "symbol": r.symbol,
                        "side": r.side.value,
                        "filled_qty": r.filled_qty,
                        "fill_price": r.filled_avg_price,
                        "client_order_id": r.client_order_id,
                    },
                )

    @staticmethod
    def _is_option(intent) -> bool:
        return isinstance(intent, OptionOrderIntent)
