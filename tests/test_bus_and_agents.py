"""Testes da fundacao agentica: MessageBus, BaseAgent, IngestionAgent, get_bars."""

from __future__ import annotations

from decimal import Decimal

from agents.base import BaseAgent
from agents.ingestion import MARKET_DATA, IngestionAgent
from broker.fake_broker import FakeBroker
from orchestration.bus import InMemoryBus, Message


# --- MessageBus -------------------------------------------------------------
def test_bus_publish_consume_fifo():
    bus = InMemoryBus()
    bus.publish("intents", {"a": 1})
    bus.publish("intents", {"a": 2})
    assert bus.pending("intents") == 2
    m1 = bus.consume("intents")
    m2 = bus.consume("intents")
    assert m1.payload == {"a": 1} and m2.payload == {"a": 2}
    assert bus.consume("intents") is None  # vazio => None (nao bloqueia)


def test_bus_drain():
    bus = InMemoryBus()
    for i in range(3):
        bus.publish("fills", i)
    drained = bus.drain("fills")
    assert [m.payload for m in drained] == [0, 1, 2]
    assert bus.pending("fills") == 0


def test_bus_meta_preserved():
    bus = InMemoryBus()
    bus.publish("alerts", "x", actor="monitor")
    m = bus.consume("alerts")
    assert m.meta == {"actor": "monitor"}


# --- BaseAgent --------------------------------------------------------------
class _EchoAgent(BaseAgent):
    def __init__(self):
        super().__init__("echo", inbox="in")
        self.seen = []

    def handle(self, msg: Message, bus):
        self.seen.append(msg.payload)
        bus.publish("out", msg.payload * 2)


def test_base_agent_consumes_all_and_publishes():
    bus = InMemoryBus()
    bus.publish("in", 5)
    bus.publish("in", 7)
    agent = _EchoAgent()
    agent.step(bus)
    assert agent.seen == [5, 7]
    assert [m.payload for m in bus.drain("out")] == [10, 14]


class _BoomAgent(BaseAgent):
    def __init__(self):
        super().__init__("boom", inbox="in")

    def handle(self, msg, bus):
        raise RuntimeError("falha proposital")


def test_base_agent_isolates_handler_errors():
    bus = InMemoryBus()
    bus.publish("in", 1)
    bus.publish("in", 2)
    _BoomAgent().step(bus)  # nao deve propagar excecao
    assert bus.pending("in") == 0  # ambas consumidas apesar do erro


# --- get_bars + IngestionAgent ---------------------------------------------
def test_fake_broker_bars():
    b = FakeBroker()
    b.set_bars("AAPL", [100, 101, 102, 103])
    assert b.get_bars("AAPL", limit=2) == [Decimal("102"), Decimal("103")]
    assert b.get_bars("MSFT") == []  # sem barras => vazio


def test_ingestion_publishes_market_data():
    b = FakeBroker()
    b.set_bars("AAPL", [100, 101, 102])
    b.set_bars("MSFT", [200, 199])
    bus = InMemoryBus()
    IngestionAgent(b, ["AAPL", "MSFT"], bars_limit=10).step(bus)

    msg = bus.consume(MARKET_DATA)
    assert msg is not None
    assert msg.payload["AAPL"] == [Decimal("100"), Decimal("101"), Decimal("102")]
    assert msg.payload["MSFT"] == [Decimal("200"), Decimal("199")]
