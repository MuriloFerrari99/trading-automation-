"""Testes do relogio de mercado e do scheduler (Fase D)."""

from __future__ import annotations

from datetime import datetime, timezone

from broker.fake_broker import FakeBroker
from core.market_clock import MarketClock
from scheduling.scheduler import add_interval_job, make_scheduler


def test_market_clock_info_open_closed():
    b = FakeBroker(market_open=True)
    clock = MarketClock(b)
    assert clock.is_open() is True
    assert clock.info().is_open is True

    b.set_market_open(False)
    nxt = datetime(2026, 6, 16, 13, 30, tzinfo=timezone.utc)
    b.set_clock(next_open=nxt)
    info = clock.info()
    assert info.is_open is False
    assert info.next_open == nxt
    assert clock.next_open() == nxt


def test_scheduler_is_utc():
    sched = make_scheduler()
    assert str(sched.timezone) == "UTC"


def test_add_interval_job_registers():
    sched = make_scheduler()
    calls = []
    add_interval_job(sched, lambda: calls.append(1), minutes=10, job_id="monitor")
    job = sched.get_job("monitor")
    assert job is not None
    # nao iniciamos o scheduler (seria bloqueante); so validamos o registro.
    assert job.id == "monitor"


def test_monitor_skips_and_logs_next_open_when_closed():
    from agents.monitor import Monitor
    from orchestration.base import AgentOrchestrator, CycleResult

    class _Orch(AgentOrchestrator):
        def __init__(self):
            self.ran = False

        def run_cycle(self) -> CycleResult:
            self.ran = True
            return CycleResult(intents=[], results=[])

    b = FakeBroker(market_open=False)
    b.set_clock(next_open=datetime(2026, 6, 16, 13, 30, tzinfo=timezone.utc))
    orch = _Orch()
    monitor = Monitor(orch, MarketClock(b), interval_minutes=10)
    assert monitor.tick() is None
    assert orch.ran is False  # nao rodou com mercado fechado
