"""Agendamento com APScheduler (doc 05 §6).

Regras para trading US:
- O scheduler roda internamente em UTC (DST-safe; APScheduler recomenda evitar
  horarios na virada do horario de verao).
- A checagem real de "mercado aberto" usa o clock da corretora (que ja resolve
  feriados e early-close via calendar) — nunca um horario fixo hardcoded.
- IntervalTrigger de 5-15 min; o tick faz no-op se o mercado estiver fechado.
"""

from __future__ import annotations

from typing import Callable

DEFAULT_TIMEZONE = "UTC"


def make_scheduler(timezone: str = DEFAULT_TIMEZONE):
    """Cria um BlockingScheduler em UTC (import tardio do APScheduler)."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    return BlockingScheduler(timezone=timezone)


def add_interval_job(
    scheduler,
    func: Callable[[], object],
    *,
    minutes: int,
    job_id: str,
):
    """Agenda `func` a cada `minutes` minutos (alinhado ao relogio UTC)."""
    from apscheduler.triggers.interval import IntervalTrigger

    return scheduler.add_job(func, IntervalTrigger(minutes=minutes), id=job_id)
