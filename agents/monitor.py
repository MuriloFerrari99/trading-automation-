"""Agente Monitor.

Roda em intervalos (5-15 min) APENAS durante o pregao. A cada tick:
- Verifica o relogio de mercado; se fechado, pula o ciclo.
- Dispara um ciclo de orquestracao (Planner -> Executor) que ajusta os
  trailing stops e envia ordens quando disparadas.

`tick()` e isolado e testavel; `start()` agenda os ticks via APScheduler.
"""

from __future__ import annotations

import logging

from core.market_clock import MarketClock
from orchestration.base import AgentOrchestrator, CycleResult

logger = logging.getLogger("agent.monitor")


class Monitor:
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        clock: MarketClock,
        *,
        interval_minutes: int = 10,
    ) -> None:
        self._orchestrator = orchestrator
        self._clock = clock
        self._interval_minutes = interval_minutes

    def tick(self) -> CycleResult | None:
        """Executa um ciclo se o mercado estiver aberto; senao pula."""
        clock = self._clock.info()
        if not clock.is_open:
            nxt = clock.next_open
            logger.info(
                "Mercado fechado — ciclo pulado.%s",
                f" Proxima abertura: {nxt}." if nxt else "",
            )
            return None
        logger.info("Mercado aberto — iniciando ciclo de orquestracao.")
        result = self._orchestrator.run_cycle()
        logger.info(
            "Ciclo concluido: %d sinal(is) sugerido(s), %d intencao(oes), "
            "%d ordem(ns) executada(s).",
            result.signals_count, result.intents_count, result.executed_count,
        )
        return result

    def start(self) -> None:
        """Inicia o agendamento bloqueante (UTC; DST-safe) via APScheduler."""
        from scheduling.scheduler import add_interval_job, make_scheduler

        scheduler = make_scheduler()  # timezone=UTC
        add_interval_job(
            scheduler, self.tick, minutes=self._interval_minutes, job_id="monitor"
        )
        logger.info(
            "Monitor agendado a cada %d min (UTC). Aguardando pregao...",
            self._interval_minutes,
        )
        try:
            self.tick()  # executa um ciclo imediato ao iniciar
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Monitor interrompido.")
            scheduler.shutdown(wait=False)
