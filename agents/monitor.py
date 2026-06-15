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
        if not self._clock.is_open():
            logger.info("Mercado fechado — ciclo do Monitor pulado.")
            return None
        logger.info("Mercado aberto — iniciando ciclo de orquestracao.")
        result = self._orchestrator.run_cycle()
        logger.info(
            "Ciclo concluido: %d intencao(oes), %d ordem(ns) executada(s).",
            result.intents_count, result.executed_count,
        )
        return result

    def start(self) -> None:
        """Inicia o agendamento bloqueante (uso em producao via main.py)."""
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()
        scheduler.add_job(
            self.tick,
            "interval",
            minutes=self._interval_minutes,
            next_run_time=None,
        )
        logger.info(
            "Monitor agendado a cada %d min. Aguardando pregao...",
            self._interval_minutes,
        )
        try:
            self.tick()  # executa um ciclo imediato ao iniciar
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Monitor interrompido.")
            scheduler.shutdown(wait=False)
