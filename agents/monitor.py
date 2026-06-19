"""Agente Monitor.

Roda em intervalos (5-15 min) APENAS durante o pregao. A cada tick:
- Verifica o relogio de mercado; se fechado, pula o ciclo.
- Dispara um ciclo de orquestracao (Planner -> Executor) que ajusta os
  trailing stops e envia ordens quando disparadas.

`tick()` e isolado e testavel; `start()` agenda os ticks via APScheduler.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

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
        run_when_closed: bool = False,
        on_schedule_start: Callable[[object], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._clock = clock
        self._interval_minutes = interval_minutes
        # 24/7: quando ha ativos de cripto na watchlist, o ciclo roda mesmo com o
        # pregao de acoes fechado (as estrategias gateiam os ativos de acao).
        self._run_when_closed = run_when_closed
        # Hook ADITIVO e OPCIONAL: chamado uma vez com o scheduler ja criado,
        # antes do start(). Permite a um caller (main.py) agendar jobs extras
        # (ex.: captura EOD do beta) SEM acoplar o Monitor a essas dependencias.
        # Default None => comportamento legado inalterado.
        self._on_schedule_start = on_schedule_start

    def tick(self) -> CycleResult | None:
        """Executa um ciclo se o mercado estiver aberto (ou 24/7 p/ cripto)."""
        clock = self._clock.info()
        if not clock.is_open and not self._run_when_closed:
            nxt = clock.next_open
            logger.info(
                "Mercado fechado — ciclo pulado.%s",
                f" Proxima abertura: {nxt}." if nxt else "",
            )
            return None
        if clock.is_open:
            logger.info("Mercado aberto — iniciando ciclo de orquestracao.")
        else:
            logger.info("Pregao fechado, mas ha cripto (24/7) — rodando ciclo.")
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
        # Jobs extras opcionais (aditivo). Best-effort: nunca derruba o Monitor.
        if self._on_schedule_start is not None:
            try:
                self._on_schedule_start(scheduler)
            except Exception:  # noqa: BLE001 - hook aditivo nao deve quebrar o loop
                logger.exception("Hook on_schedule_start falhou (ignorado).")
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
