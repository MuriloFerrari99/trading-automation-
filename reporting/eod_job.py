"""Agendamento do EOD: captura o NAV e regenera o tearsheet QuantStats 1x/dia.

Liga o tearsheet ao CICLO sem reescrever o scheduler existente — basta uma
chamada para registrar o job no APScheduler que o sistema ja usa. Pos-fechamento,
cada dia: grava o snapshot oficial de NAV (idempotente) e atualiza o relatorio.

COMO PLUGAR (em main.build_app, onde ja existem `scheduler`, `broker`, `db`):
    from data.state_repo import StateRepository
    from reporting.nav_repo import NavHistoryRepo
    from reporting.eod_job import register_eod_report_job
    register_eod_report_job(
        scheduler, broker,
        repo=NavHistoryRepo(connection=db.conn),
        state=StateRepository(db),
        hour=21, minute=10,            # ~pos-fechamento US (UTC); ajuste ao seu TZ
    )

Rodar uma vez na mao (sem agendar):
    uv run python -m reporting.eod_job --once   (requer broker configurado; ver build_app)
"""

from __future__ import annotations

import logging

from apscheduler.triggers.cron import CronTrigger

from reporting.nav_repo import DEFAULT_DB_PATH, NavHistoryRepo
from reporting.quantstats_report import DEFAULT_OUTPUT, capture_and_report

logger = logging.getLogger("reporting.eod_job")

DEFAULT_JOB_ID = "eod_nav_tearsheet"


def register_eod_report_job(
    scheduler,
    broker,
    *,
    repo: NavHistoryRepo | None = None,
    state=None,
    db_path=DEFAULT_DB_PATH,
    output=DEFAULT_OUTPUT,
    hour: int = 21,
    minute: int = 10,
    job_id: str = DEFAULT_JOB_ID,
):
    """Registra (ou substitui) o job diario de captura de NAV + tearsheet.

    Retorna o Job do APScheduler. Erros dentro do tick sao logados, nunca
    derrubam o scheduler (o ciclo de trading nao para por causa do relatorio).
    """

    def _tick():
        try:
            result = capture_and_report(
                broker,
                repo=repo,
                state=state,
                db_path=db_path,
                output=output,
            )
            logger.info(
                "EOD: NAV gravado (%s) + tearsheet %s",
                (result.get("nav_row") or {}).get("date"),
                result.get("report"),
            )
            return result
        except Exception:  # noqa: BLE001
            logger.exception("Job EOD (NAV+tearsheet) falhou neste tick; segue o loop.")
            return None

    return scheduler.add_job(
        _tick,
        CronTrigger(hour=hour, minute=minute),
        id=job_id,
        replace_existing=True,
    )
