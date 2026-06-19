"""Plug VIVO da estrategia de beta — ADITIVO e atras de FLAG (desligada por padrao).

Este modulo concentra TODO o acoplamento do rebalanceador de beta ao caminho
vivo (paper Alpaca), para que as edicoes em main.py/agents/monitor.py sejam
MINIMAS e nao quebrem o sistema Alpaca existente. Outro processo/usuario pode
estar mexendo nesses arquivos: aqui o efeito e estritamente aditivo.

DUAS FUNCOES, ambas no-op enquanto a flag estiver desligada:
  1. run_beta_rebalance_cycle(...) — calcula o plano de rebalance mensal (sem
     look-ahead) e o EXECUTA via o Executor existente (kill switch + idempotencia
     + reconciliacao continuam valendo). Marca o mes como rebalanceado SO apos a
     execucao confirmar.
  2. schedule_eod_capture(...) — agenda um job APScheduler pos-fechamento que
     chama reporting.capture.capture_eod (snapshot diario de NAV idempotente),
     conforme o docstring de capture.py (Opcao A).

A FLAG:
  Ligada por env var BETA_LIVE_ENABLED (truthy: 1/true/yes/on). DESLIGADA por
  padrao. Enquanto desligada:
    - build_app NAO adiciona a estrategia de beta nem o job de captura;
    - importar este modulo NAO baixa dados, NAO toca o broker, NAO agenda nada.
  COMO LIGAR (so depois da auditoria do Coder):
    export BETA_LIVE_ENABLED=1     # no .env ou no ambiente do processo
  e reiniciar o processo (main.py). Ainda assim, o kill switch e os guards
  continuam podendo barrar ordens.

PARIDADE: o plano usa strategies.beta_rebalancer.target_weights_for_date, a
MESMA funcao provada igual ao backtest no teste de paridade.
"""

from __future__ import annotations

import logging

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("strategies.beta_live")

# Flag mestre do caminho vivo do beta. DESLIGADA por padrao.
_FLAG_ENV = "BETA_LIVE_ENABLED"


class BetaLiveSettings(BaseSettings):
    """Flag de ARME do caminho vivo do beta, lida pela MESMA mecanica do resto da
    config (config/settings.py, config/risk.py): uma BaseSettings do pydantic-
    settings com `env_file='.env'`.

    POR QUE: ler `os.environ` cru tornava `BETA_LIVE_ENABLED=1` no `.env` INERTE —
    o `.env` nao entra em os.environ sozinho, entao o daemon bootava no caminho
    LEGADO em silencio mesmo com a flag no `.env`. Lendo via pydantic-settings, a
    flag e armada TANTO pelo `.env` QUANTO por env var (env var tem PRECEDENCIA),
    exatamente como as demais settings do projeto.

    DESLIGADA por padrao: sem `.env` e sem env var, `beta_live_enabled` = False e
    o caminho vivo fica inteiramente inerte (comportamento legado intacto)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OFF por padrao. Alias BETA_LIVE_ENABLED (igual ao padrao das outras settings).
    # pydantic-settings interpreta truthy/falsy de bool (1/0, true/false, yes/no,
    # on/off), cobrindo os mesmos valores que o parser cru aceitava.
    beta_live_enabled: bool = Field(False, alias=_FLAG_ENV)


# MF-5: JANELA PERTO DO FECHAMENTO para disparar o rebalance. O backtest calcula
# retorno/turnover a preco de FECHAMENTO; o vivo manda ordens MARKET, que enchem
# ao preco do tick. Se disparassemos no 1o tick do dia (manha), o fill seria a um
# preco intraday arbitrario, divergindo dos pesos de fechamento. Disparando so nos
# ultimos minutos do pregao, o fill MARKET fica PROXIMO do close de r+1 — o preco
# que define os pesos do backtest — minimizando o erro de execucao.
#
# PREMISSA DE FILL (documentada): em paper, a ordem e MARKET com TIF=DAY (cripto:
# GTC) e enche ao melhor preco disponivel no momento do disparo, dentro desta
# janela perto do close. Assume-se que esse preco ~ close de r+1. O residual
# (slippage + horario != close exato) e um erro de execucao de 2a ordem; em
# frequencia mensal e pequeno, mas EXISTE e separa o NAV realizado da paridade
# teorica de fechamento. Para replicar o close ao pe da letra seria preciso
# MarketOnClose/LimitOnClose (nao usado aqui).
_NEAR_CLOSE_WINDOW_MIN = 20


def beta_live_enabled() -> bool:
    """True somente se a flag BETA_LIVE_ENABLED estiver explicitamente ligada.

    Le a flag pela MESMA mecanica do resto da config (pydantic-settings), entao
    ela e armada por env var INLINE *ou* pelo `.env` (env var tem precedencia).
    Antes lia os.environ cru — e a flag no `.env` ficava inerte (bug de robustez
    pego no go-live: o daemon bootava no caminho legado em silencio).

    Padrao DESLIGADO: sem env var e sem `.env`, retorna False e o caminho vivo do
    beta fica inteiramente inerte (comportamento legado intacto). Instanciado a
    cada chamada (sem cache): mudancas de env/.env e overrides de teste valem na
    hora, sem estado preso."""
    return BetaLiveSettings().beta_live_enabled  # type: ignore[call-arg]


def _is_near_close(broker, *, window_min: int = _NEAR_CLOSE_WINDOW_MIN) -> bool:
    """True se estamos dentro da janela de `window_min` minutos antes do fechamento
    do pregao (MF-5), usando o relogio da CORRETORA (sem skew do relogio local).

    Politica (fail-safe, nao trava o rebalance p/ sempre):
      - mercado FECHADO: NAO e janela de close (espera o pregao). Isso impede
        disparar um rebalance num fim de semana/feriado (cripto 24/7) longe do
        close de acoes — o rebalance mensal deve casar com o close de r+1.
      - mercado ABERTO e faltam <= window_min p/ o next_close: e a janela. True.
      - mercado ABERTO e faltam MAIS que window_min: ainda nao. False.
      - relogio sem next_close/timestamp (indisponivel): FAIL-OPEN (True) — melhor
        rebalancear no dia devido do que pular o mes inteiro por falta de relogio.
    """
    try:
        clock = broker.get_clock()
    except Exception:  # noqa: BLE001 - sem relogio legivel => fail-open (nao trava o mes)
        logger.warning("beta_live: relogio indisponivel; assumindo janela de close.")
        return True
    if not getattr(clock, "is_open", False):
        return False
    next_close = getattr(clock, "next_close", None)
    now = getattr(clock, "timestamp", None)
    if next_close is None or now is None:
        logger.warning("beta_live: clock sem next_close/timestamp; fail-open (janela de close).")
        return True
    minutes_to_close = (next_close - now).total_seconds() / 60.0
    return 0 <= minutes_to_close <= window_min


def _asof_yesterday(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Ultimo fechamento DISPONIVEL no painel (= 'ontem', sem look-ahead).

    O painel (load_panel) so contem barras ja fechadas; a ultima e o fechamento
    mais recente. Usar a ultima linha como asof garante que a decisao de hoje so
    usa dados <= o ultimo fechamento."""
    if panel is None or panel.empty:
        return None
    return panel.index[-1]


def run_beta_rebalance_cycle(
    broker,
    executor,
    *,
    state=None,
    guard=None,
    panel: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    force: bool = False,
) -> dict:
    """Roda UM ciclo de rebalance mensal do beta e executa as ordens de diferenca.

    NO-OP se a flag estiver desligada (a menos que force=True, p/ teste/manual).
    Retorna um resumo auditavel (dict). Etapas:
      1. carrega o painel (mesma fonte do backtest) se nao injetado;
      2. asof = ultimo fechamento (ontem) — sem look-ahead;
      3. plano de rebalance SE o gatilho mensal disparar (senao no-op);
      4. executa as intencoes pelo Executor (kill switch/idempotencia valem);
      5. marca o mes como rebalanceado SO apos a execucao.
    """
    if not force and not beta_live_enabled():
        logger.debug("beta_live desligado (flag %s off) — rebalance pulado.", _FLAG_ENV)
        return {"enabled": False, "executed": 0, "intents": 0}

    # Import tardio: nao baixa dados nem carrega o nucleo de pesos no import.
    from strategies.beta_rebalancer import BetaRebalancer, load_production_panel, mark_rebalanced

    if panel is None:
        panel, _classes = load_production_panel()
    if panel is None or panel.empty:
        logger.warning("beta_live: painel vazio (dados pendentes?) — rebalance pulado.")
        return {"enabled": True, "executed": 0, "intents": 0, "note": "painel vazio"}

    if asof is None:
        asof = _asof_yesterday(panel)
    if asof is None:
        return {"enabled": True, "executed": 0, "intents": 0, "note": "sem asof"}

    rebalancer = BetaRebalancer(broker, state=state, guard=guard)
    plan = rebalancer.plan_if_due(panel, asof, force=force)
    if plan is None:
        logger.info("beta_live: rebalance mensal nao e devido (asof=%s).", pd.Timestamp(asof).date())
        return {"enabled": True, "executed": 0, "intents": 0, "due": False}

    # MF-5: o mes e devido, mas so EXECUTA perto do fechamento, p/ o fill MARKET
    # ficar proximo do close de r+1 (os pesos do backtest sao de fechamento). Fora
    # da janela, NAO marca o mes — volta a tentar num tick mais perto do close.
    # `force` (teste/manual) ignora a janela.
    if not force and not _is_near_close(broker):
        logger.info(
            "beta_live: rebalance devido (asof=%s) mas fora da janela de fechamento "
            "— adiado p/ perto do close (MF-5).",
            plan.asof,
        )
        return {
            "enabled": True,
            "due": True,
            "deferred_to_close": True,
            "executed": 0,
            "intents": len(plan.intents),
        }

    logger.info(
        "beta_live: rebalance devido (asof=%s) — %d intencao(oes), %d na banda, %d bloqueada(s).",
        plan.asof, len(plan.intents), len(plan.skipped_in_band), len(plan.blocked),
    )

    results = executor.execute_many(plan.intents) if plan.intents else []

    # Marca o mes como rebalanceado SO apos a execucao (se cair antes, volta a
    # ser devido). Mesmo sem intencoes (carteira ja no alvo dentro da banda), o
    # mes esta resolvido — marca p/ nao reavaliar a cada tick.
    mark_rebalanced(state, asof)

    return {
        "enabled": True,
        "due": True,
        "asof": plan.asof,
        "intents": len(plan.intents),
        "executed": len(results),
        "in_band": len(plan.skipped_in_band),
        "blocked": plan.blocked,
        "equity": str(plan.equity),
    }


def schedule_eod_capture(scheduler, broker, *, db_conn=None, state=None, clock=None) -> bool:
    """Agenda o snapshot EOD de NAV pos-fechamento (Opcao A do capture.py).

    NO-OP se a flag estiver desligada (retorna False). Quando ligada, agenda um
    job diario logo apos o next_close do mercado. O UNIQUE em `date` + UPSERT do
    nav_history garantem 1 linha EOD/dia mesmo se o job rodar 2x.

    Mantemos best-effort: qualquer falha ao agendar e logada e NAO derruba o
    sistema Alpaca existente."""
    if not beta_live_enabled():
        logger.debug("beta_live desligado — captura EOD nao agendada.")
        return False
    try:
        from apscheduler.triggers.cron import CronTrigger

        from reporting.capture import capture_eod
        from reporting.nav_repo import NavHistoryRepo

        repo = NavHistoryRepo(connection=db_conn) if db_conn is not None else None

        def _capture_tick() -> None:
            try:
                regime = _spy_regime(broker)
                capture_eod(broker, repo=repo, state=state, regime=regime)
            except Exception:  # noqa: BLE001 - captura e best-effort; nao derruba o loop
                logger.exception("beta_live: falha na captura EOD (ignorada).")

        # Horario do job: ~20 min apos o fechamento padrao (16:00 ET = 20:00/21:00
        # UTC conforme DST). Como o scheduler roda em UTC e o pregao fecha 20:00
        # UTC (horario de verao) / 21:00 UTC (inverno), agendamos 21:20 UTC, que
        # cai apos o fechamento o ano todo. O capture_eod e idempotente por dia.
        scheduler.add_job(
            _capture_tick,
            CronTrigger(hour=21, minute=20, timezone="UTC"),
            id="beta_eod_capture",
            replace_existing=True,
        )
        logger.info("beta_live: captura EOD agendada (cron 21:20 UTC, pos-fechamento).")
        return True
    except Exception:  # noqa: BLE001 - agendamento e aditivo; nunca quebra o boot
        logger.exception("beta_live: falha ao agendar captura EOD (ignorada).")
        return False


def refresh_production_panel() -> dict:
    """Re-baixa/ESTENDE o cache de precos do painel (NOVO-2), p/ o 'asof' avancar.

    NO-OP se a flag estiver desligada (retorna {refreshed: False}). Quando ligada,
    chama load_production_panel(force=True) — a MESMA fonte do backtest (yfinance
    p/ os ETFs/cripto) — de modo que data/beta_cache/*.csv ganha os pregoes novos.
    Sem isto o painel le um cache estatico, o asof CONGELA no ultimo close cacheado
    e o gatilho mensal para de disparar (ou negocia preco velho) — NOVO-2.

    PARIDADE: usa o MESMO load_panel do backtest; o teste de paridade peso continua
    valendo sobre o cache atualizado (a IDENTIDADE de calculo nao muda — so a data
    do ultimo close avanca).

    FAIL-SAFE: se a rede falhar, NAO levanta — loga e mantem o ultimo cache em
    disco (load_close so reescreve o csv quando o download volta dados; uma falha
    de rede deixa o cache antigo intacto). Assim uma indisponibilidade de rede
    degrada para 'usa o ultimo painel', nunca derruba o sistema."""
    if not beta_live_enabled():
        logger.debug("beta_live desligado — refresh de painel pulado.")
        return {"refreshed": False, "enabled": False}
    try:
        from strategies.beta_rebalancer import load_production_panel

        panel, _classes = load_production_panel(force=True)
        if panel is None or panel.empty:
            logger.warning("beta_live: refresh de painel devolveu vazio — cache mantido.")
            return {"refreshed": False, "enabled": True, "note": "painel vazio"}
        last = pd.Timestamp(panel.index[-1])
        logger.info(
            "beta_live: painel atualizado — %d barras, ultimo close %s (asof avanca).",
            len(panel), last.date(),
        )
        return {"refreshed": True, "enabled": True, "rows": int(len(panel)), "asof": last}
    except Exception:  # noqa: BLE001 - rede instavel => mantem o ultimo cache, nao quebra
        logger.exception("beta_live: refresh de painel falhou (rede?); usando ultimo cache.")
        return {"refreshed": False, "enabled": True, "note": "falha de rede (fail-safe)"}


def schedule_panel_refresh(scheduler, *, refresher=None) -> bool:
    """Agenda o REFRESH diario do painel de precos (NOVO-2), atras da MESMA flag.

    NO-OP se a flag estiver desligada (retorna False). Quando ligada, agenda um
    job APScheduler que roda refresh_production_panel() uma vez por dia.

    CADENCIA (documentada): cron DIARIO as 11:00 UTC (~06:00-07:00 ET, PRE-mercado
    dos EUA). Nesse horario o fechamento do pregao ANTERIOR ja esta liquidado no
    yfinance, entao o painel ganha a barra de ontem ANTES da janela de close de
    hoje (onde o rebalance dispara, ~19:40-20:40 UTC). Assim o asof do rebalance
    de hoje = o close de ONTEM (a base do backtest, .shift(1)), e a deteccao de
    virada de mes ve os primeiros pregoes do mes novo a tempo. Diario (nao so em
    virada de mes) mantem o cache sempre fresco e barato; load_panel e idempotente.

    FAIL-SAFE: refresh_production_panel ja trata falha de rede internamente (mantem
    o ultimo cache, nao levanta). O agendamento e best-effort: qualquer falha ao
    agendar e logada e NAO derruba o sistema Alpaca existente."""
    if not beta_live_enabled():
        logger.debug("beta_live desligado — refresh de painel nao agendado.")
        return False
    try:
        from apscheduler.triggers.cron import CronTrigger

        do_refresh = refresher or refresh_production_panel

        def _refresh_tick() -> None:
            try:
                do_refresh()
            except Exception:  # noqa: BLE001 - refresh e best-effort; nao derruba o loop
                logger.exception("beta_live: tick de refresh de painel falhou (ignorado).")

        scheduler.add_job(
            _refresh_tick,
            CronTrigger(hour=11, minute=0, timezone="UTC"),
            id="beta_panel_refresh",
            replace_existing=True,
        )
        logger.info("beta_live: refresh de painel agendado (cron 11:00 UTC, pre-mercado).")
        return True
    except Exception:  # noqa: BLE001 - agendamento e aditivo; nunca quebra o boot
        logger.exception("beta_live: falha ao agendar refresh de painel (ignorada).")
        return False


def _spy_regime(broker) -> str:
    """Regime agregado do dia via SPY (best-effort; 'unknown' se indisponivel)."""
    try:
        from feedback.regime import classify_regime

        bars = broker.get_bars("SPY", limit=80, timeframe="1Day")
        closes = [float(b) for b in bars]
        if len(closes) >= 35:
            return str(getattr(classify_regime(closes), "value", classify_regime(closes)))
    except Exception:  # noqa: BLE001
        logger.debug("beta_live: regime do SPY indisponivel; usando 'unknown'.")
    return "unknown"
