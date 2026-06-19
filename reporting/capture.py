"""Captura EOD do NAV — o PONTO DE PLUG do track record no caminho vivo.

AGNOSTICO de estrategia: depende apenas da interface BrokerClient (get_account/
get_positions/get_last_price). Serve para o beta disciplinado e para qualquer
estrategia — comeca a contar para o track record ASSIM QUE rodar em paper,
inclusive no periodo de "conta parada" (baseline), o que e exatamente o que
torna o historico a prova de cherry-picking.

O QUE FAZ, EM UM TICK:
  1. Le equity/cash do broker (mark-to-market = a fonte de verdade da conta).
  2. Soma o valor de mercado das posicoes (current_price; fallback get_last_price)
     -> long_market_value, gross/net exposure (sobre o equity).
  3. Snapshota o benchmark vivo NO MESMO DIA: preco de SPY (total-return) e
     avanca o NAV sintetico 60/40 (estado persistido em `state`, sem look-ahead).
  4. Grava UM snapshot oficial (source="eod") em nav_history (append-only + hash).

IDEMPOTENTE POR DIA: rodar 2x no mesmo dia faz UPSERT do snapshot do dia (nao
duplica) e nunca toca dias anteriores. Sobrevive a restart (o estado do 60/40
fica em `state`).

ONDE PLUGAR (quando a estrategia for ao paper) — ver docstring de capture_eod.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from datetime import date

from broker.base import BrokerClient
from reporting import benchmarks
from reporting.nav_repo import DEFAULT_DB_PATH, NavHistoryRepo

logger = logging.getLogger("reporting.capture")

# Chave do estado incremental do 60/40 na tabela `state`.
_SIXTY_FORTY_STATE_KEY = "track_record:sixty_forty_state"

# Tickers cujo preco precisamos amostrar para o benchmark (SPY + componentes 60/40).
_BENCH_TICKERS = sorted({benchmarks.SPY, benchmarks.IEF})


def _market_value(position, broker: BrokerClient) -> Decimal:
    """Valor de mercado de uma posicao (current_price; fallback get_last_price)."""
    price = getattr(position, "current_price", None)
    if price is None:
        try:
            price = broker.get_last_price(position.symbol)
        except Exception:  # noqa: BLE001 - preco indisponivel -> usa entrada como ultimo recurso
            logger.warning("Preco indisponivel para %s; usando avg_entry_price.", position.symbol)
            price = position.avg_entry_price
    return Decimal(str(position.qty)) * Decimal(str(price))


def _bench_prices(broker: BrokerClient) -> dict[str, Decimal]:
    """Precos do dia para os tickers de benchmark (best-effort; ausencia -> faltante)."""
    prices: dict[str, Decimal] = {}
    for ticker in _BENCH_TICKERS:
        try:
            prices[ticker] = Decimal(str(broker.get_last_price(ticker)))
        except Exception:  # noqa: BLE001 - benchmark e best-effort, nao deve derrubar a captura
            logger.warning("Preco de benchmark indisponivel: %s", ticker)
    return prices


def _load_6040_state(state) -> benchmarks.SixtyFortyState | None:
    if state is None:
        return None
    raw = state.get(_SIXTY_FORTY_STATE_KEY)
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return benchmarks.SixtyFortyState(
            nav=Decimal(d["nav"]),
            units={k: Decimal(v) for k, v in d["units"].items()},
            last_month=d.get("last_month"),
            last_prices={k: Decimal(v) for k, v in d.get("last_prices", {}).items()},
        )
    except (ValueError, KeyError, TypeError):
        logger.warning("Estado do 60/40 corrompido em `state`; reinicializando.")
        return None


def _save_6040_state(state, st: benchmarks.SixtyFortyState) -> None:
    if state is None:
        return
    state.set(
        _SIXTY_FORTY_STATE_KEY,
        json.dumps(
            {
                "nav": str(st.nav),
                "units": {k: str(v) for k, v in st.units.items()},
                "last_month": st.last_month,
                "last_prices": {k: str(v) for k, v in st.last_prices.items()},
            }
        ),
    )


def capture_eod(
    broker: BrokerClient,
    *,
    repo: NavHistoryRepo | None = None,
    db_path=DEFAULT_DB_PATH,
    state=None,
    regime: str = "unknown",
    day: str | date | None = None,
    source: str = "eod",
) -> dict:
    """Grava UM snapshot oficial de NAV do dia. Retorna a linha gravada.

    Parametros:
      broker  : qualquer BrokerClient (paper). Fonte de verdade do equity.
      repo    : NavHistoryRepo ja aberto (compartilha conexao); senao abre em db_path.
      state   : StateRepository p/ persistir o NAV incremental do 60/40 (opcional,
                mas recomendado em producao p/ sobreviver a restart).
      regime  : regime agregado do dia (ex.: feedback.regime.classify_regime do SPY).
      day     : data do snapshot (default: hoje UTC). source="eod" e o oficial.

    ONDE PLUGAR QUANDO A ESTRATEGIA FOR AO PAPER (caminho vivo):
      Opcao A (recomendada) — job APScheduler dedicado pos-fechamento em
        agents/monitor.py: agende um tick diario logo apos broker.get_clock()
        .next_close e chame capture_eod(broker, repo=..., state=...). O UNIQUE em
        `date` + o UPSERT garantem 1 linha EOD/dia mesmo se o job rodar 2x.
      Opcao B — um passo final no pipeline do bus
        (orchestration/bus_orchestrator.py) que, ao detectar a transicao de
        mercado-aberto -> fechado, chama capture_eod uma vez.
      Em build_app (main.py) ja existem `broker` e `StateRepository(db)`; basta
        instanciar NavHistoryRepo(connection=db.conn) e passar ambos aqui.
    """
    own = repo is None
    r = repo or NavHistoryRepo(db_path=db_path)
    try:
        account = broker.get_account()
        equity = Decimal(str(account.equity))
        cash = Decimal(str(account.cash))

        positions = broker.get_positions()
        long_mv = Decimal("0")
        net_mv = Decimal("0")
        for p in positions:
            mv = _market_value(p, broker)
            if mv > 0:
                long_mv += mv
            net_mv += mv  # short (qty<0) entra negativo no net

        gross_mv = sum((abs(_market_value(p, broker)) for p in positions), Decimal("0"))
        gross_exposure = (gross_mv / equity) if equity > 0 else Decimal("0")
        net_exposure = (net_mv / equity) if equity > 0 else Decimal("0")

        # Benchmark vivo no MESMO dia.
        day_str = day if isinstance(day, str) else (day.isoformat() if day else None)
        prices = _bench_prices(broker)
        bench_spy = prices.get(benchmarks.SPY)

        prev_state = _load_6040_state(state)
        bench_6040 = None
        if prices:
            new_state = benchmarks.step(prev_state, day_str or _today(), prices)
            bench_6040 = new_state.nav
            _save_6040_state(state, new_state)

        row = r.record_snapshot(
            day=day,
            equity=equity,
            cash=cash,
            long_market_value=long_mv,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            regime=regime,
            bench_spy=bench_spy,
            bench_6040=bench_6040,
            source=source,
        )
        logger.info(
            "NAV snapshot %s: equity=%s gross=%.2f%% (source=%s)",
            row["date"], row["equity"], float(gross_exposure) * 100, source,
        )
        return row
    finally:
        if own:
            r.close()


def _today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()
