"""Testes do matching FIFO por lote no FeedbackAgent (Fase 2).

Garante que uma venda fecha as compras na ordem certa (mais antiga primeiro) e
SO os lotes efetivamente consumidos — sem mais atribuir o resultado a todos os
longs do simbolo de uma vez (o bug do matching v1 que contaminava o dataset).
"""

from __future__ import annotations

from decimal import Decimal

from agents.feedback_agent import FeedbackAgent
from data.db import Database
from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, OutcomeStatus
from orchestration.bus import InMemoryBus


def _log():
    return DecisionLog(connection=Database(":memory:").conn)


def _record_buy(log, coid, ref):
    return log.record(
        Decision(strategy="ladder", symbol="AAPL", action=DecisionAction.BUY,
                 regime=MarketRegime.UNKNOWN, reference_price=Decimal(str(ref)),
                 client_order_id=coid)
    )


def _buy_fill(agent, bus, coid, qty, price):
    bus.publish("fills", {"symbol": "AAPL", "side": "buy", "filled_qty": Decimal(str(qty)),
                          "fill_price": Decimal(str(price)), "client_order_id": coid})
    agent.step(bus)


def _sell_fill(agent, bus, qty, price):
    bus.publish("fills", {"symbol": "AAPL", "side": "sell",
                          "filled_qty": Decimal(str(qty)), "fill_price": Decimal(str(price))})
    agent.step(bus)


def _by_id(log, did):
    return next(r for r in log.all_records() if r["id"] == did)


def test_sell_closes_only_oldest_lot_fifo():
    """Compra A: 5@100 ; Compra B: 10@90. Venda de 5@110 fecha SO o lote A
    (FIFO); o lote B continua aberto."""
    log = _log()
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    a = _record_buy(log, "A", 100)
    b = _record_buy(log, "B", 90)
    _buy_fill(agent, bus, "A", 5, 100)
    _buy_fill(agent, bus, "B", 10, 90)

    _sell_fill(agent, bus, 5, 110)

    assert _by_id(log, a)["outcome_status"] == OutcomeStatus.WIN.value
    assert Decimal(_by_id(log, a)["exit_price"]) == Decimal("110")
    # Lote B intacto: ainda aberto, remaining inalterado.
    assert _by_id(log, b)["outcome_status"] == OutcomeStatus.OPEN.value
    assert Decimal(_by_id(log, b)["remaining_qty"]) == Decimal("10")


def test_sell_crossing_two_lots_partials_the_second():
    """Compra A: 5@100 ; Compra B: 5@100. Venda de 8@110 fecha A inteiro e
    consome 3 de B (resta 2 abertos em B)."""
    log = _log()
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    a = _record_buy(log, "A", 100)
    b = _record_buy(log, "B", 100)
    _buy_fill(agent, bus, "A", 5, 100)
    _buy_fill(agent, bus, "B", 5, 100)

    _sell_fill(agent, bus, 8, 110)

    assert _by_id(log, a)["outcome_status"] == OutcomeStatus.WIN.value
    assert _by_id(log, b)["outcome_status"] == OutcomeStatus.OPEN.value
    assert Decimal(_by_id(log, b)["remaining_qty"]) == Decimal("2")


def test_partial_sell_keeps_lot_open_until_fully_consumed():
    """Compra 10@100. Venda 4@110 nao fecha (resta 6). Venda 6@120 fecha."""
    log = _log()
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    a = _record_buy(log, "A", 100)
    _buy_fill(agent, bus, "A", 10, 100)

    _sell_fill(agent, bus, 4, 110)
    assert _by_id(log, a)["outcome_status"] == OutcomeStatus.OPEN.value
    assert Decimal(_by_id(log, a)["remaining_qty"]) == Decimal("6")

    _sell_fill(agent, bus, 6, 120)
    row = _by_id(log, a)
    assert row["outcome_status"] == OutcomeStatus.WIN.value
    assert Decimal(row["exit_price"]) == Decimal("120")


def test_sell_does_not_touch_other_symbols():
    log = _log()
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    aapl = _record_buy(log, "A", 100)
    _buy_fill(agent, bus, "A", 5, 100)
    # decisao de outro simbolo, aberta:
    msft = log.record(
        Decision(strategy="t", symbol="MSFT", action=DecisionAction.BUY,
                 regime=MarketRegime.UNKNOWN, reference_price=Decimal("400"))
    )
    _sell_fill(agent, bus, 5, 110)  # venda de AAPL
    assert _by_id(log, aapl)["outcome_status"] == OutcomeStatus.WIN.value
    assert _by_id(log, msft)["outcome_status"] == OutcomeStatus.OPEN.value  # MSFT intacto


def test_legacy_lot_without_qty_still_closes():
    """Decisao sem fill de compra (qty NULL) cai no fechamento integral (v1)."""
    log = _log()
    bus = InMemoryBus()
    agent = FeedbackAgent(log)
    a = _record_buy(log, "A", 100)  # sem buy fill => remaining_qty fica NULL
    _sell_fill(agent, bus, 10, 90)
    assert _by_id(log, a)["outcome_status"] == OutcomeStatus.LOSS.value


def test_migration_adds_columns_to_existing_db():
    """Banco criado sem as colunas de lote recebe-as via migracao idempotente."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # tabela 'v1' sem qty/remaining_qty
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, "
        "strategy TEXT, symbol TEXT, action TEXT, regime TEXT, reference_price TEXT, "
        "signal_strength REAL DEFAULT 0, context TEXT, client_order_id TEXT, "
        "outcome_status TEXT DEFAULT 'open', entry_price TEXT, exit_price TEXT, "
        "realized_pnl TEXT, return_pct REAL, closed_at TEXT, outcome_note TEXT)"
    )
    conn.commit()
    DecisionLog(connection=conn)  # dispara a migracao
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(decisions)")}
    assert {"qty", "remaining_qty"} <= cols
