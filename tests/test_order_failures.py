"""Ramos de falha do caminho de ordem (o caminho do dinheiro).

Cobre o que o FakeBroker antes nao conseguia exercitar: fill PARCIAL e timeout
pos-registro (broker tem a ordem, cliente nao ve a resposta) + recuperacao
idempotente na reconexao.
"""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from broker.fake_broker import FakeBroker
from core.kill_switch import KillSwitch
from core.models import OrderIntent, OrderSide
from data.audit_log import AuditLog
from data.db import Database
from data.order_repo import PARTIALLY_FILLED, OrderRepository
from data.trade_logger import TradeLogger


def _buy(symbol="AAPL", qty="10"):
    return OrderIntent(symbol=symbol, side=OrderSide.BUY, qty=Decimal(qty), strategy="t")


def test_partial_fill_persists_partially_filled():
    db = Database(":memory:")
    broker = FakeBroker(prices={"AAPL": Decimal("100")}, partial_fill_ratio=Decimal("0.5"))
    repo = OrderRepository(db)
    ex = Executor(
        broker, TradeLogger(db), KillSwitch("/tmp/__nk_pf__"),
        order_repo=repo, audit=AuditLog(db),
    )
    intent = _buy(qty="10")
    result = ex.execute(intent)

    # filled_qty REAL (nunca assume fill total).
    assert result is not None
    assert result.filled_qty == Decimal("5")
    assert result.status == "partially_filled"
    row = repo.get(intent.client_order_id)
    assert row["status"] == PARTIALLY_FILLED
    assert Decimal(row["filled_qty"]) == Decimal("5")


def test_timeout_after_record_then_idempotent_recovery_on_reconnect():
    db = Database(":memory:")
    broker = FakeBroker(prices={"AAPL": Decimal("100")}, fail_after_record=True)
    repo = OrderRepository(db)
    ks = KillSwitch("/tmp/__nk_recon__")
    audit = AuditLog(db)
    intent = _buy(qty="5")

    # Sessao 1: timeout pos-registro. Cliente nao ve resposta (retorna None),
    # mas o broker JA registrou a ordem.
    ex1 = Executor(
        broker, TradeLogger(db), ks, order_repo=repo, audit=audit,
        max_retries=1, backoff_seconds=0,
    )
    assert ex1.execute(intent) is None
    assert broker.get_order_by_client_id(intent.client_order_id) is not None

    # "Reconexao": novo Executor, rede de volta. A MESMA intencao nao duplica —
    # a idempotencia pre-submit reencontra a ordem no broker e nao reenvia.
    broker.set_fail_after_record(False)
    ex2 = Executor(broker, TradeLogger(db), ks, order_repo=repo, audit=audit)
    assert ex2.execute(intent) is None
    assert broker.submitted == [intent]  # submetida UMA unica vez
