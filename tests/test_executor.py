"""Testes do Executor: kill switch, validacao, retry e auditoria."""

from __future__ import annotations

from decimal import Decimal

from agents.executor import Executor
from core.models import OrderIntent, OrderResult, OrderSide


def _buy(symbol="AAPL", qty="10"):
    return OrderIntent(symbol=symbol, side=OrderSide.BUY, qty=Decimal(qty), strategy="test")


def test_kill_switch_blocks_order(broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    kill_switch.engage("teste")
    ex = Executor(broker, trade_logger, kill_switch)
    result = ex.execute(_buy())
    assert result is None
    assert broker.submitted == []  # nenhuma ordem chegou na corretora
    assert trade_logger.recent() == []


def test_executes_and_logs(broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    ex = Executor(broker, trade_logger, kill_switch)
    result = ex.execute(_buy(qty="5"))
    assert result is not None
    assert result.status == "filled"
    logged = trade_logger.recent()
    assert len(logged) == 1
    assert logged[0]["symbol"] == "AAPL"
    assert logged[0]["strategy"] == "test"


def test_validation_blocks_insufficient_buying_power(trade_logger, kill_switch):
    from broker.fake_broker import FakeBroker

    broker = FakeBroker(cash=Decimal("100"), prices={"AAPL": Decimal("100")})
    ex = Executor(broker, trade_logger, kill_switch)
    result = ex.execute(_buy(qty="10"))  # custo ~1000 > buying power 100
    assert result is None
    assert broker.submitted == []


def test_kill_switch_engaged_before_network_blocks_submit(broker, trade_logger, kill_switch):
    """Kill switch engajado DURANTE a preparacao (apos record_pending) barra a rede."""
    broker.set_price("AAPL", "100")
    ks = kill_switch
    original = broker.submit_order

    def engage_then_submit(intent):  # nao deveria nem ser chamado
        return original(intent)

    # Engaja o kill switch no meio: a checagem antes da rede deve barrar.
    real_get = broker.get_order_by_client_id

    def get_and_engage(cid):
        ks.engage("engajado no meio do fluxo")
        return real_get(cid)

    broker.get_order_by_client_id = get_and_engage  # type: ignore[method-assign]
    broker.submit_order = engage_then_submit  # type: ignore[method-assign]
    ex = Executor(broker, trade_logger, ks)
    result = ex.execute(_buy(qty="1"))
    assert result is None
    assert broker.submitted == []  # nada chegou na corretora


def test_max_orders_per_cycle_caps_batch(broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    ex = Executor(broker, trade_logger, kill_switch, max_orders_per_cycle=2)
    # symbols distintos => client_order_ids distintos (evita colisao de idempotencia).
    intents = [_buy(symbol=s, qty="1") for s in ("AAA", "BBB", "CCC", "DDD", "EEE")]
    for s in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        broker.set_price(s, "100")
    results = ex.execute_many(intents)
    assert len(results) == 2  # so as 2 primeiras passam (anti-rajada)
    assert len(broker.submitted) == 2


def test_retry_then_success(broker, trade_logger, kill_switch):
    broker.set_price("AAPL", "100")
    calls = {"n": 0}
    original = broker.submit_order

    def flaky(intent):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("erro transitorio da API")
        return original(intent)

    broker.submit_order = flaky  # type: ignore[method-assign]
    ex = Executor(broker, trade_logger, kill_switch, max_retries=3, backoff_seconds=0)
    result = ex.execute(_buy(qty="1"))
    assert isinstance(result, OrderResult)
    assert calls["n"] == 2  # falhou 1x, sucesso na 2a
