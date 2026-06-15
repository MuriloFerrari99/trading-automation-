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
