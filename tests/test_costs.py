"""Testes do modelo de custos e da sua integracao com o SimBroker."""

from __future__ import annotations

from decimal import Decimal

from core.models import OrderIntent, OrderSide, OrderType
from simulation.costs import CRYPTO_BASE, EQUITY_BASE, CostModel
from simulation.sim_broker import SimBroker


def test_round_trip_and_stress():
    assert CRYPTO_BASE.per_side_bps == 35.0
    assert CRYPTO_BASE.round_trip_bps == 70.0
    s = CRYPTO_BASE.stressed(2.0)
    assert s.commission_bps == 50.0
    assert s.slippage_bps == 20.0
    assert s.round_trip_bps == 140.0
    assert "stress2x" in s.name


def test_crypto_costs_higher_than_equity():
    assert CRYPTO_BASE.round_trip_bps > EQUITY_BASE.round_trip_bps * 5


def test_sim_broker_applies_cost_model():
    bars = [100.0] * 5
    broker = SimBroker("BTCUSD", bars, bars, bars, bars, cash=10_000.0, cost=CRYPTO_BASE)
    broker.set_index(0)
    # compra a mercado -> preenche no open do proximo bar com slippage + comissao
    intent = OrderIntent(
        strategy="t", symbol="BTCUSD", side=OrderSide.BUY,
        qty=Decimal("1"), order_type=OrderType.MARKET,
    )
    broker.submit_order(intent)
    fills = broker.fill_pending_at_open(1)
    assert len(fills) == 1
    # preco de compra = 100 * (1 + 10bps slippage) = 100.10
    assert abs(float(fills[0]["fill_price"]) - 100.10) < 1e-6
    # caixa debitado = 100.10 + comissao 25bps sobre o nocional
    expected_cash = 10_000.0 - (100.10 + 100.10 * 0.0025)
    assert abs(float(broker.get_account().cash) - expected_cash) < 1e-4


def test_default_bps_still_work_without_cost_model():
    bars = [50.0] * 3
    broker = SimBroker("AAPL", bars, bars, bars, bars)  # sem cost= : usa defaults 5/5 bps
    assert broker.get_account().cash == Decimal("100000.0")
