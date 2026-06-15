"""Testes dos SignalProviders e do SignalService.

Inclui a garantia central: sinais NUNCA viram ordens (nao chegam ao Executor).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from core.models import OrderSide
from strategies.signals.base import SignalProvider, SignalService
from strategies.signals.congress import (
    CongressDisclosure,
    CongressTradingProvider,
    StaticCongressSource,
)
from strategies.signals.smart_money import (
    FundPositionChange,
    SmartMoneyProvider,
    StaticSmartMoneySource,
)


# --- Congress ---------------------------------------------------------------
def _disclosure(disclosure_date: date, symbol="NVDA", side=OrderSide.BUY):
    return CongressDisclosure(
        symbol=symbol,
        side=side,
        politician="Rep. Doe",
        transaction_date=disclosure_date,
        disclosure_date=disclosure_date,
    )


def test_congress_fresh_disclosure_high_confidence():
    today = date(2026, 6, 15)
    src = StaticCongressSource([_disclosure(date(2026, 6, 15))])
    provider = CongressTradingProvider(src, today=today)
    signals = provider.fetch()
    assert len(signals) == 1
    assert signals[0].source == "congress"
    assert signals[0].confidence == 1.0  # idade 0d


def test_congress_confidence_decays_with_age():
    today = date(2026, 6, 15)
    # divulgada ha ~22 dias (metade do limite de 45) -> confianca ~0.5
    src = StaticCongressSource([_disclosure(date(2026, 5, 24))])
    provider = CongressTradingProvider(src, today=today)
    sig = provider.fetch()[0]
    assert 0.45 < sig.confidence < 0.55


def test_congress_drops_stale_disclosure():
    today = date(2026, 6, 15)
    # > 45 dias atras: sem edge, descartado
    src = StaticCongressSource([_disclosure(date(2026, 4, 1))])
    provider = CongressTradingProvider(src, today=today)
    assert provider.fetch() == []


# --- Smart money ------------------------------------------------------------
def test_smart_money_signal_and_threshold():
    src = StaticSmartMoneySource(
        [
            FundPositionChange(symbol="AAPL", fund="BigFund", side=OrderSide.BUY, change_ratio=Decimal("0.5")),
            FundPositionChange(symbol="TSLA", fund="BigFund", side=OrderSide.SELL, change_ratio=Decimal("0.02")),  # ruido
        ]
    )
    provider = SmartMoneyProvider(src)  # min_change_ratio default 0.10
    signals = provider.fetch()
    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].confidence == 0.5


# --- Service ----------------------------------------------------------------
class _BoomProvider(SignalProvider):
    name = "boom"

    def fetch(self):
        raise RuntimeError("fonte indisponivel")


def test_service_isolates_failing_provider():
    good = SmartMoneyProvider(
        StaticSmartMoneySource(
            [FundPositionChange(symbol="MSFT", fund="F", side=OrderSide.BUY, change_ratio=Decimal("0.3"))]
        )
    )
    service = SignalService([_BoomProvider(), good])
    signals = service.collect()  # provider que falha nao derruba o resto
    assert len(signals) == 1
    assert signals[0].symbol == "MSFT"
