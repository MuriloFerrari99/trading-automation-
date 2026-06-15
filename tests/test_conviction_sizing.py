"""Testes do sizing por conviccao (o 'cerebro' ligado ao caminho vivo).

Cobre dois niveis:
  1. RiskManager conviction-aware: a confianca dirige a fracao de risco-por-trade
     (0 -> nao opera; alta -> ate o teto), sempre DENTRO dos caps duros.
  2. Planner: computa a confianca (via enricher) e a thread-a ate o sizing,
     apenas para intents que ABREM risco.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from config.risk import RiskSettings, build_dynamic_sizer
from core.models import OrderIntent, OrderSide
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard
from sizing.dynamic import DynamicSizer


def _settings(**over):
    base = dict(
        _env_file=None,
        RISK_PER_TRADE_PCT=Decimal("0.01"),
        MAX_PER_SYMBOL_PCT=Decimal("0.20"),
        MAX_PORTFOLIO_HEAT_PCT=Decimal("0.10"),
        DAILY_LOSS_LIMIT_PCT=Decimal("0.03"),
        MAX_DRAWDOWN_PCT=Decimal("0.20"),
        ASSUMED_STOP_PCT=Decimal("0.10"),
    )
    base.update(over)
    return RiskSettings(**base)


def _rm(sizer=None):
    return RiskManager(
        _settings(), PortfolioRiskGuard(start_equity=Decimal("100000")), sizer=sizer
    )


def _buy(qty="500"):
    return OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal(qty), strategy="t")


# --- RiskManager conviction-aware -------------------------------------------
def test_high_confidence_sizes_above_fixed():
    """Equity 100k, stop 10% => $10/acao de risco. Fixo 1% => 100 acoes; com
    alta conviccao a fracao sobe ate 2% => 200 acoes. A conviccao AUMENTOU o
    tamanho (dentro do teto por simbolo, que aqui tambem permite 200)."""
    sizer = DynamicSizer(kelly_cap=0.25, max_risk_pct=0.02, confidence_floor=0.5)
    d = _rm(sizer).assess(
        _buy(), Decimal("100000"), [], Decimal("100"), confidence=0.9
    )
    assert d.approved and d.intent.qty == Decimal("200")


def test_no_edge_vetoes():
    """Confianca abaixo do floor => Kelly 0 => nao opera (veto explicito)."""
    sizer = DynamicSizer(confidence_floor=0.5)
    d = _rm(sizer).assess(
        _buy(), Decimal("100000"), [], Decimal("100"), confidence=0.3
    )
    assert not d.approved and "sem edge" in d.reason


def test_confidence_none_falls_back_to_fixed():
    """Sizer presente, mas sem confianca => fracao FIXA legada (1% => 100)."""
    sizer = DynamicSizer()
    d = _rm(sizer).assess(_buy(), Decimal("100000"), [], Decimal("100"))
    assert d.approved and d.intent.qty == Decimal("100")


def test_no_sizer_ignores_confidence():
    """Sem sizer, uma confianca passada e ignorada => comportamento legado."""
    d = _rm(sizer=None).assess(
        _buy(), Decimal("100000"), [], Decimal("100"), confidence=0.9
    )
    assert d.approved and d.intent.qty == Decimal("100")


def test_caps_still_bind_with_conviction():
    """A conviccao NUNCA fura o teto por simbolo: 2% pediria 200, mas com uma
    posicao ja aberta o teto por simbolo (20% = 200 no total) limita o adicional."""
    sizer = DynamicSizer(kelly_cap=0.25, max_risk_pct=0.02, confidence_floor=0.5)
    from core.models import Position

    pos = [Position(symbol="AAPL", qty=Decimal("150"), avg_entry_price=Decimal("100"))]
    d = _rm(sizer).assess(_buy(), Decimal("100000"), pos, Decimal("100"), confidence=0.9)
    # total <= 200 => adicional <= 50, mesmo a 2%.
    assert d.approved and d.intent.qty == Decimal("50")


def test_gradient_with_low_kelly_cap():
    """Com kelly_cap baixo a fracao deixa de saturar e a confianca vira gradiente:
    mais confianca => mais risco => mais acoes. (Alavanca de tuning exposta.)"""
    sizer = DynamicSizer(kelly_cap=0.02, max_risk_pct=0.02, min_risk_pct=0.0, confidence_floor=0.5)
    rm = _rm(sizer)
    q_low = rm.assess(_buy(), Decimal("100000"), [], Decimal("100"), confidence=0.55).intent.qty
    q_high = rm.assess(_buy(), Decimal("100000"), [], Decimal("100"), confidence=0.95).intent.qty
    assert Decimal("0") < q_low < q_high


def test_build_dynamic_sizer_neutral_is_baseline_high_is_cap():
    """Default de config (decisao do CFO): NEUTRO (0.5) ~ 1% (= baseline fixo de
    hoje); ALTA conviccao escala ate o teto de 2%. Ligar o cerebro nao aumenta o
    risco por si so — so quando ha edge real."""
    sizer = build_dynamic_sizer(_settings())
    assert sizer.max_risk_pct == 0.02
    assert abs(sizer.risk_pct(0.5) - 0.01) < 1e-9   # neutro -> 1% (como antes)
    assert abs(sizer.risk_pct(0.9) - 0.02) < 1e-9   # alta conviccao -> teto 2%


# --- Planner: thread da confianca ate o sizing ------------------------------
class _StubStrategy:
    name = "stub"

    def __init__(self, intents):
        self._intents = intents

    def evaluate(self, ctx):
        return list(self._intents)


class _FakeEnricher:
    """Enricher de teste: devolve uma confianca fixa (duck-typed; o Planner so
    le .confidence e .context)."""

    def __init__(self, confidence):
        self._c = confidence

    def enrich(self, **kw):
        return SimpleNamespace(confidence=self._c, context={"fake": True})


def _planner(broker, enricher=None, sizer=None):
    from agents.planner import Planner
    from config.watchlist import Watchlist, WatchlistItem
    from data.db import Database
    from data.state_repo import StateRepository

    rm = RiskManager(
        _settings(), PortfolioRiskGuard(start_equity=Decimal("100000")), sizer=sizer
    )
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL")])
    state = StateRepository(Database(":memory:"))
    return Planner(
        broker, state, wl, [_StubStrategy([_buy()])],
        risk_manager=rm, enricher=enricher,
    )


def test_planner_threads_confidence_into_sizing():
    from broker.fake_broker import FakeBroker

    broker = FakeBroker(cash=Decimal("100000"), prices={"AAPL": Decimal("100")})
    broker.set_bars("AAPL", [100.0] * 60)  # barras p/ o regime (UNKNOWN ok)
    sizer = DynamicSizer(kelly_cap=0.25, max_risk_pct=0.02, confidence_floor=0.5)

    # Sem enricher: sizing fixo 1% => 100 acoes.
    legacy = _planner(broker, enricher=None, sizer=sizer).plan()
    assert len(legacy) == 1 and legacy[0].qty == Decimal("100")

    # Com enricher de alta confianca: sobe a 2% => 200 acoes.
    high = _planner(broker, enricher=_FakeEnricher(0.9), sizer=sizer).plan()
    assert len(high) == 1 and high[0].qty == Decimal("200")


def test_planner_vetoes_buy_without_edge():
    from broker.fake_broker import FakeBroker

    broker = FakeBroker(cash=Decimal("100000"), prices={"AAPL": Decimal("100")})
    broker.set_bars("AAPL", [100.0] * 60)
    sizer = DynamicSizer(confidence_floor=0.5)
    out = _planner(broker, enricher=_FakeEnricher(0.2), sizer=sizer).plan()
    assert out == []  # sem edge => nenhuma intencao sai do planner
