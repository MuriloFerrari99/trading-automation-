"""Testes do PortfolioRiskGuard e do RiskManager."""

from __future__ import annotations

from decimal import Decimal

from config.risk import RiskSettings
from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderSide,
)
from risk.manager import RiskManager
from risk.portfolio_guard import PortfolioRiskGuard


def _settings():
    return RiskSettings(
        _env_file=None,
        RISK_PER_TRADE_PCT=Decimal("0.01"),
        MAX_PER_SYMBOL_PCT=Decimal("0.20"),
        MAX_PORTFOLIO_HEAT_PCT=Decimal("0.10"),
        DAILY_LOSS_LIMIT_PCT=Decimal("0.03"),
        MAX_DRAWDOWN_PCT=Decimal("0.20"),
    )


def _guard(start=Decimal("100000"), **kw):
    return PortfolioRiskGuard(start_equity=start, **kw)


# --- PortfolioRiskGuard -----------------------------------------------------
def test_daily_loss_halts():
    g = _guard(daily_loss_pct=Decimal("0.03"))
    assert g.update(Decimal("98000")) is None  # -2%, ok
    reason = g.update(Decimal("96000"))  # -4% => halt
    assert reason is not None and g.trading_halted


def test_max_drawdown_halts():
    g = _guard(max_dd_pct=Decimal("0.20"))
    g.update(Decimal("120000"))  # pico 120k
    reason = g.update(Decimal("95000"))  # -20.8% do pico => halt
    assert reason is not None and g.trading_halted


def test_peak_update_callback_persists_high_water():
    """O pico persistido (callback) deixa o max drawdown sobreviver a restart."""
    saved = {}
    g = PortfolioRiskGuard(
        start_equity=Decimal("100000"),
        max_dd_pct=Decimal("0.20"),
        on_peak_update=lambda p: saved.__setitem__("peak", p),
    )
    g.update(Decimal("120000"))  # novo pico => persiste 120k
    assert saved["peak"] == Decimal("120000")

    # "Restart": novo guard restaura o pico de 120k (em vez de reancorar em 100k).
    g2 = PortfolioRiskGuard(
        start_equity=Decimal("100000"), max_dd_pct=Decimal("0.20"),
        peak_equity=saved["peak"],
    )
    reason = g2.update(Decimal("95000"))  # -20.8% do pico 120k => halt
    assert reason is not None and g2.trading_halted


def test_can_open_blocks_when_halted():
    g = _guard()
    g._halt("teste")
    ok, _ = g.can_open(Decimal("0.05"), Decimal("0.05"))
    assert ok is False


def test_can_open_per_symbol_and_heat():
    g = _guard()
    assert g.can_open(Decimal("0.25"), Decimal("0.05"))[0] is False  # >20%/simbolo
    assert g.can_open(Decimal("0.10"), Decimal("0.15"))[0] is False  # heat >10%
    assert g.can_open(Decimal("0.10"), Decimal("0.05"))[0] is True


# --- RiskManager ------------------------------------------------------------
def _rm(**guard_kw):
    return RiskManager(_settings(), _guard(**guard_kw))


def test_exit_always_allowed_even_when_halted():
    rm = _rm()
    rm.guard._halt("teste")
    sell = OrderIntent(symbol="AAPL", side=OrderSide.SELL, qty=Decimal("10"), strategy="t")
    d = rm.assess(sell, Decimal("100000"), [], Decimal("100"))
    assert d.approved  # saida nao e barrada por halt


def test_buy_vetoed_when_halted():
    rm = _rm()
    rm.begin_cycle(Decimal("90000"))  # -10% => halt
    buy = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"), strategy="t")
    d = rm.assess(buy, Decimal("90000"), [], Decimal("100"))
    assert not d.approved


def test_buy_failsafe_without_price():
    rm = _rm()
    buy = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"), strategy="t")
    d = rm.assess(buy, Decimal("100000"), [], None)  # sem preco
    assert not d.approved


def test_buy_qty_capped_by_risk_per_trade():
    rm = _rm()
    # equity 100k, risk 1% => $1k; stop assumido 10% do preco => $10/acao de
    # risco => 100 acoes. Pedindo 500, o cap de risco-por-trade binda em 100
    # (antes mesmo do teto por simbolo de 200).
    buy = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("500"), strategy="t")
    d = rm.assess(buy, Decimal("100000"), [], Decimal("100"))
    assert d.approved
    assert d.intent.qty == Decimal("100")


def test_buy_qty_capped_by_per_symbol_when_stop_is_tight():
    # Com stop assumido apertado (5%), o risco-por-trade permite 200 acoes, entao
    # quem binda passa a ser o teto por simbolo (tambem 200).
    rm = RiskManager(
        RiskSettings(
            _env_file=None,
            RISK_PER_TRADE_PCT=Decimal("0.01"),
            MAX_PER_SYMBOL_PCT=Decimal("0.20"),
            MAX_PORTFOLIO_HEAT_PCT=Decimal("0.10"),
            DAILY_LOSS_LIMIT_PCT=Decimal("0.03"),
            MAX_DRAWDOWN_PCT=Decimal("0.20"),
            ASSUMED_STOP_PCT=Decimal("0.05"),
        ),
        _guard(),
    )
    buy = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("500"), strategy="t")
    d = rm.assess(buy, Decimal("100000"), [], Decimal("100"))
    assert d.approved and d.intent.qty == Decimal("200")


def test_buy_approved_within_limits():
    rm = _rm()
    buy = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("50"), strategy="t")
    d = rm.assess(buy, Decimal("100000"), [], Decimal("100"))
    assert d.approved and d.intent.qty == Decimal("50")


def test_covered_call_is_not_risk_increasing():
    rm = _rm()
    contract = OptionContract(
        occ_symbol="KO260101C00060000", underlying="KO",
        option_type=OptionType.CALL, strike=Decimal("60"),
        expiration=__import__("datetime").date(2026, 1, 1),
    )
    call = OptionOrderIntent(contract=contract, qty=Decimal("1"))
    assert rm.is_risk_increasing(call) is False


def test_cash_secured_put_is_risk_increasing():
    rm = _rm()
    contract = OptionContract(
        occ_symbol="KO260101P00054000", underlying="KO",
        option_type=OptionType.PUT, strike=Decimal("54"),
        expiration=__import__("datetime").date(2026, 1, 1),
    )
    put = OptionOrderIntent(contract=contract, qty=Decimal("1"))
    assert rm.is_risk_increasing(put) is True
