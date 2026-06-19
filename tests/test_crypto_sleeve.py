"""Testes da trava de risco isolada do sleeve de cripto (Sistema 2).

Cobre: teto rigido de alavancagem (mesmo com input absurdo), parede de fogo
(nao toca capital do Sistema 1), kills de perda diaria e drawdown, e sizing por
risco correto contra o capital ISOLADO do sleeve.
"""

from __future__ import annotations

from decimal import Decimal

from config.crypto_sleeve import LEVERAGE_HARD_CEILING, CryptoSleeveSettings
from core.kill_switch import KillSwitch
from risk.crypto_sleeve import CryptoSleeveGuard


def _settings(**kw) -> CryptoSleeveSettings:
    base = dict(
        _env_file=None,
        CRYPTO_SLEEVE_CAPITAL_USD=Decimal("1000"),
        CRYPTO_LEVERAGE_CAP=Decimal("2"),
        CRYPTO_RISK_PER_TRADE_PCT=Decimal("0.005"),
        CRYPTO_DAILY_LOSS_LIMIT_PCT=Decimal("0.03"),
        CRYPTO_MAX_DRAWDOWN_PCT=Decimal("0.10"),
        CRYPTO_MAX_CONCURRENT_POSITIONS=3,
        CRYPTO_MAX_GROSS_EXPOSURE_X=Decimal("2"),
    )
    base.update(kw)
    return CryptoSleeveSettings(**base)  # type: ignore[arg-type]


def _guard(tmp_path=None, **kw) -> CryptoSleeveGuard:
    # Kill switch apontando p/ um sentinela inexistente => desengajado.
    sentinel = (tmp_path / "NOPE") if tmp_path is not None else "/tmp/__no_such_kill_switch__"
    return CryptoSleeveGuard(_settings(**kw), kill_switch=KillSwitch(sentinel))


# --- Teto de alavancagem ----------------------------------------------------
def test_leverage_ceiling_enforced_on_absurd_config():
    """Configurar 50x deve cair para o ceiling rigido (3x)."""
    g = _guard(CRYPTO_LEVERAGE_CAP=Decimal("50"))
    assert g.leverage_cap == LEVERAGE_HARD_CEILING
    assert g.leverage_cap <= Decimal("3")


def test_leverage_ceiling_caps_position_notional(tmp_path):
    """Mesmo pedindo muito, o notional nunca implica leverage acima do ceiling."""
    # Stop colado (risco/share minusculo) tentaria comprar muita qty.
    g = _guard(tmp_path, CRYPTO_LEVERAGE_CAP=Decimal("50"))
    d = g.allowed_size(price=Decimal("100"), stop_price=Decimal("99.99"))
    assert d.allowed
    # notional / capital nunca passa do ceiling (3x).
    assert d.implied_leverage <= LEVERAGE_HARD_CEILING + Decimal("0.0001")


def test_gross_exposure_capped_below_leverage():
    """Exposicao bruta nunca excede a alavancagem efetiva."""
    g = _guard(CRYPTO_MAX_GROSS_EXPOSURE_X=Decimal("99"))
    assert g.max_gross_exposure_x <= g.leverage_cap


# --- Parede de fogo (nao toca capital do Sistema 1) -------------------------
def test_firewall_sizing_uses_only_sleeve_capital(tmp_path):
    """Sizing dimensiona contra o capital do sleeve, nao um equity gigante."""
    g = _guard(tmp_path)
    # risco = 1000 * 0.5% = $5; risco/share = |100-95| = 5 => qty = 1.0
    d = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"))
    assert d.allowed
    assert d.qty == Decimal("1")
    # Notional ($100) jamais excede o capital * ceiling ($3000).
    assert d.notional_usd <= g.sleeve_capital_usd * g.leverage_cap


def test_firewall_capital_independent_of_system1():
    """Trocar o equity do Sistema 1 nao muda nada: guard so conhece o sleeve."""
    g_small = _guard()
    # Nao existe nenhum parametro de equity do Sistema 1 no guard.
    assert g_small.sleeve_capital_usd == Decimal("1000")
    assert not hasattr(g_small, "start_equity")  # nao herda equity do Sistema 1


# --- Kills independentes -----------------------------------------------------
def test_daily_loss_kill_triggers():
    g = _guard()
    assert g.update(Decimal("980")) is None  # -2%, ok
    reason = g.update(Decimal("960"))  # -4% => kill
    assert reason is not None and g.trading_halted


def test_drawdown_kill_triggers():
    g = _guard()
    g.update(Decimal("1200"))  # pico 1200
    reason = g.update(Decimal("1070"))  # -10.8% do pico => kill
    assert reason is not None and g.trading_halted


def test_halt_blocks_new_size(tmp_path):
    g = _guard(tmp_path)
    g.update(Decimal("950"))  # -5% => halt diario
    d = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"))
    assert not d.allowed
    assert d.qty == Decimal("0")
    assert "halt" in d.reason.lower()


def test_daily_kill_independent_of_drawdown():
    """Perda diaria mata sozinha, mesmo sem drawdown pico-a-vale grande."""
    g = _guard()
    # Sem novo pico: cai direto -3.5% do start => kill diario antes do -10% DD.
    reason = g.update(Decimal("965"))
    assert reason is not None and "daily" in reason.lower()


# --- Sizing por risco --------------------------------------------------------
def test_risk_sizing_correct(tmp_path):
    g = _guard(tmp_path)
    # capital 1000, risco 0.5% = $5; |200-190|=10 => qty = 0.5
    d = g.allowed_size(price=Decimal("200"), stop_price=Decimal("190"))
    assert d.allowed
    assert d.qty == Decimal("0.5")


def test_signal_scales_risk(tmp_path):
    g = _guard(tmp_path)
    full = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"), signal=1.0)
    half = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"), signal=0.5)
    assert half.qty == full.qty / 2


def test_invalid_inputs_blocked(tmp_path):
    g = _guard(tmp_path)
    assert not g.allowed_size(price=Decimal("0"), stop_price=Decimal("95")).allowed
    assert not g.allowed_size(price=Decimal("100"), stop_price=Decimal("0")).allowed
    assert not g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"), signal=0).allowed


def test_max_concurrent_positions_blocks(tmp_path):
    g = _guard(tmp_path)
    ok = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"), open_positions=2)
    blocked = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"), open_positions=3)
    assert ok.allowed and not blocked.allowed


def test_gross_exposure_cap_blocks(tmp_path):
    g = _guard(tmp_path)
    # Ja com notional = capital*gross (2x = $2000) => sem espaco.
    d = g.allowed_size(
        price=Decimal("100"),
        stop_price=Decimal("95"),
        current_gross_notional_usd=Decimal("2000"),
    )
    assert not d.allowed


# --- Kill switch global ------------------------------------------------------
def test_global_kill_switch_respected(tmp_path):
    sentinel = tmp_path / "KILL_SWITCH"
    ks = KillSwitch(sentinel)
    g = CryptoSleeveGuard(_settings(), kill_switch=ks)
    assert g.allowed_size(price=Decimal("100"), stop_price=Decimal("95")).allowed
    ks.engage("teste")
    d = g.allowed_size(price=Decimal("100"), stop_price=Decimal("95"))
    assert not d.allowed
    assert "kill switch global" in d.reason.lower()
