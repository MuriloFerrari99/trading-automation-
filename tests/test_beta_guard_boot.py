"""Regressao do HALT FANTASMA do guard de beta (NOVO-1 pelo caminho da excecao).

resolve_sleeve_nav ENGOLE uma falha transitoria de get_account e devolve NAV=0.
Se o guard for alimentado com esse 0 (start_equity coagido a 1), daily_pl vira
(0-1)/1 = -100% => HALT, e o book do beta congela pela sessao inteira (so cura
com restart do daemon). O fix: NAV <= 0 = leitura ilegivel/conta vazia => o gate
diario fica INERTE no boot, nao halta. Ver main._build_beta_guard."""

from __future__ import annotations

from decimal import Decimal

import pytest

from broker.fake_broker import FakeBroker
from data.audit_log import AuditLog
from main import _build_beta_guard


@pytest.fixture(autouse=True)
def _force_paper_only(monkeypatch):
    # Forca o ramo paper SO-BETA (NAV = equity da conta), independente do .env do
    # dev: e nesse ramo que resolve_sleeve_nav engole o erro de get_account.
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL", "0")
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL_PCT", "1.0")


def test_no_phantom_halt_when_account_unreadable_at_boot(db, state, monkeypatch):
    """Hiccup transitorio do broker no boot (get_account levanta) => NAV=0; o guard
    NAO pode haltar por isso (senao congela o book pela sessao)."""
    broker = FakeBroker(cash=Decimal("100000"), prices={})

    def _raise():
        raise RuntimeError("broker indisponivel (transitorio)")

    monkeypatch.setattr(broker, "get_account", _raise)

    guard = _build_beta_guard(broker, state, AuditLog(db))
    assert not guard.trading_halted, "NAV ilegivel no boot NAO deve haltar o book"


def test_healthy_boot_is_not_halted(db, state):
    """Boot normal (conta com caixa, flat): NAV positivo => guard sadio/inerte."""
    broker = FakeBroker(cash=Decimal("100000"), prices={})
    guard = _build_beta_guard(broker, state, AuditLog(db))
    assert not guard.trading_halted


def test_real_drawdown_still_halts(db, state):
    """A guarda NAO mascara perda REAL: com start/peak ja persistidos (dia anterior
    a 100k) e o NAV de boot bem abaixo do limite de DD, o guard DEVE haltar."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()
    state.set_decimal(f"beta:risk_start_equity:{today}", Decimal("100000"))
    state.set_decimal("beta:risk_peak_equity", Decimal("100000"))

    # NAV de boot = 50k => -50% do pico, muito alem do DD-halt (-43%). Deve haltar.
    broker = FakeBroker(cash=Decimal("50000"), prices={})
    guard = _build_beta_guard(broker, state, AuditLog(db))
    assert guard.trading_halted, "perda REAL alem do DD-halt deve haltar"
