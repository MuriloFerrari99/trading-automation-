"""Testes do guard de seguranca paper-only em config.settings."""

from __future__ import annotations

import pytest

from config.settings import PAPER_ENDPOINT, LiveTradingBlockedError, Settings


def _make(**overrides) -> Settings:
    base = dict(
        ALPACA_API_KEY="key",
        ALPACA_SECRET_KEY="secret",
        ALPACA_ENDPOINT=PAPER_ENDPOINT,
        LIVE_TRADING=False,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_paper_settings_ok():
    s = _make()
    assert s.is_paper is True
    assert s.alpaca_endpoint == PAPER_ENDPOINT


def test_live_trading_blocked():
    with pytest.raises(LiveTradingBlockedError):
        _make(LIVE_TRADING=True)


def test_non_paper_endpoint_blocked():
    with pytest.raises(LiveTradingBlockedError):
        _make(ALPACA_ENDPOINT="https://api.alpaca.markets")


def test_blank_credentials_rejected():
    with pytest.raises(ValueError):
        _make(ALPACA_API_KEY="   ")
