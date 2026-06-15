"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from broker.fake_broker import FakeBroker
from config.watchlist import Watchlist, WatchlistItem
from core.kill_switch import KillSwitch
from data.db import Database
from data.state_repo import StateRepository
from data.trade_logger import TradeLogger


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def state(db: Database) -> StateRepository:
    return StateRepository(db)


@pytest.fixture
def trade_logger(db: Database) -> TradeLogger:
    return TradeLogger(db)


@pytest.fixture
def kill_switch(tmp_path, monkeypatch) -> KillSwitch:
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    return KillSwitch(tmp_path / "KILL_SWITCH")


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker(cash=Decimal("100000"), prices={})


@pytest.fixture
def watchlist() -> Watchlist:
    return Watchlist(
        items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))]
    )
