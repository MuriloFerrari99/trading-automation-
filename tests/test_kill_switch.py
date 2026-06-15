"""Testes do kill switch central."""

from __future__ import annotations

import pytest

from core.kill_switch import KillSwitch, KillSwitchEngagedError


def test_clear_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    assert ks.is_engaged() is False
    ks.ensure_clear()  # nao levanta


def test_engaged_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KILL_SWITCH", "true")
    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    assert ks.is_engaged() is True
    with pytest.raises(KillSwitchEngagedError):
        ks.ensure_clear()


def test_engaged_by_file(tmp_path, monkeypatch):
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    ks.engage("teste")
    assert ks.is_engaged() is True
    with pytest.raises(KillSwitchEngagedError):
        ks.ensure_clear()
    ks.release()
    assert ks.is_engaged() is False
