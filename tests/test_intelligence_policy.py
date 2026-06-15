"""Testes da DecisionPolicy (pura)."""

from __future__ import annotations

from feedback.evaluation import GroupStats
from intelligence.decision_policy import DecisionPolicy


def _combo(*, n_trades, total_pnl, wins, gross_profit, gross_loss) -> GroupStats:
    g = GroupStats(label="x")
    g.n_trades = n_trades
    g.total_pnl = total_pnl
    g.wins = wins
    g.losses = n_trades - wins
    g.gross_profit = gross_profit
    g.gross_loss = gross_loss
    return g


def test_sem_historico_permite():
    v = DecisionPolicy().evaluate_combo(None)
    assert v.allow is True
    assert "insuficiente" in v.reason


def test_amostra_pequena_permite():
    combo = _combo(n_trades=3, total_pnl=-100, wins=0, gross_profit=0, gross_loss=-100)
    v = DecisionPolicy(min_samples=12).evaluate_combo(combo)
    assert v.allow is True  # poucos trades: nao confia no historico ainda


def test_expectancy_negativa_veta():
    combo = _combo(n_trades=20, total_pnl=-100, wins=5, gross_profit=50, gross_loss=-150)
    v = DecisionPolicy(min_samples=12).evaluate_combo(combo)
    assert v.allow is False
    assert "VETADO" in v.reason


def test_expectancy_positiva_permite():
    combo = _combo(n_trades=20, total_pnl=200, wins=14, gross_profit=300, gross_loss=-100)
    v = DecisionPolicy(min_samples=12).evaluate_combo(combo)
    assert v.allow is True


def test_sinal_alinhado_eleva_score():
    combo = _combo(n_trades=20, total_pnl=200, wins=14, gross_profit=300, gross_loss=-100)
    pol = DecisionPolicy(min_samples=12)
    fraco = pol.evaluate_combo(combo, signal_strength=0.0).score
    forte = pol.evaluate_combo(combo, signal_strength=1.0).score
    assert forte > fraco
    assert 0.0 <= fraco <= 1.0 and 0.0 <= forte <= 1.0
