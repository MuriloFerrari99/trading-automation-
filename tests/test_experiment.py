"""Testes offline do experimento de melhoria (sem rede)."""

from __future__ import annotations

from simulation.experiment import RunConfig, run_continuous, run_experiment


def _series(closes):
    opens = [closes[0]] + closes[:-1]
    return {
        "open": opens,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
    }


def _downtrend(n=160, start=100.0, drift=-0.01):
    # queda forte e sustentada: cenario classico de falling knife p/ o ladder.
    return [start * (1 + drift) ** i for i in range(n)]


def test_risk_layer_eliminates_ruin_in_downtrend():
    ohlc = _series(_downtrend())

    baseline = run_continuous(ohlc and "X" or "X", ohlc, RunConfig("baseline"))
    with_risk = run_continuous("X", ohlc, RunConfig("risk", use_risk=True, use_stop=True))

    assert baseline is not None and with_risk is not None
    # baseline cru pode estourar a conta (MDD <= -100%); com risco, nunca.
    assert with_risk.max_drawdown > -0.999


def test_run_continuous_returns_oos_metrics():
    ohlc = _series([100 + i * 0.1 for i in range(160)])  # leve alta
    m = run_continuous("X", ohlc, RunConfig("risk", use_risk=True))
    assert m is not None
    assert hasattr(m, "sharpe")


def test_run_experiment_produces_three_configs_per_cost():
    data = {
        "UP": _series([100 * 1.005 ** i for i in range(200)]),
        "DOWN": _series(_downtrend(200)),
    }
    results = run_experiment(data, costs=(5.0,))
    names = [r.name for r in results]
    assert names == ["A_baseline", "B_risk+stop", "C_risk+decision"]
    # config com risco nao deve ter mais ruina que o baseline
    by = {r.name: r for r in results}
    assert by["B_risk+stop"].n_ruined <= by["A_baseline"].n_ruined
