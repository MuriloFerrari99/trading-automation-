"""Teste do spike alphalens (IC cross-sectional sobre klines reais da Binance).

Opt-in: alphalens-reloaded e extra (uv sync --extra research). Sem ele, ou sem os
klines cacheados em data/binance_1m, o teste e PULADO.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("alphalens")

from simulation.alphalens_spike import build_panel, reversal_factor, run_spike

_HAS_DATA = any(Path("data/binance_1m").glob("*/*.csv.gz"))
_SOME_MONTHS = ["2024-06", "2024-07", "2024-08"]


@pytest.mark.skipif(not _HAS_DATA, reason="klines binance_1m ausentes")
def test_painel_e_fator_tem_formato_alphalens():
    prices = build_panel(months=_SOME_MONTHS)
    assert prices.shape[1] >= 2  # cross-section precisa de >=2 ativos
    factor = reversal_factor(prices)
    assert factor.index.names == ["date", "asset"]
    assert factor.notna().all()


@pytest.mark.skipif(not _HAS_DATA, reason="klines binance_1m ausentes")
def test_ic_calcula_para_cada_horizonte():
    r = run_spike(months=_SOME_MONTHS, periods=(1, 6))
    assert r["n_assets"] >= 2
    assert len(r["ic_mean"]) == 2  # um IC por horizonte pedido
    assert all(-1.0 <= v <= 1.0 for v in r["ic_mean"].values())
    assert r["verdict"]
