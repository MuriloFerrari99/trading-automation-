"""Teste do spike hftbacktest (backtest tick-by-tick com queue + latencia).

Opt-in: hftbacktest e extra (uv sync --extra hft). Sem ele, o teste e PULADO.
Feed pequeno e determinista (seed fixa) — em memoria, sem rede/broker.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hftbacktest")

from microstructure.hft_spike import run_spike, synthetic_feed


def test_feed_tem_formato_de_evento_valido():
    import hftbacktest as hbt

    data = synthetic_feed(n_steps=50, seed=7)
    assert data.dtype == hbt.event_dtype
    assert len(data) > 0
    # ha eventos de book (DEPTH) e de fluxo (TRADE)
    assert (data["ev"] & hbt.DEPTH_EVENT).any()
    assert (data["ev"] & hbt.TRADE_EVENT).any()


def test_engine_constroi_processa_eventos_e_aceita_ordem():
    r = run_spike(n_steps=120, seed=7)
    assert r["submitted"] is True, "caminho de submissao de ordem falhou"
    assert r["processed"] > 1, "engine nao processou eventos do feed"
    assert r["num_trades"] >= 0 and r["fee_paga"] >= 0.0  # contabilidade legivel
