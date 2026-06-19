"""Fases 3+4 do Nautilus — port de estrategia de alpha real + node ao vivo (paper).

Fase 3: a IntradayMomentum (event-driven, no Nautilus) reproduz o calculo
analitico vetorizado do tribunal (simulation.crypto_intraday.run_h1_config) ao
nivel do bps — mesmos trades, mesmo gross. A diferenca residual e amostragem de
execucao (OPEN vs CLOSE), ~0 em 1min.

Fase 4: a fiacao do TradingNode ao vivo (dados Binance + execucao sandbox) monta
sem erro OFFLINE (nao conecta).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BTC_CACHE = Path("data/binance_1m/BTCUSDT/BTCUSDT-1m-2024-06.csv.gz")
_skip_crypto = pytest.mark.skipif(not _BTC_CACHE.exists(), reason="cache de klines 1min ausente")


@_skip_crypto
def test_momentum_port_reproduz_o_analitico():
    from simulation.nautilus_momentum import validate

    p = validate("BTCUSDT", k=15, z_thr=1.5, hold=10, months=["2024-06"])
    # Mesmo numero de trades: o motor dispara exatamente o que o analitico calcula.
    assert p.nt_n_trades == p.an_n_trades, f"nt={p.nt_n_trades} an={p.an_n_trades}"
    assert p.nt_n_trades > 100, "esperado muitos trades intraday em 1 mes"
    # Gross medio por trade bate ao nivel do bps (dif = amostragem OPEN vs CLOSE).
    assert p.mean_diff_bps < 0.05, (
        f"gross medio divergente: nt={p.nt_mean_gross_bps:.3f} an={p.an_mean_gross_bps:.3f} bps"
    )


def test_live_node_fiacao_monta():
    """A fiacao do node ao vivo (paper) constroi sem erro, sem conectar."""
    from simulation.nautilus_live_node import build_node

    node = build_node(testnet=True)
    try:
        assert str(node.trader_id) == "LIVE-MOM-001"
    finally:
        node.dispose()
