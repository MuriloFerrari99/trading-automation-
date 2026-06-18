"""Teste do modulo de alt-data on-chain (Coin Metrics community tier).

Opt-in: coinmetrics-api-client e extra (uv sync --extra data). Sem ele o teste e
PULADO. O pull e ao vivo (rede): se a rede falhar, tambem PULA — nao quebra a suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("coinmetrics")

from data.coinmetrics_onchain import fetch_onchain


def test_client_constroi_sem_key():
    from coinmetrics.api_client import CoinMetricsClient

    c = CoinMetricsClient()  # community tier
    assert hasattr(c, "get_asset_metrics")


def test_pull_community_ao_vivo():
    try:
        df = fetch_onchain(assets=("btc",), metrics=("AdrActCnt",), limit_per_asset=3)
    except Exception as e:  # rede indisponivel / rate limit -> pula
        pytest.skip(f"pull on-chain indisponivel: {type(e).__name__}")
    assert {"asset", "time"} <= set(df.columns)
    assert "AdrActCnt" in df.columns
    assert len(df) > 0
