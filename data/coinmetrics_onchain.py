"""Alt-data on-chain via Coin Metrics (community tier — gratis, sem API key).

Radar de stack (Fase 2): o veredito consolidado F2 e que alpha price-based morreu e
a unica frente ainda nao esgotada e DADO ALTERNATIVO ortogonal a preco. Metricas
on-chain (enderecos ativos, contagem de transacoes, etc.) sao exatamente isso —
nao derivam de preco. O cliente oficial da Coin Metrics expoe um tier "community"
gratuito e auditavel, ideal para um primeiro corte de sinal antes de pagar dados.

Este modulo so PUXA e cacheia series (LE rede, ESCREVE csv em data/onchain_cache).
Nao decide trade, nao toca broker. O proximo passo e cruzar essas series com retorno
futuro (ex.: via simulation.alphalens_spike) para medir IC e decidir se ha edge.

CLI:
    uv run --with coinmetrics-api-client python -m data.coinmetrics_onchain
    # ou:  uv sync --extra data && uv run python -m data.coinmetrics_onchain
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path("data/onchain_cache")
# Metricas do tier community (gratis, confirmadas) — todas ortogonais a preco.
COMMUNITY_METRICS = ("AdrActCnt", "TxCnt", "SplyCur")
DEFAULT_ASSETS = ("btc", "eth")


def fetch_onchain(
    assets=DEFAULT_ASSETS,
    metrics=COMMUNITY_METRICS,
    *,
    frequency: str = "1d",
    page_size: int = 1000,
    limit_per_asset: int | None = None,
) -> pd.DataFrame:
    """Puxa metricas on-chain do community tier -> DataFrame longo (asset, time, ...)."""
    from coinmetrics.api_client import CoinMetricsClient

    client = CoinMetricsClient()  # sem key = community tier
    resp = client.get_asset_metrics(
        assets=list(assets),
        metrics=list(metrics),
        frequency=frequency,
        page_size=page_size,
        limit_per_asset=limit_per_asset,
    )
    return resp.to_dataframe()


def cache_onchain(
    assets=DEFAULT_ASSETS,
    metrics=COMMUNITY_METRICS,
    *,
    frequency: str = "1d",
) -> Path:
    """Puxa e grava um csv por (asset) em data/onchain_cache. Retorna o diretorio."""
    df = fetch_onchain(assets, metrics, frequency=frequency)
    CACHE.mkdir(parents=True, exist_ok=True)
    for asset, g in df.groupby("asset"):
        g.to_csv(CACHE / f"{asset}_{frequency}.csv", index=False)
    return CACHE


def main() -> None:
    df = fetch_onchain(limit_per_asset=5)
    print("=== Coin Metrics on-chain (community tier, gratis) ===")
    print(f"metricas: {', '.join(COMMUNITY_METRICS)}  | ativos: {', '.join(DEFAULT_ASSETS)}")
    cols = [c for c in df.columns if c in ("asset", "time", *COMMUNITY_METRICS)]
    print(df[cols].tail(10).to_string(index=False))
    print("OK — alt-data on-chain ortogonal a preco, sem API key. Proximo: medir IC vs retorno (alphalens).")


if __name__ == "__main__":
    main()
