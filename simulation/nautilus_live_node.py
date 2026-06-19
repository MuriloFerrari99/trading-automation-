"""Fase 4 da adocao do Nautilus — TradingNode AO VIVO (paper) com a estrategia portada.

ATENCAO sobre Alpaca: o NautilusTrader NAO tem adapter oficial da Alpaca (os
adapters existentes sao binance, bybit, interactive_brokers, okx, coinbase_intx,
dydx, hyperliquid, betfair, polymarket, databento, tardis, bitmex). Portanto o
caminho REAL de paper trading event-driven aqui e:

    feed de dados AO VIVO da Binance  +  SandboxExecutionClient (fills simulados
    pelo MESMO matching engine do backtest, contra os precos ao vivo).

Isso e paper trading de verdade: precos reais entrando em tempo real, execucao
simulada deterministica. A MESMA IntradayMomentum (simulation/nautilus_momentum.py)
que validamos no backtest roda sem alteracao — so muda o ambiente (LIVE em vez de
BACKTEST). Para ir a um broker de acoes (Alpaca) seria preciso escrever um adapter
(fora do escopo desta fase).

Construir/validar a fiacao e OFFLINE e seguro (nao conecta). Rodar de fato
(`--run`) conecta na Binance (rede; chaves via BINANCE_API_KEY/BINANCE_API_SECRET
no .env — dados de mercado da Binance sao publicos, mas o adapter espera as chaves)
e fica no ar ate Ctrl+C.

CLI:
    uv run python -m simulation.nautilus_live_node            # valida a fiacao e sai
    uv run python -m simulation.nautilus_live_node --run      # conecta e roda ao vivo
    uv run python -m simulation.nautilus_live_node --run --testnet
"""

from __future__ import annotations

import os
import sys

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import (
    ImportableStrategyConfig,
    InstrumentProviderConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId

VENUE = "BINANCE"
INSTRUMENT_ID = "BTCUSDT-PERP.BINANCE"  # perp: a estrategia tem SHORT
BAR_TYPE = f"{INSTRUMENT_ID}-1-MINUTE-LAST-EXTERNAL"  # klines 1min vem do proprio venue (EXTERNAL)

# Mesma config validada no backtest (simulation/nautilus_momentum.py).
STRATEGY_PARAMS = {"k": 15, "z_thr": 1.5, "hold": 10}


def build_config(*, testnet: bool = False) -> TradingNodeConfig:
    """Monta a TradingNodeConfig (puro, OFFLINE — so valida o schema/fiacao)."""
    instrument_provider = InstrumentProviderConfig(
        load_ids=frozenset([InstrumentId.from_str(INSTRUMENT_ID)])
    )

    data_client = BinanceDataClientConfig(
        account_type=BinanceAccountType.USDT_FUTURE if hasattr(BinanceAccountType, "USDT_FUTURE")
        else BinanceAccountType.USDT_FUTURES,
        testnet=testnet,
        instrument_provider=instrument_provider,
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
    )

    # Execucao SIMULADA (paper) pelo matching engine do Nautilus, na venue BINANCE.
    exec_client = SandboxExecutionClientConfig(
        venue=VENUE,
        account_type="MARGIN",  # perp suporta SHORT
        base_currency="USDT",
        starting_balances=["1_000_000 USDT"],
        oms_type="NETTING",
        instrument_provider=instrument_provider,
    )

    strategy = ImportableStrategyConfig(
        strategy_path="simulation.nautilus_momentum:IntradayMomentum",
        config_path="simulation.nautilus_momentum:IntradayMomentumConfig",
        config={
            "instrument_id": INSTRUMENT_ID,
            "bar_type": BAR_TYPE,
            **STRATEGY_PARAMS,
        },
    )

    return TradingNodeConfig(
        trader_id="LIVE-MOM-001",
        logging=LoggingConfig(log_level="INFO"),
        data_clients={VENUE: data_client},
        exec_clients={VENUE: exec_client},
        strategies=[strategy],
    )


def build_node(*, testnet: bool = False) -> TradingNode:
    """Instancia o node e registra as factories (OFFLINE — nao conecta ainda)."""
    node = TradingNode(config=build_config(testnet=testnet))
    node.add_data_client_factory(VENUE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(VENUE, SandboxLiveExecClientFactory)
    return node


def main() -> None:
    args = sys.argv[1:]
    run = "--run" in args
    testnet = "--testnet" in args

    print("=== Nautilus TradingNode (paper: dados Binance ao vivo + execucao sandbox) ===")
    node = build_node(testnet=testnet)  # valida toda a fiacao, sem conectar
    print(f"  node            : {node.trader_id}")
    print(f"  instrumento     : {INSTRUMENT_ID}")
    print(f"  barras          : {BAR_TYPE}")
    print(f"  estrategia      : IntradayMomentum {STRATEGY_PARAMS}")
    print(f"  data client     : Binance ({'testnet' if testnet else 'live'}) USDT-FUTURES")
    print("  exec client     : Sandbox (fills simulados pelo matching engine)")
    print("  FIACAO VALIDA (config + node + factories montados sem erro).")

    if not run:
        print("\nDry-run: nao conectei. Para rodar ao vivo:")
        print("  uv run python -m simulation.nautilus_live_node --run [--testnet]")
        print("  (precisa de BINANCE_API_KEY/BINANCE_API_SECRET no ambiente)")
        node.dispose()
        return

    if not os.getenv("BINANCE_API_KEY"):
        print("\n[abortado] --run pedido mas BINANCE_API_KEY ausente no ambiente.")
        node.dispose()
        return

    print("\nConectando na Binance e rodando ao vivo (Ctrl+C para parar)...")
    try:
        node.build()
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
