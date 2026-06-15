"""Camada de acesso a corretora (isolada atras de interface)."""

from broker.base import AccountInfo, BrokerClient
from broker.fake_broker import FakeBroker

__all__ = ["AccountInfo", "BrokerClient", "FakeBroker"]

# AlpacaBroker e importado sob demanda (requer o SDK alpaca-py instalado):
#   from broker.alpaca_broker import AlpacaBroker
