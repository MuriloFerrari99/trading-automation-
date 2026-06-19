"""System 3 — scalping por ORDER-FLOW / MICROESTRUTURA (cripto, Binance).

Track ISOLADO e firewalled: NAO toca em nada do beta (beta_*, nav_history,
main.py, Alpaca) nem nos tribunais existentes. Reutiliza, sem quebrar:

  - simulation.binance_book : downloader/coletor de bookTicker + coletor L2 ao vivo.
  - simulation.statistics   : aparato estatistico (Sharpe/PSR/DSR) p/ a Fase 2.

FASE 1 (este pacote): a pergunta fundamental e barata — EXISTE sinal de
order-flow que PREVE o retorno de curtissimo prazo? Ainda NAO e claim de
tradabilidade; so "ha sinal?". O criterio e o Information Coefficient (IC):
correlacao do sinal em t com o retorno forward t->t+h, sem look-ahead, e seu
decaimento/estabilidade ao longo de horizontes e dias.

Modulos:
  - microstructure.data    : downloader de aggTrades (gratis) + bookTicker
                             (wired; arquivo historico pode estar indisponivel).
  - microstructure.signals : sinais de order-flow (trade-flow imbalance é o
                             unico construivel so com aggTrades; queue imbalance,
                             microprice e OFI exigem top-of-book/bookTicker).
  - microstructure.ic      : medicao de IC por sinal x horizonte + decaimento +
                             estabilidade entre dias, e o relatorio/veredito.
"""

from __future__ import annotations

CACHE_DIRNAME = "data/microstructure_cache"
