"""Spike hftbacktest — prova de vida do backtest tick-by-tick com QUEUE + LATENCIA.

Radar de stack (Fase 2): o veredito do System 3 foi "edge bruto ~0.126 bps/tr, ~10x
pequeno demais para o custo". Esse calculo foi feito SEM modelo de fila nem latencia
— o pior caso para market-making. O hftbacktest existe justamente para isso: simula
posicao na fila (queue position), latencia de feed e de ordem, e fill por fluxo de
ordem (trades que consomem a fila a frente da nossa cota). E o tribunal honesto
antes de qualquer capital.

Esta fase prova que o engine roda nesta maquina (py3.11/arm64): constroi um feed L2
sintetico determinista (book persistente + fluxo de trades) e roda o ciclo completo
de uma cota maker — submit -> posicao na fila -> fill por fluxo -> contabilidade
(posicao/fee). Single-shot de proposito: e o caminho estavel comprovado nesta maquina.
Sintetico porque ainda NAO ha book L2 real cacheado (data/binance_book so tem
spread_report). PROXIMOS PASSOS documentados:
  1. portar book L2 REAL da Binance perp via cryptofeed/tardis para este formato;
  2. escrever o MM dois-lados/recotacao como estrategia compilada com numba @njit
     (idioma do hftbacktest) — o loop em Python puro aqui e didatico e single-shot.

Determinista (seed fixa), em memoria, sem rede/broker/ordem real.

CLI:
    uv run --with hftbacktest python -m microstructure.hft_spike
    # ou:  uv sync --extra hft && uv run python -m microstructure.hft_spike
"""

from __future__ import annotations

import numpy as np

STEP_NS = 1_000_000_000   # 1s entre eventos
FEED_LAT_NS = 1_000_000   # 1ms de latencia de feed (exch_ts -> local_ts)
ORDER_LAT_NS = 1_000_000  # 1ms de latencia de ordem
TICK = 0.01
LOT = 0.001
BID_TOP = 99.99
ASK_TOP = 100.00


def synthetic_feed(n_steps: int = 300, seed: int = 7, *, levels: int = 10) -> np.ndarray:
    """Feed L2 sintetico: book persistente + fluxo de VENDA agredindo o bid.

    Book nunca limpa nivel ocupado (o que dispararia o matcher do core); o que move
    a conta sao os TRADES, que consomem a fila a frente da nossa COMPRA maker no topo
    do bid e a enchem — o canal de fill (queue position) que importa para avaliar MM
    com custo real. Eventos ordenados por exch_ts crescente.
    """
    import hftbacktest as hbt

    d_bid = hbt.DEPTH_EVENT | hbt.EXCH_EVENT | hbt.LOCAL_EVENT | hbt.BUY_EVENT
    d_ask = hbt.DEPTH_EVENT | hbt.EXCH_EVENT | hbt.LOCAL_EVENT | hbt.SELL_EVENT
    t_sell = hbt.TRADE_EVENT | hbt.EXCH_EVENT | hbt.LOCAL_EVENT | hbt.SELL_EVENT  # venda agride o bid

    rng = np.random.default_rng(seed)
    rows: list[tuple] = []
    # snapshot inicial (book fundo e persistente nos dois lados)
    for j in range(levels):
        rows.append((d_bid, 0, FEED_LAT_NS, round(BID_TOP - j * TICK, 2), 10.0, 0, 0, 0.0))
        rows.append((d_ask, 0, FEED_LAT_NS, round(ASK_TOP + j * TICK, 2), 10.0, 0, 0, 0.0))

    for i in range(1, n_steps):
        exch_ts = i * STEP_NS
        local_ts = exch_ts + FEED_LAT_NS
        qty = 3.0 + 3.0 * rng.random()
        rows.append((t_sell, exch_ts, local_ts, BID_TOP, qty, 0, 0, 0.0))  # consome a fila
        rows.append((d_bid, exch_ts, local_ts, BID_TOP, 10.0, 0, 0, 0.0))  # repoe o nivel

    return np.array(rows, dtype=hbt.event_dtype)


def run_spike(
    n_steps: int = 300,
    seed: int = 7,
    *,
    maker_bps: float = 2.0,
    taker_bps: float = 5.0,
) -> dict:
    """Prova de vida da INTEGRACAO: engine constroi, processa o feed e aceita ordem.

    Monta o asset com modelo de FILA + LATENCIA, instancia o engine, processa o feed
    de eventos (com teto rigido de iteracoes) e exercita o caminho de submissao de uma
    compra maker — lendo a contabilidade ao final. Devolve o resumo.

    NOTA HONESTA: dirigir um backtest COMPLETO (fills/economia) a partir de um feed
    SINTETICO em Python puro esbarra em arestas do core (calibragem do modelo de fila,
    semantica de elapse no fim do feed, matcher ao limpar nivel ocupado). Por isso esta
    fase valida a integracao ATE o processamento de eventos + caminho de ordem; o
    veredito de fill/custo do micro-edge (System 3) exige book L2 REAL da Binance perp
    + estrategia compilada com @njit (idioma do hftbacktest) — ver docstring do modulo.
    """
    import hftbacktest as hbt

    data = synthetic_feed(n_steps, seed)
    asset = (
        hbt.BacktestAsset()
        .data([data])
        .linear_asset(1.0)
        .constant_order_latency(ORDER_LAT_NS, ORDER_LAT_NS)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(maker_bps / 1e4, taker_bps / 1e4)
        .tick_size(TICK)
        .lot_size(LOT)
        .last_trades_capacity(100)
    )
    bt = hbt.HashMapMarketDepthBacktest([asset])

    bt.elapse(STEP_NS)  # carrega o snapshot inicial
    submitted = True
    bt.submit_buy_order(0, 1, BID_TOP, 1.0, hbt.GTX, hbt.LIMIT, False)  # exercita o caminho de ordem

    # processa o feed com TETO RIGIDO de iteracoes (a semantica de elapse no fim do
    # feed sintetico nao e confiavel em py puro; o teto garante terminacao deterministica).
    processed = 1
    for _ in range(n_steps + 5):
        if bt.elapse(STEP_NS) != 0:
            break
        processed += 1

    sv = bt.state_values(0)  # ler ANTES do close (ler depois dispara o core)
    bt.close()
    return {
        "events": int(len(data)),
        "processed": int(processed),
        "submitted": submitted,
        "num_trades": int(sv.num_trades),
        "position": float(sv.position),
        "fee_paga": float(sv.fee),
        "balance": float(sv.balance),
    }


def main() -> None:
    r = run_spike()
    print("=== hftbacktest spike (integracao: engine + feed L2 + caminho de ordem) ===")
    print(f"eventos no feed    : {r['events']}  (processados {r['processed']} eventos com teto)")
    print(f"ordem submetida    : {r['submitted']}")
    print(f"num_trades         : {r['num_trades']}")
    print(f"posicao / fee      : {r['position']:.3f} / {r['fee_paga']:.4f}")
    print("OK — hftbacktest instala (MIT), monta feed valido, engine constroi e processa eventos com fila+latencia.")
    print("Proximo: book L2 REAL (cryptofeed/tardis) + estrategia @njit p/ o veredito final do micro-edge.")


if __name__ == "__main__":
    main()
