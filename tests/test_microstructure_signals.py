"""Testes do System 3 (microestrutura) — corretude dos sinais e do IC sem look-ahead.

Foco: garantir que (a) os sinais de order-flow sao calculados como definidos,
(b) NAO ha look-ahead (sinal em t usa so passado; retorno usa so futuro), e
(c) um sinal SINTETICO com edge plantado produz IC alto e positivo, enquanto
ruido puro produz IC ~0. Sem rede.
"""

from __future__ import annotations

import numpy as np

from microstructure.data import TradeArrays
from microstructure.ic import _pearson, _rankdata, _spearman, run_ic_study
from microstructure.signals import (
    build_trade_signals,
    forward_log_return,
    order_flow_imbalance,
    queue_imbalance,
    microprice_deviation,
)


def _make_trades(ts_ms, price, side, qty, symbol="BTCUSDT", day="2026-06-13"):
    ts = np.asarray(ts_ms, dtype=np.int64)
    px = np.asarray(price, dtype=np.float64)
    sd = np.asarray(side, dtype=np.int64)
    qt = np.asarray(qty, dtype=np.float64)
    return TradeArrays(
        symbol=symbol, day=day, market="um",
        ts_ms=ts, price=px, qty=qt, side=sd, signed_qty=sd.astype(float) * qt,
    )


def test_rankdata_ties_average():
    r = _rankdata(np.array([10.0, 10.0, 20.0, 5.0]))
    # 5->1, 10 e 10 -> media de ranks 2,3 = 2.5, 20 -> 4
    assert list(r) == [2.5, 2.5, 4.0, 1.0]


def test_pearson_perfect_and_spearman_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(_pearson(x, 2 * x + 1) - 1.0) < 1e-9
    # relacao monotonica nao-linear: Spearman=1, Pearson<1
    y = x**3
    assert abs(_spearman(x, y) - 1.0) < 1e-9
    assert _pearson(x, y) < 1.0


def test_queue_imbalance_bounds_and_sign():
    qi = queue_imbalance(np.array([10.0, 0.0, 5.0]), np.array([0.0, 10.0, 5.0]))
    assert np.allclose(qi, [1.0, -1.0, 0.0])


def test_microprice_deviation_leans_to_thin_side():
    # bid=100, ask=101, muito mais tamanho no bid (fila grande no bid) ->
    # microprice puxa p/ o ASK (lado fino) -> deviation > 0.
    dev = microprice_deviation(
        np.array([100.0]), np.array([101.0]), np.array([100.0]), np.array([1.0])
    )
    assert dev[0] > 0


def test_ofi_sign_on_rising_bid():
    # bid sobe e ask sobe com tamanhos positivos -> OFI deve refletir pressao compradora
    bid = np.array([100.0, 100.5])
    ask = np.array([101.0, 101.5])
    bs = np.array([5.0, 5.0])
    as_ = np.array([5.0, 5.0])
    e = order_flow_imbalance(bid, ask, bs, as_)
    # db>0 -> +bs[1]; da>0 -> +as_[0]; net = 5 + 5 = 10 > 0
    assert e[1] > 0


def test_forward_return_no_lookahead_tail_is_nan():
    price = np.array([100.0, 101.0, 102.0, 103.0])
    r = forward_log_return(price, step_s=1, horizon_s=1)
    # ultimo ponto nao tem futuro -> NaN; r[0]=log(101/100)
    assert np.isnan(r[-1])
    assert abs(r[0] - np.log(101 / 100)) < 1e-12
    # r[i] usa P[i+1] (futuro), nunca P[i-1]
    assert abs(r[1] - np.log(102 / 101)) < 1e-12


def test_windowed_signal_uses_only_past():
    # trades: t=0.5s buy, t=1.5s sell. Grade step=1s -> pontos 0,1,2...
    # No ponto t=1s, janela 1s = (0,1] inclui o buy de 0.5s, NAO o sell de 1.5s.
    ts = [500, 1500]
    side = [1, -1]
    qty = [1.0, 1.0]
    price = [100.0, 100.0]
    ta = _make_trades(ts, price, side, qty)
    grid = build_trade_signals(ta, step_s=1, windows_s=(1,))
    # encontra indice do ponto de grade == 1000ms
    i = int(np.where(grid.ts_ms == 1000)[0][0])
    # tfi_1s em t=1s deve ser +1 (so o buy passado), nao 0 (que seria buy+sell)
    assert abs(grid.signals["tfi_1s"][i] - 1.0) < 1e-9
    # em t=2s (=2000ms), janela (1000,2000] pega o sell de 1500 -> tfi=-1
    j = int(np.where(grid.ts_ms == 2000)[0][0])
    assert abs(grid.signals["tfi_1s"][j] - (-1.0)) < 1e-9


def test_planted_edge_gives_high_ic_and_noise_gives_zero():
    """Constroi um dia onde o fluxo comprador PRECEDE alta de preco -> IC alto+.

    A cada segundo: escolhe um sinal aleatorio de fluxo; se compra liquida, o preco
    do PROXIMO segundo sobe; se venda, desce. Trade-flow deve prever o retorno 1s.
    """
    rng = np.random.default_rng(0)
    n_sec = 1200
    base_t = 1_700_000_000_000
    ts, side, qty, price = [], [], [], []
    p = 100.0
    flow_dir = rng.choice([-1, 1], size=n_sec)
    for k in range(n_sec):
        # um trade por segundo, no meio do segundo, com a direcao planejada
        ts.append(base_t + k * 1000 + 500)
        side.append(int(flow_dir[k]))
        qty.append(1.0)
        price.append(p)
        # o preco do PROXIMO segundo reage ao fluxo deste (com ruido)
        p *= 1.0 + flow_dir[k] * 0.0005 + rng.normal(0, 0.0001)
    ta = _make_trades(ts, price, side, qty)
    study = run_ic_study(
        "BTCUSDT", step_s=1, windows_s=(1,), horizons_s=(1,), trades_by_day=[ta]
    )
    assert study is not None
    ic = study.table[("tfi_1s", 1)].ic_spearman_mean
    assert ic > 0.3, f"edge plantado deveria dar IC alto+, veio {ic}"

    # agora ruido puro: fluxo independente do retorno -> IC ~ 0
    ts2, side2, qty2, price2 = [], [], [], []
    p = 100.0
    for k in range(n_sec):
        ts2.append(base_t + k * 1000 + 500)
        side2.append(int(rng.choice([-1, 1])))
        qty2.append(1.0)
        price2.append(p)
        p *= 1.0 + rng.normal(0, 0.0003)  # retorno independente do fluxo
    ta2 = _make_trades(ts2, price2, side2, qty2)
    study2 = run_ic_study(
        "BTCUSDT", step_s=1, windows_s=(1,), horizons_s=(1,), trades_by_day=[ta2]
    )
    ic2 = abs(study2.table[("tfi_1s", 1)].ic_spearman_mean)
    assert ic2 < 0.12, f"ruido deveria dar IC~0, veio {ic2}"
