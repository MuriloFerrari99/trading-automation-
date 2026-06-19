"""Sinais de order-flow + grade temporal para medir poder preditivo SEM look-ahead.

A Fase 1 pergunta: o sinal em t prevê o retorno t->t+h? Para responder sem
look-ahead, projetamos tudo numa GRADE DE RELOGIO fixa (ex.: a cada 1s):

  - Em cada ponto de grade t, o SINAL usa SO eventos em (t-window, t]  (passado).
  - O PRECO de referencia em t e o ultimo trade <= t (last-trade price).
  - O RETORNO FORWARD de horizonte h e r(t,h) = log(P(t+h)/P(t)) (futuro, disjunto
    do sinal). Sem sobreposicao sinal/retorno: o sinal termina em t, o retorno comeca em t.

Por que last-trade price como "mid"? Profundidade L2 historica nao e gratis e o
bookTicker (topo de livro) esta indisponivel no arquivo publico hoje. Com so
aggTrades, o melhor proxy de preco e o ultimo trade. Isso INFLA um pouco o ruido
(bid-ask bounce) mas NAO introduz look-ahead. Sinais que dependem de tamanhos do
livro (queue imbalance, microprice, OFI estilo Cont-Kukanov) ficam definidos aqui
porem SO calculaveis quando houver bookTicker/L2 — ver build_book_signals.

SINAIS construidos so com aggTrades (trade-flow):
  - tfi      : trade-flow imbalance = (V_buy - V_sell)/(V_buy + V_sell) na janela.
  - tfi_cnt  : imbalance por CONTAGEM de trades (robusto a outliers de tamanho).
  - tfi_log  : sign-log do fluxo assinado liquido (comprime caudas).
  - ofi_trade: aproximacao de OFI so-trade = soma do fluxo assinado / preco (proxy;
               o OFI "de verdade" usa mudancas de tamanho no topo do livro).

Saida: TimeGrid com vetores alinhados (ts, price, dict de sinais) + utilitario para
retornos forward por horizonte. Tudo vetorizado em numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from microstructure.data import TradeArrays

# Horizontes de retorno forward (segundos) — microestrutura decai rapido.
DEFAULT_HORIZONS_S: tuple[int, ...] = (1, 5, 10, 30, 60)

# Janelas de lookback do sinal (segundos). Sinal usa (t-window, t].
DEFAULT_WINDOWS_S: tuple[int, ...] = (1, 5, 10)


@dataclass
class TimeGrid:
    """Grade de relogio fixa derivada de um dia de trades.

    ts_ms   : pontos de grade (int64, espacados por step_s).
    price   : ultimo trade price <= ts (float64); NaN antes do 1o trade.
    signals : nome -> vetor alinhado a ts_ms (float64).
    """

    symbol: str
    day: str
    step_s: int
    ts_ms: np.ndarray
    price: np.ndarray
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.ts_ms.size)


def _grid_points(ts_ms: np.ndarray, step_s: int) -> np.ndarray:
    """Pontos de grade alinhados a multiplos de step_s, cobrindo [t0, tN]."""
    step_ms = step_s * 1000
    t0 = int(ts_ms[0])
    tN = int(ts_ms[-1])
    start = (t0 // step_ms) * step_ms
    return np.arange(start, tN + step_ms, step_ms, dtype=np.int64)


def _last_price_at(grid_ts: np.ndarray, trade_ts: np.ndarray, trade_px: np.ndarray) -> np.ndarray:
    """Ultimo trade price <= cada ponto de grade (step LOCF). NaN se nenhum trade ainda."""
    # idx do ultimo trade com trade_ts <= grid_ts: searchsorted 'right' - 1
    idx = np.searchsorted(trade_ts, grid_ts, side="right") - 1
    out = np.full(grid_ts.shape, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = trade_px[idx[valid]]
    return out


def _windowed_sums(
    grid_ts: np.ndarray,
    trade_ts: np.ndarray,
    values: np.ndarray,
    window_s: int,
) -> np.ndarray:
    """Soma de `values` dos trades em (t-window, t] para cada ponto de grade t.

    Vetorizado via soma cumulativa + dois searchsorted (O((G+N) log N)).
    Janela ABERTA a esquerda, FECHADA a direita: usa side='right' nos dois limites.
    """
    window_ms = window_s * 1000
    csum = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
    hi = np.searchsorted(trade_ts, grid_ts, side="right")  # # trades com ts <= t
    lo = np.searchsorted(trade_ts, grid_ts - window_ms, side="right")  # ts <= t-window
    return csum[hi] - csum[lo]


def build_trade_signals(
    trades: TradeArrays,
    *,
    step_s: int = 1,
    windows_s: tuple[int, ...] = DEFAULT_WINDOWS_S,
) -> TimeGrid:
    """Constroi a grade + sinais de trade-flow (so aggTrades) p/ um dia.

    Para cada janela w em windows_s gera: tfi_w, tfi_cnt_w, tfi_log_w, ofi_trade_w.
    Sem look-ahead: sinal em t usa (t-w, t]; preco em t e o ultimo trade <= t.
    """
    grid = _grid_points(trades.ts_ms, step_s)
    price = _last_price_at(grid, trades.ts_ms, trades.price)

    buy_vol_flag = (trades.side > 0).astype(np.float64) * trades.qty
    sell_vol_flag = (trades.side < 0).astype(np.float64) * trades.qty
    buy_cnt_flag = (trades.side > 0).astype(np.float64)
    sell_cnt_flag = (trades.side < 0).astype(np.float64)
    signed = trades.signed_qty  # side * qty

    signals: dict[str, np.ndarray] = {}
    eps = 1e-12
    for w in windows_s:
        vbuy = _windowed_sums(grid, trades.ts_ms, buy_vol_flag, w)
        vsell = _windowed_sums(grid, trades.ts_ms, sell_vol_flag, w)
        cbuy = _windowed_sums(grid, trades.ts_ms, buy_cnt_flag, w)
        csell = _windowed_sums(grid, trades.ts_ms, sell_cnt_flag, w)
        net = _windowed_sums(grid, trades.ts_ms, signed, w)

        tot_vol = vbuy + vsell
        tot_cnt = cbuy + csell
        tfi = np.where(tot_vol > eps, (vbuy - vsell) / (tot_vol + eps), 0.0)
        tfi_cnt = np.where(tot_cnt > eps, (cbuy - csell) / (tot_cnt + eps), 0.0)
        # sign-log: comprime caudas do fluxo assinado liquido
        tfi_log = np.sign(net) * np.log1p(np.abs(net))
        # OFI so-trade (proxy): fluxo assinado normalizado pelo preco (unidade ~ qty)
        ofi_trade = np.where(np.isfinite(price) & (price > 0), net / price, 0.0)

        signals[f"tfi_{w}s"] = tfi
        signals[f"tfi_cnt_{w}s"] = tfi_cnt
        signals[f"tfi_log_{w}s"] = tfi_log
        signals[f"ofi_trade_{w}s"] = ofi_trade

    return TimeGrid(
        symbol=trades.symbol, day=trades.day, step_s=step_s,
        ts_ms=grid, price=price, signals=signals,
    )


def forward_log_return(price: np.ndarray, step_s: int, horizon_s: int) -> np.ndarray:
    """Retorno log forward de t->t+h sobre a grade. r[i] = log(P[i+k]/P[i]), k=h/step.

    Posicoes sem futuro suficiente -> NaN. Sem look-ahead: usa precos FUTUROS apenas.
    """
    k = max(1, round(horizon_s / step_s))
    out = np.full(price.shape, np.nan, dtype=np.float64)
    if k >= price.size:
        return out
    p0 = price[:-k]
    p1 = price[k:]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(p1 / p0)
    good = np.isfinite(r) & np.isfinite(p0) & (p0 > 0)
    out[:-k][good] = r[good]
    return out


# ---------------------------------------------------------------------------
# SINAIS DE LIVRO (queue imbalance / microprice / OFI) — exigem bookTicker/L2
# ---------------------------------------------------------------------------

def queue_imbalance(bid_size: np.ndarray, ask_size: np.ndarray) -> np.ndarray:
    """(bid_size - ask_size)/(bid_size + ask_size). Em [-1, 1]. Exige top-of-book."""
    bid_size = np.asarray(bid_size, dtype=np.float64)
    ask_size = np.asarray(ask_size, dtype=np.float64)
    tot = bid_size + ask_size
    return np.where(tot > 0, (bid_size - ask_size) / tot, 0.0)


def microprice_deviation(
    bid: np.ndarray, ask: np.ndarray, bid_size: np.ndarray, ask_size: np.ndarray
) -> np.ndarray:
    """(microprice - mid)/mid. microprice = (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz).

    O microprice pesa o preco do lado com MENOS fila (mais provavel de mover).
    Exige top-of-book (bookTicker/L2): indisponivel no arquivo publico hoje.
    """
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    bs = np.asarray(bid_size, dtype=np.float64)
    as_ = np.asarray(ask_size, dtype=np.float64)
    tot = bs + as_
    micro = np.where(tot > 0, (bid * as_ + ask * bs) / tot, np.nan)
    mid = (bid + ask) / 2.0
    return np.where(mid > 0, (micro - mid) / mid, np.nan)


def order_flow_imbalance(
    bid: np.ndarray, ask: np.ndarray, bid_size: np.ndarray, ask_size: np.ndarray
) -> np.ndarray:
    """OFI estilo Cont-Kukanov a partir de top-of-book sequencial.

    e_n = 1{b_n>=b_{n-1}}*B_n - 1{b_n<=b_{n-1}}*B_{n-1}
        - 1{a_n<=a_{n-1}}*A_n + 1{a_n>=a_{n-1}}*A_{n-1}
    (B=bid_size, A=ask_size). Mede pressao liquida no topo do livro.
    Exige a SEQUENCIA de bookTicker: indisponivel no arquivo publico hoje (wired).
    """
    bid = np.asarray(bid, dtype=np.float64)
    ask = np.asarray(ask, dtype=np.float64)
    bs = np.asarray(bid_size, dtype=np.float64)
    as_ = np.asarray(ask_size, dtype=np.float64)
    n = bid.size
    e = np.zeros(n, dtype=np.float64)
    if n < 2:
        return e
    db = bid[1:] - bid[:-1]
    da = ask[1:] - ask[:-1]
    e[1:] = (
        (db >= 0) * bs[1:] - (db <= 0) * bs[:-1]
        - (da <= 0) * as_[1:] + (da >= 0) * as_[:-1]
    )
    return e
