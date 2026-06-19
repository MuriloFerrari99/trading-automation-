"""Fase 3 da adocao do Nautilus — portar uma ESTRATEGIA DE ALPHA real (event-driven).

Ate aqui portamos o trailing protetivo (Fase 2). Agora portamos o sinal de alpha
H1 do nosso tribunal de cripto intraday (simulation/crypto_intraday.run_h1_config):

  sinal no CLOSE de t:  z = momentum_k / vol_k   (momentum = soma dos ultimos k
  retornos log de 1min; vol_k = desvio dos retornos de 1min na janela VOL_WINDOW,
  escalado por sqrt(k)). Long se z>+z_thr, short se z<-z_thr; segura `hold` barras;
  posicoes NAO sobrepostas.

O tribunal calcula isso de forma VETORIZADA (entra no open[t+1], sai no open[t+1+h]).
Aqui rodamos a MESMA logica EVENT-DRIVEN no Nautilus (perp BTCUSDT + conta MARGIN,
pois o sinal tem SHORT — conta CASH spot nao shorta). Validacao: a distribuicao de
retornos por trade do motor bate com a do calculo analitico? A unica diferenca
esperada e amostragem de execucao (analitico usa OPENs; o bar_execution do Nautilus
casa a mercado no CLOSE da barra) — e ja provamos (nautilus_backtest) que em 1min
isso e ~0. Custo ZERO dos dois lados (fee do perp zerada) para isolar o motor.

CLI:
    uv run python -m simulation.nautilus_momentum                 # BTCUSDT k15 z1.5 h10
    uv run python -m simulation.nautilus_momentum ETHUSDT 10 2.0 5
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from simulation.crypto_intraday import VOL_WINDOW, load_symbol, run_h1_config
from simulation.nautilus_backtest import _zero_fees

START_BALANCE = 1_000_000.0
TRADE_QTY = 1.0  # BTC por trade; irrelevante p/ retorno%% (comparamos gross por trade)


# --------------------------------------------------------------------------- #
# Estrategia portada                                                          #
# --------------------------------------------------------------------------- #
class IntradayMomentumConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    k: int
    z_thr: float
    hold: int
    vol_window: int = VOL_WINDOW
    trade_qty: float = TRADE_QTY


class IntradayMomentum(Strategy):
    def __init__(self, config: IntradayMomentumConfig) -> None:
        super().__init__(config)
        self._bar_type = BarType.from_str(config.bar_type)
        # Janela FIXA dos ultimos vol_window retornos (vol_window >= k na grade) —
        # evita O(n^2) de reconstruir a serie inteira a cada barra em 263k barras.
        self._r: deque[float] = deque(maxlen=config.vol_window)
        self._seen = 0                     # total de barras vistas (warmup)
        self._prev_close: float | None = None
        self._idx = -1                     # indice da barra corrente
        self._in_pos = False
        self._entry_idx = -1
        self._entry_side: OrderSide | None = None
        self.trade_returns: list[float] = []  # realized_return por posicao fechada

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self._bar_type.instrument_id)
        self.subscribe_bars(self._bar_type)

    def on_position_closed(self, event) -> None:
        # realized_return ja encodifica o lado (long: (saida-entrada)/entrada; short: inverso).
        self.trade_returns.append(float(event.realized_return))

    def _market(self, side: OrderSide, *, reduce_only: bool) -> None:
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=side,
                quantity=self.instrument.make_qty(self.config.trade_qty),
                reduce_only=reduce_only,
            )
        )

    def on_bar(self, bar: Bar) -> None:
        self._idx += 1
        self._seen += 1
        close = float(bar.close)
        r1 = 0.0 if self._prev_close is None else float(np.log(close / self._prev_close))
        self._r.append(r1)
        self._prev_close = close

        k, vw, hold = self.config.k, self.config.vol_window, self.config.hold

        # Em posicao: fecha apos `hold` barras (mesma duracao do analitico).
        if self._in_pos:
            if self._idx - self._entry_idx >= hold:
                exit_side = OrderSide.SELL if self._entry_side == OrderSide.BUY else OrderSide.BUY
                self._market(exit_side, reduce_only=True)
                self._in_pos = False
                self._entry_side = None
            return

        # Flat: precisa de historico (igual ao analitico, 1a barra valida em vw+k).
        if self._seen <= vw + k:
            return
        r = np.fromiter(self._r, dtype=float)  # deque de <= vw elementos: barato
        mom_k = float(r[-k:].sum())
        vol_k = float(r.std(ddof=0)) * float(np.sqrt(k))  # r ja sao os ultimos vw retornos
        z = mom_k / vol_k if vol_k > 0 else 0.0
        if not np.isfinite(z) or abs(z) < self.config.z_thr:
            return

        side = OrderSide.BUY if z > 0 else OrderSide.SELL
        self._market(side, reduce_only=False)
        self._in_pos = True
        self._entry_idx = self._idx
        self._entry_side = side


# --------------------------------------------------------------------------- #
# Harness de validacao                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MomentumParity:
    symbol: str
    k: int
    z_thr: float
    hold: int
    bars: int
    nt_n_trades: int
    an_n_trades: int
    nt_mean_gross_bps: float
    an_mean_gross_bps: float
    nt_total_gross_pct: float
    an_total_gross_pct: float

    @property
    def mean_diff_bps(self) -> float:
        return abs(self.nt_mean_gross_bps - self.an_mean_gross_bps)


def _run_nautilus(df, k: int, z_thr: float, hold: int, *, quiet: bool = True) -> list[float]:
    instrument = _zero_fees(TestInstrumentProvider.btcusdt_perp_binance())
    venue = instrument.id.venue
    engine = BacktestEngine(
        config=BacktestEngineConfig(trader_id="NT-MOM-001", logging=LoggingConfig(bypass_logging=quiet))
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,  # MARGIN: perp suporta SHORT
        base_currency=instrument.quote_currency,
        starting_balances=[Money(START_BALANCE, instrument.quote_currency)],
    )
    engine.add_instrument(instrument)
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    engine.add_data(BarDataWrangler(bar_type, instrument).process(df))
    strat = IntradayMomentum(
        IntradayMomentumConfig(
            instrument_id=str(instrument.id), bar_type=str(bar_type), k=k, z_thr=z_thr, hold=hold
        )
    )
    engine.add_strategy(strat)
    engine.run()
    out = list(strat.trade_returns)
    engine.dispose()
    return out


def validate(symbol: str, k: int = 15, z_thr: float = 1.5, hold: int = 10, *, months=None, quiet: bool = True) -> MomentumParity:
    df = load_symbol(symbol, months=months) if months else load_symbol(symbol)
    if df is None or df.empty:
        raise FileNotFoundError(f"sem klines 1min para {symbol}")
    df = df[["open", "high", "low", "close", "volume"]].astype(float).copy()

    nt = _run_nautilus(df, k, z_thr, hold, quiet=quiet)
    gross, _net_t, _net_m, _n = run_h1_config(df, k, z_thr, hold)  # analitico (vetorizado)

    nt_arr = np.asarray(nt, dtype=float)
    return MomentumParity(
        symbol=symbol, k=k, z_thr=z_thr, hold=hold, bars=len(df),
        nt_n_trades=nt_arr.size,
        an_n_trades=gross.size,
        nt_mean_gross_bps=float(nt_arr.mean() * 1e4) if nt_arr.size else 0.0,
        an_mean_gross_bps=float(gross.mean() * 1e4) if gross.size else 0.0,
        nt_total_gross_pct=float(nt_arr.sum() * 100.0) if nt_arr.size else 0.0,
        an_total_gross_pct=float(gross.sum() * 100.0) if gross.size else 0.0,
    )


def main() -> None:
    a = sys.argv[1:]
    symbol = a[0] if len(a) > 0 else "BTCUSDT"
    k = int(a[1]) if len(a) > 1 else 15
    z_thr = float(a[2]) if len(a) > 2 else 1.5
    hold = int(a[3]) if len(a) > 3 else 10
    months = ["2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]

    print("=== Nautilus (event-driven) x analitico (vetorizado) — H1 momentum ===\n")
    p = validate(symbol, k, z_thr, hold, months=months)
    print(f"[{p.symbol}] k={p.k} z_thr={p.z_thr} hold={p.hold}  ({p.bars} barras de 1min)")
    print(f"  trades : nautilus {p.nt_n_trades}  | analitico {p.an_n_trades}")
    print(f"  gross medio/trade : nautilus {p.nt_mean_gross_bps:+.3f} bps | analitico {p.an_mean_gross_bps:+.3f} bps")
    print(f"  gross total       : nautilus {p.nt_total_gross_pct:+.3f}%  | analitico {p.an_total_gross_pct:+.3f}%")
    print(f"  |dif| gross medio : {p.mean_diff_bps:.3f} bps")
    print("\nMotor event-driven reproduz o sinal de alpha real. Diferenca = amostragem de")
    print("execucao (analitico usa OPEN[t+1]; Nautilus casa no CLOSE), ~0 em 1min.")


if __name__ == "__main__":
    main()
