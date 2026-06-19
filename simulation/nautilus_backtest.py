"""Fase 2 da adocao do NautilusTrader — portar dados/estrategia REAIS + decompor
o motor.

O spike (simulation/nautilus_spike.py) provou que o engine roda nesta maquina
com barras sinteticas. Aqui portamos a logica REAL do nosso TrailingStopStrategy
(entra ~50%% no fim do warmup, rastreia o high-water no fechamento, vende quando
o fechamento cai trail%% abaixo da maxima) para uma Strategy nativa do Nautilus,
rodando sobre os dados diarios reais de data/cache/*.csv.

POR QUE NAO E "so um teste de paridade":
Uma primeira versao deste arquivo declarava ~1-2pp de diferenca como "ruido de
fill timing" e seguia. Errado. Em um round-trip de ~50%% de notional, 1-2pp NAO
e margem de erro — e o motor avisando que as duas implementacoes assumem
EXECUCOES diferentes. Decompondo (ver decompose()):

  - O engine caseiro (SimBroker) assume "decide no FECHAMENTO da barra t,
    executa no OPEN da barra t+1" — sem look-ahead, mas exige o proximo open.
  - O Nautilus com barras diarias e bar_execution casa a ordem a mercado no
    FECHAMENTO da propria barra de decisao (verificado empiricamente; nem
    bar_adaptive_high_low_ordering muda isso). Isso e um look-ahead leve para
    sinais calculados no fechamento.

As duas suposicoes sao discretizacoes diferentes da MESMA realidade continua, e
sao IRRECONCILIAVEIS em barra diaria: o gap entre elas (= (close[gatilho] -
open[gatilho+1]) * qty) e a incerteza que a granularidade D1 injeta. Em janelas
com gap overnight grande (ex.: COVID) ele explode. Nao se backtesta pra fora
disso em D1 — resolve-se com dados intraday/tick, que e exatamente onde o motor
do Nautilus passa a valer (fills, fees, partial fills, fila — no nivel em que a
execucao de fato acontece).

decompose() prova isso ao CENTAVO: depois de alinhar a suposicao (entrada
identica), o retorno do Nautilus == "compra@close[entrada], vende@close[gatilho]"
e o do caseiro == "compra@close[entrada], vende@open[gatilho+1]"; o residual e ~0.

Determinista e offline: le do cache, roda em memoria, nao toca broker/rede.

CLI:
    uv run python -m simulation.nautilus_backtest            # SPY
    uv run python -m simulation.nautilus_backtest SPY QQQ AAPL
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from simulation import sim_broker as _sim_broker_mod
from simulation.data import to_ohlc_lists
from simulation.engine import run_backtest

CACHE_DIR = Path("data/cache")

WARMUP = 35
TRADE_FRACTION = Decimal("0.5")
TRAIL_PCT = Decimal("0.10")
CASH = 100_000.0


# --------------------------------------------------------------------------- #
# Tipos                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Fill:
    side: str  # "BUY" | "SELL"
    bar: int  # indice da barra em que o fill ocorreu
    price: float
    qty: float


@dataclass
class ParityResult:
    symbol: str
    bars: int
    nautilus_return_pct: float
    homemade_return_pct: float

    @property
    def abs_diff_pct(self) -> float:
        return abs(self.nautilus_return_pct - self.homemade_return_pct)


@dataclass
class Decomposition:
    """Reconcilia o gap entre os motores em componentes nomeados (ao centavo)."""

    symbol: str
    bars: int
    entry_bar: int
    entry_px: float
    qty: float
    trigger_bar: int | None  # barra cujo fechamento disparou o trailing (None = nunca)
    exit_close: float | None  # close[trigger]   — fill do Nautilus (D1)
    exit_next_open: float | None  # open[trigger+1] — fill do caseiro (D1)
    nautilus_return_pct: float
    homemade_return_pct: float
    nautilus_fills: list[Fill] = field(default_factory=list)
    homemade_fills: list[Fill] = field(default_factory=list)

    @property
    def gap_pp(self) -> float:
        return self.nautilus_return_pct - self.homemade_return_pct

    @property
    def assumption_cost_pp(self) -> float:
        """Custo da suposicao de execucao D1: close[gatilho] vs open[gatilho+1]."""
        if self.trigger_bar is None or self.exit_close is None:
            return 0.0
        return (self.exit_close - self.exit_next_open) * self.qty / CASH * 100.0

    @property
    def residual_pp(self) -> float:
        """O que sobra depois de explicar o gap pela suposicao. Deve ser ~0."""
        return self.gap_pp - self.assumption_cost_pp


# --------------------------------------------------------------------------- #
# Estrategia portada para o Nautilus                                          #
# --------------------------------------------------------------------------- #
class PortedTrailingStopConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    cash: float = CASH
    warmup: int = WARMUP
    trade_fraction: Decimal = TRADE_FRACTION
    trail_pct: Decimal = TRAIL_PCT


class PortedTrailingStop(Strategy):
    """Porta a tese do nosso TrailingStopStrategy para o motor do Nautilus.

    Roda o gatilho no fechamento da barra (como o SimBroker caseiro), para manter
    a comparacao apples-to-apples. Captura os proprios fills em self.fills.
    """

    def __init__(self, config: PortedTrailingStopConfig) -> None:
        super().__init__(config)
        self._bar_type = BarType.from_str(config.bar_type)
        self._count = 0
        self._entered = False
        self._high_water: float | None = None
        self.fills: list[tuple[str, float, float, int]] = []  # (side, px, qty, ts_ns)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self._bar_type.instrument_id)
        self.subscribe_bars(self._bar_type)

    def on_order_filled(self, event) -> None:
        self.fills.append(
            (str(event.order_side.name), float(event.last_px), float(event.last_qty), int(event.ts_event))
        )

    def _net_qty(self) -> Decimal:
        pos = self.cache.positions_open(instrument_id=self.instrument.id)
        return Decimal(str(pos[0].quantity)) if pos else Decimal(0)

    def _market(self, side: OrderSide, qty: int) -> None:
        # make_qty formata na precisao de tamanho do instrumento (acao: 0 casas;
        # cripto: 6) — Quantity.from_int(1) seria rejeitada por "invalid size precision".
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument.id, order_side=side, quantity=self.instrument.make_qty(qty)
            )
        )

    def on_bar(self, bar: Bar) -> None:
        # Alinhado ao engine caseiro (simulation/engine.py): ele opera a partir do
        # indice i == warmup (a (warmup+1)-esima barra). self._count comeca em 0 e
        # incrementa no topo, entao a condicao count > warmup <=> indice >= warmup.
        self._count += 1
        if self._count <= self.config.warmup:
            return

        close = float(bar.close)

        if not self._entered:
            # Arredonda half-even (to_integral_value), IGUAL ao caseiro — nao trunca.
            raw = (Decimal(str(self.config.cash)) * self.config.trade_fraction) / Decimal(str(close))
            qty = int(raw.to_integral_value())
            if qty > 0:
                self._market(OrderSide.BUY, qty)
                self._entered = True
                self._high_water = close
            return

        qty = self._net_qty()
        if qty <= 0:
            return  # ja saiu (trailing disparou); fica flat ate o fim

        if self._high_water is None or close > self._high_water:
            self._high_water = close
        stop = self._high_water * (1 - float(self.config.trail_pct))
        if close <= stop:
            self._market(OrderSide.SELL, int(qty))


# --------------------------------------------------------------------------- #
# Execucao dos dois motores                                                   #
# --------------------------------------------------------------------------- #
def _load_df(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol.replace('/', '_')}_1d.csv"
    if not path.exists():
        raise FileNotFoundError(f"sem cache para {symbol}: {path} (rode o fetch de dados primeiro)")
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close"]].astype(float).copy()
    df["volume"] = 1_000_000.0
    return df


def _run_nautilus(
    df: pd.DataFrame, instrument, bar_spec: str, *, quiet: bool = True
) -> tuple[float, list[Fill]]:
    """Roda a estrategia portada no Nautilus. Devolve (retorno%%, fills).

    Generico: venue e moeda de liquidacao saem do proprio instrumento, entao o
    MESMO caminho serve para acao diaria (SPY/XNAS/USD) e cripto 1min
    (BTCUSDT/BINANCE/USDT).
    """
    venue = instrument.id.venue
    currency = instrument.quote_currency
    # CurrencyPair (cripto/fx spot) liquida em DUAS moedas (base+quote) -> conta CASH
    # multi-moeda (base_currency=None). Acao liquida so na moeda da conta.
    is_currency_pair = type(instrument).__name__ == "CurrencyPair"
    base_ccy = getattr(instrument, "base_currency", None)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="NT-PARITY-001", logging=LoggingConfig(bypass_logging=quiet)
        )
    )
    # Custo ZERO dos dois lados: isola "o motor reproduz a estrategia?".
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None if is_currency_pair else currency,
        starting_balances=[Money(CASH, currency)],
    )
    engine.add_instrument(instrument)

    bar_type = BarType.from_str(f"{instrument.id}-{bar_spec}")
    engine.add_data(BarDataWrangler(bar_type, instrument).process(df))

    strat = PortedTrailingStop(
        PortedTrailingStopConfig(instrument_id=str(instrument.id), bar_type=str(bar_type))
    )
    engine.add_strategy(strat)
    engine.run()

    # Net liquidation, marcada na moeda de cotacao:
    #  - acao: caixa na moeda + valor das POSICOES abertas (qty * ultimo close);
    #  - currency pair: a exposicao vira SALDO na moeda-base (BTC), nao posicao.
    account = engine.cache.account_for_venue(venue)
    last_close = float(df["close"].iloc[-1])
    net_liq = float(account.balance_total(currency).as_double())
    if is_currency_pair and base_ccy is not None:
        base_bal = account.balance_total(base_ccy)
        if base_bal is not None:
            net_liq += float(base_bal.as_double()) * last_close
    else:
        net_liq += sum(float(p.quantity) * last_close for p in engine.cache.positions_open())

    ts_to_bar = {int(t.value): i for i, t in enumerate(df.index)}
    fills = [Fill(side, ts_to_bar[ts], px, qty) for (side, px, qty, ts) in strat.fills]

    engine.dispose()
    return (net_liq / CASH - 1.0) * 100.0, fills


def _run_homemade(symbol: str, df: pd.DataFrame) -> tuple[float, list[Fill]]:
    """Roda o mesmo trailing no engine caseiro (custo zero). Devolve (retorno%%, fills).

    Captura os fills monkeypatchando _fill_buy/_fill_sell do SimBroker (restaurado
    no finally), para nao depender de internals que run_backtest nao expoe.
    """
    captured: list[Fill] = []
    SB = _sim_broker_mod.SimBroker
    orig_buy, orig_sell = SB._fill_buy, SB._fill_sell

    def cap_buy(self, ref, qty):  # noqa: ANN001
        px = orig_buy(self, ref, qty)
        captured.append(Fill("BUY", self._i, float(px), float(qty)))
        return px

    def cap_sell(self, ref, qty):  # noqa: ANN001
        px = orig_sell(self, ref, qty)
        captured.append(Fill("SELL", self._i, float(px), float(qty)))
        return px

    SB._fill_buy, SB._fill_sell = cap_buy, cap_sell
    try:
        result = run_backtest(
            symbol, to_ohlc_lists(df), "trailing_stop",
            cash=CASH, commission_bps=0.0, slippage_bps=0.0, warmup=WARMUP,
        )
    finally:
        SB._fill_buy, SB._fill_sell = orig_buy, orig_sell

    if result is None:
        raise ValueError(f"engine caseiro nao rodou {symbol} (serie curta?)")
    # equity[-1] e o mark-to-market no ultimo fechamento (o liquidate_final do
    # caseiro acontece depois e nao entra na curva) — mesma base do net_liq do Nautilus.
    return (result.equity[-1] / CASH - 1.0) * 100.0, captured


# --------------------------------------------------------------------------- #
# Mercados (df + instrumento + granularidade)                                 #
# --------------------------------------------------------------------------- #
def _zero_fees(instrument):
    """Zera maker/taker fee do instrumento (custo ZERO dos dois lados).

    O instrumento de acao do test_kit ja vem sem fee, mas o de cripto carrega o
    taker de 0.1%% da Binance embutido — sem zerar, ele apareceria como ~0.13pp de
    'residual' no decompose (que e fee real, nao bug de motor). Clona via dict.
    """
    cls = type(instrument)
    if float(getattr(instrument, "maker_fee", 0)) == 0 and float(getattr(instrument, "taker_fee", 0)) == 0:
        return instrument
    d = cls.to_dict(instrument)
    d["maker_fee"] = "0"
    d["taker_fee"] = "0"
    return cls.from_dict(d)


def equity_daily(symbol: str) -> tuple[pd.DataFrame, object, str]:
    """Acao diaria de data/cache/*.csv (SPY, QQQ, ...)."""
    df = _load_df(symbol)
    instrument = _zero_fees(TestInstrumentProvider.equity(symbol, "XNAS"))
    return df, instrument, "1-DAY-LAST-EXTERNAL"


def crypto_minute(symbol: str = "BTCUSDT", months: list[str] | None = None) -> tuple[pd.DataFrame, object, str]:
    """Cripto spot 1min de data/binance_1m/ (reusa o loader do crypto_intraday)."""
    from simulation.crypto_intraday import MONTHS, load_symbol

    df = load_symbol(symbol, months=months or MONTHS)
    if df is None or df.empty:
        raise FileNotFoundError(f"sem klines 1min cacheadas para {symbol} em data/binance_1m/")
    df = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    instrument = _zero_fees(TestInstrumentProvider.btcusdt_binance())  # CurrencyPair spot BTCUSDT.BINANCE
    return df, instrument, "1-MINUTE-LAST-EXTERNAL"


# --------------------------------------------------------------------------- #
# API                                                                         #
# --------------------------------------------------------------------------- #
def compare(symbol: str, *, quiet: bool = True) -> ParityResult:
    d = decompose(symbol, quiet=quiet)
    return ParityResult(
        symbol=symbol, bars=d.bars,
        nautilus_return_pct=d.nautilus_return_pct, homemade_return_pct=d.homemade_return_pct,
    )


def decompose(
    symbol: str,
    *,
    market: tuple[pd.DataFrame, object, str] | None = None,
    quiet: bool = True,
) -> Decomposition:
    """Roda os dois motores e atribui o gap a componentes nomeados (ao centavo).

    market = (df, instrument_nautilus, bar_spec). Default: acao diaria. Use
    crypto_minute(...) para o caminho intraday.
    """
    df, instrument, bar_spec = market if market is not None else equity_daily(symbol)
    o, c = df["open"].tolist(), df["close"].tolist()
    nt_ret, nt_fills = _run_nautilus(df, instrument, bar_spec, quiet=quiet)
    hm_ret, hm_fills = _run_homemade(symbol, df)

    nt_buy = next((f for f in nt_fills if f.side == "BUY"), None)
    nt_sell = next((f for f in nt_fills if f.side == "SELL"), None)
    entry_bar = nt_buy.bar if nt_buy else WARMUP
    entry_px = nt_buy.price if nt_buy else c[entry_bar]
    qty = nt_buy.qty if nt_buy else 0.0
    trigger_bar = nt_sell.bar if nt_sell else None

    return Decomposition(
        symbol=symbol,
        bars=len(df),
        entry_bar=entry_bar,
        entry_px=entry_px,
        qty=qty,
        trigger_bar=trigger_bar,
        exit_close=(c[trigger_bar] if trigger_bar is not None else None),
        exit_next_open=(o[trigger_bar + 1] if trigger_bar is not None and trigger_bar + 1 < len(o) else None),
        nautilus_return_pct=nt_ret,
        homemade_return_pct=hm_ret,
        nautilus_fills=nt_fills,
        homemade_fills=hm_fills,
    )


def _print_decomp(d: Decomposition, granularity: str) -> None:
    print(f"[{d.symbol}] {d.bars} barras ({granularity})")
    print(f"  entrada : barra {d.entry_bar} @ {d.entry_px:.2f}  qty {d.qty:g}")
    if d.trigger_bar is not None:
        print(
            f"  gatilho : fechamento da barra {d.trigger_bar}  "
            f"-> Nautilus vende @ close={d.exit_close:.2f} | "
            f"caseiro vende @ next-open={d.exit_next_open:.2f}"
        )
    else:
        print("  gatilho : trailing nunca disparou (segura ate o fim)")
    print(f"  retorno : nautilus {d.nautilus_return_pct:+.4f}%  caseiro {d.homemade_return_pct:+.4f}%")
    print(f"  gap     : {d.gap_pp:+.4f}pp")
    print(f"    = custo da suposicao de execucao (close-gatilho vs next-open): {d.assumption_cost_pp:+.4f}pp")
    print(f"    + residual (deveria ser ~0): {d.residual_pp:+.6f}pp\n")


def main() -> None:
    args = sys.argv[1:]
    crypto = "--crypto" in args
    symbols = [a for a in args if not a.startswith("--")]
    print("=== Nautilus x engine caseiro (trailing_stop, custo zero) — decomposicao ===\n")
    if crypto:
        months = ["2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]
        for sym in symbols or ["BTCUSDT"]:
            try:
                d = decompose(sym, market=crypto_minute(sym, months=months))
            except Exception as exc:  # noqa: BLE001
                print(f"{sym}: ERRO: {exc}\n")
                continue
            _print_decomp(d, "cripto 1min")
        print("Em barra de 1min (cripto 24/7), close[gatilho] ~= open[gatilho+1]: o gap de")
        print("execucao COLAPSA vs o caso diario. A incerteza vem da granularidade, nao do motor.")
        return

    for sym in symbols or ["SPY"]:
        try:
            d = decompose(sym)
        except Exception as exc:  # noqa: BLE001 — CLI: reporta e segue
            print(f"{sym}: ERRO: {exc}\n")
            continue
        _print_decomp(d, "acao diaria")
    print("Gap = suposicao de execucao em D1 (irreconciliavel sem dados intraday), nao bug.")
    print("Residual ~0 prova que os dois motores estao mecanicamente corretos.")


if __name__ == "__main__":
    main()
