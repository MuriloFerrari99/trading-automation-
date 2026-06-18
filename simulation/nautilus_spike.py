"""Spike NautilusTrader — prova de vida do engine event-driven, ISOLADO.

Fase 1 da adocao do Nautilus (radar de stack): rodar uma estrategia simples no
BacktestEngine do Nautilus EM PARALELO ao engine atual, sem desligar nada. O
objetivo aqui nao e estrategia boa — e provar que o engine roda nesta maquina
(py3.11/arm64), processa barras, casa ordens e fecha conta. A partir daqui da
para portar dados/estrategias reais incrementalmente.

Determinista (seed fixa) e sem efeito externo: gera barras sinteticas, roda em
memoria, nao toca broker, banco nem rede.

CLI:
    uv run python -m simulation.nautilus_spike
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def _synthetic_bars_df(n: int, seed: int) -> pd.DataFrame:
    """OHLCV sintetico de 1 minuto em torno de ~1.10 (perfil EUR/USD)."""
    rng = np.random.default_rng(seed)
    # Sem drift e com vol suficiente p/ os EMAs (10/20) cruzarem varias vezes.
    rets = 0.0015 * rng.standard_normal(n)
    close = 1.10 * np.cumprod(1 + rets)
    open_ = np.concatenate([[1.10], close[:-1]])
    spread = np.abs(0.0003 * rng.standard_normal(n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def run_spike(n_bars: int = 500, seed: int = 7, *, quiet: bool = True) -> dict:
    """Roda o backtest e devolve um resumo (ordens, fills, posicoes, conta)."""
    venue = Venue("SIM")
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="SPIKE-001",
            logging=LoggingConfig(bypass_logging=quiet),
        )
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(1_000_000, USD)],
    )

    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue=venue)
    engine.add_instrument(instrument)

    # LAST (nao BID): com barras de um lado so, o matching por barra sintetiza o
    # mercado a partir do preco de LAST. BID sozinho gera "no market" (sem ask).
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(_synthetic_bars_df(n_bars, seed))
    engine.add_data(bars)

    engine.add_strategy(
        EMACross(
            EMACrossConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=Decimal("100000"),
                fast_ema_period=10,
                slow_ema_period=20,
            )
        )
    )

    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    account = engine.trader.generate_account_report(venue)
    resumo = {
        "bars": len(bars),
        "fills": 0 if fills is None else len(fills),
        "positions": 0 if positions is None else len(positions),
        "account_rows": 0 if account is None else len(account),
    }
    engine.dispose()
    return resumo


def main() -> None:
    r = run_spike(quiet=True)
    print("=== Nautilus spike (EMACross, EUR/USD sintetico) ===")
    print(f"barras processadas : {r['bars']}")
    print(f"ordens executadas  : {r['fills']}")
    print(f"posicoes           : {r['positions']}")
    print(f"linhas de conta    : {r['account_rows']}")
    print("OK — engine Nautilus roda nesta maquina. Proximo: portar dados/estrategia reais.")


if __name__ == "__main__":
    main()
