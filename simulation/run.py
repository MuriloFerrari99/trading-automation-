"""CLI do sweep de simulacao.

Uso:
    uv run python -m simulation.run                 # universo padrao, ~10k+ janelas
    uv run python -m simulation.run --window 120 --step 6
    uv run python -m simulation.run --max-runs 2000 # amostra rapida
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Universo padrao: acoes liquidas de varios setores + pares de cripto (a
# "paridade entre moedas" nativa da Alpaca). Ajustavel por --stocks.
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "NFLX",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "XOM", "CVX", "KO", "PEP",
    "PG", "JNJ", "PFE", "MRK", "UNH", "HD", "WMT", "COST", "DIS", "NKE",
    "BA", "CAT", "GE", "MMM", "T", "VZ", "ORCL", "CRM", "ADBE", "QCOM",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT",
]
DEFAULT_CRYPTO = ["BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "DOGE/USD", "AVAX/USD"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep de simulacao (dados reais Alpaca)")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--window", type=int, default=160)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--no-crypto", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--report-file", default="data/sim_report.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    log = logging.getLogger("simulation.run")

    from simulation.data import fetch_daily, to_ohlc_lists
    from simulation.montecarlo import format_report, run_sweep, summarize

    crypto = [] if args.no_crypto else DEFAULT_CRYPTO
    log.info("Buscando dados (%d acoes, %d cripto, %dy)...", len(DEFAULT_STOCKS), len(crypto), args.years)
    frames = fetch_daily(DEFAULT_STOCKS, crypto, years=args.years, force=args.force_fetch)
    data = {sym: to_ohlc_lists(df) for sym, df in frames.items()}
    log.info("Dados carregados: %d simbolos.", len(data))

    results = run_sweep(
        data, window_len=args.window, step=args.step,
        commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
        max_runs=args.max_runs,
    )
    report = summarize(results)
    text = format_report(report)
    print(text)

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("Relatorio salvo em %s (%d backtests).", out, report.total_runs)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
