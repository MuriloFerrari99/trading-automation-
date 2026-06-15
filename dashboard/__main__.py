"""Entrypoint do dashboard: `uv run python -m dashboard`."""

from __future__ import annotations

import argparse
import logging

from dashboard.server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Painel de monitoramento de trades (le data/trading.sqlite)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="host (padrao: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="porta (padrao: 8787)")
    parser.add_argument(
        "--db", default="data/trading.sqlite", help="caminho do SQLite"
    )
    parser.add_argument(
        "--no-broker",
        action="store_true",
        help="nao consultar a corretora ao vivo (so dados do SQLite, offline)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    serve(
        args.db,
        host=args.host,
        port=args.port,
        use_broker=not args.no_broker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
