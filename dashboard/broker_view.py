"""Enriquecimento opcional com dados ao vivo da corretora.

Isolado de proposito: o dashboard funciona 100% offline a partir do SQLite. Este
modulo so e usado quando o usuario quer ver equity/cash atuais e P&L NAO
realizado das posicoes abertas (preco atual x preco medio de entrada).

Tudo aqui degrada com graca: se faltar credencial, a rede cair, ou o SDK nao
estiver disponivel, retornamos `(None, None, "error: ...")` e o dashboard segue
mostrando os dados historicos do banco.
"""

from __future__ import annotations

import logging
from decimal import Decimal

logger = logging.getLogger("dashboard.broker")


def _f(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch_live_view() -> tuple[dict | None, list[dict] | None, str]:
    """Retorna (account, positions, status).

    - account: {equity, cash, buying_power, options_level} ou None
    - positions: lista de dicts com P&L nao-realizado, ou None
    - status: "live" em sucesso, "error: <motivo>" em falha
    """
    try:
        from config.settings import get_settings
        from main import build_broker

        settings = get_settings()
        broker = build_broker(settings)
        acct = broker.get_account()
        account = {
            "equity": _f(acct.equity),
            "cash": _f(acct.cash),
            "buying_power": _f(acct.buying_power),
            "options_level": acct.options_level,
        }

        positions: list[dict] = []
        for p in broker.get_positions():
            qty = p.qty if isinstance(p.qty, Decimal) else Decimal(str(p.qty))
            avg = p.avg_entry_price
            cur = p.current_price
            mkt_val = (qty * cur) if cur is not None else None
            cost = qty * avg
            upnl = (mkt_val - cost) if mkt_val is not None else None
            upnl_pct = (
                float(upnl / cost) if (upnl is not None and cost != 0) else None
            )
            positions.append(
                {
                    "symbol": p.symbol,
                    "qty": _f(qty),
                    "avg_entry_price": _f(avg),
                    "current_price": _f(cur),
                    "market_value": _f(mkt_val),
                    "unrealized_pnl": _f(upnl),
                    "unrealized_pnl_pct": upnl_pct,
                }
            )
        return account, positions, "live"
    except Exception as exc:  # noqa: BLE001 - degradacao intencional
        logger.warning("Broker ao vivo indisponivel: %s", exc)
        return None, None, f"error: {type(exc).__name__}"
