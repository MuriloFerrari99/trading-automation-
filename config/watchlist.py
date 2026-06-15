"""Carregamento da watchlist (ativos monitorados + parametros por ativo).

A watchlist e configuravel via `config/watchlist.yaml`. Cada item traz o
percentual de trailing stop especifico do ativo.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_WATCHLIST_PATH = Path("config/watchlist.yaml")


class WatchlistItem(BaseModel):
    symbol: str
    # Fracao (0..1). No YAML o valor e informado em pontos percentuais (ex 10.0)
    # e convertido aqui para fracao (0.10).
    trailing_stop_pct: Decimal = Field(..., gt=0, lt=1)

    @field_validator("symbol")
    @classmethod
    def _norm(cls, v: str) -> str:
        return v.strip().upper()


class Watchlist(BaseModel):
    items: list[WatchlistItem]

    def symbols(self) -> list[str]:
        return [i.symbol for i in self.items]

    def get(self, symbol: str) -> WatchlistItem | None:
        symbol = symbol.upper()
        return next((i for i in self.items if i.symbol == symbol), None)


def load_watchlist(path: Path | str = DEFAULT_WATCHLIST_PATH) -> Watchlist:
    """Le e valida a watchlist do YAML, convertendo % para fracao."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = []
    for entry in raw.get("symbols", []):
        pct_points = Decimal(str(entry["trailing_stop_pct"]))
        items.append(
            WatchlistItem(
                symbol=entry["symbol"],
                trailing_stop_pct=pct_points / Decimal(100),
            )
        )
    if not items:
        raise ValueError(f"Watchlist vazia ou invalida em {path}")
    return Watchlist(items=items)
