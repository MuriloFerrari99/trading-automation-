"""Testes do carregamento/validacao da watchlist."""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.watchlist import load_watchlist

YAML = """
symbols:
  - symbol: aapl
    trailing_stop_pct: 10.0
  - symbol: MSFT
    ladder:
      anchor_price: 400.0
      rungs:
        - drop_pct: 30.0
          qty: 10
        - drop_pct: 20.0
          qty: 5
"""


def test_loads_and_converts(tmp_path):
    p = tmp_path / "watchlist.yaml"
    p.write_text(YAML, encoding="utf-8")
    wl = load_watchlist(p)

    aapl = wl.get("AAPL")
    assert aapl is not None
    assert aapl.trailing_stop_pct == Decimal("0.1")  # 10% -> 0.10
    assert aapl.ladder is None

    msft = wl.get("MSFT")
    assert msft is not None
    assert msft.trailing_stop_pct is None
    assert msft.ladder is not None
    assert msft.ladder.anchor_price == Decimal("400")
    # rungs ordenados por profundidade (raso primeiro): 20% antes de 30%
    assert [r.drop_pct for r in msft.ladder.rungs] == [Decimal("0.2"), Decimal("0.3")]


def test_empty_watchlist_rejected(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("symbols: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_watchlist(p)
