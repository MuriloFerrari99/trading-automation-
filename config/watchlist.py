"""Carregamento da watchlist (ativos monitorados + parametros por ativo).

A watchlist e configuravel via `config/watchlist.yaml`. Cada ativo pode ter
configuracao para uma ou mais estrategias, todas opcionais:

- trailing_stop_pct: percentual de trailing stop (Nivel 1).
- ladder: compras escalonadas em quedas (Nivel 1), com ancora e degraus.

Percentuais sao informados em pontos percentuais no YAML (ex: 10.0) e
convertidos aqui para fracao (0.10).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_WATCHLIST_PATH = Path("config/watchlist.yaml")


class LadderRung(BaseModel):
    """Um degrau da escada de compras: compre `qty` quando cair `drop_pct`."""

    # Fracao (0..1) de queda em relacao a ancora.
    drop_pct: Decimal = Field(..., gt=0, lt=1)
    qty: Decimal = Field(..., gt=0)


class LadderConfig(BaseModel):
    """Configuracao de Ladder Buys de um ativo."""

    # Preco de referencia das quedas. Se ausente, usa o primeiro preco
    # observado pela estrategia (persistido no state).
    anchor_price: Decimal | None = Field(default=None, gt=0)
    rungs: list[LadderRung]
    # Stop de invalidacao global (fracao). Se o preco cair stop_loss_pct abaixo
    # da ancora, a tese quebrou: liquida a escada inteira (doc 02 §3.3). None =
    # sem stop global (comportamento legado). Deve ser mais fundo que o ultimo degrau.
    stop_loss_pct: Decimal | None = Field(default=None, gt=0, lt=1)

    @field_validator("rungs")
    @classmethod
    def _non_empty_sorted(cls, v: list[LadderRung]) -> list[LadderRung]:
        if not v:
            raise ValueError("ladder.rungs nao pode ser vazio")
        # Ordena por profundidade da queda (mais raso primeiro) para um
        # comportamento deterministico e indices estaveis.
        return sorted(v, key=lambda r: r.drop_pct)


class WheelConfig(BaseModel):
    """Configuracao da Wheel Strategy (opcoes) de um ativo.

    A habilitacao real depende ainda do gate de elegibilidade (nivel de
    opcoes da conta + liquidez), verificado em tempo de execucao.
    """

    # Distancia OTM (fracao): vende put ~otm_pct abaixo do preco e covered call
    # ~otm_pct acima do preco de custo.
    otm_pct: Decimal = Field(..., gt=0, lt=1)
    contracts: int = Field(1, gt=0)  # 1 contrato = 100 acoes
    min_dte: int = Field(20, gt=0)
    max_dte: int = Field(45, gt=0)


class WatchlistItem(BaseModel):
    symbol: str
    # Classe do ativo: "equity" (default, respeita pregao) ou "crypto" (24/7).
    # Define se as ordens podem ser geradas fora do horario de pregao de acoes.
    asset_class: str = "equity"
    # Tick size (passo minimo de preco) e lote minimo (passo de quantidade) do
    # ativo. None => sem arredondamento (acoes inteiras / comportamento legado).
    tick_size: Decimal | None = Field(default=None, gt=0)
    lot_size: Decimal | None = Field(default=None, gt=0)
    fractional: bool = False  # permite quantidade fracionada (cripto, acoes frac.)
    trailing_stop_pct: Decimal | None = Field(default=None, gt=0, lt=1)
    ladder: LadderConfig | None = None
    wheel: WheelConfig | None = None

    @field_validator("symbol")
    @classmethod
    def _norm(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("asset_class")
    @classmethod
    def _norm_asset_class(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("equity", "crypto"):
            raise ValueError(f"asset_class invalido: {v!r} (use 'equity' ou 'crypto')")
        return v

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"


class Watchlist(BaseModel):
    items: list[WatchlistItem]

    def symbols(self) -> list[str]:
        return [i.symbol for i in self.items]

    def get(self, symbol: str) -> WatchlistItem | None:
        symbol = symbol.upper()
        return next((i for i in self.items if i.symbol == symbol), None)

    def has_crypto(self) -> bool:
        """True se algum ativo opera 24/7 (cripto) — o Monitor entao nao deve
        pular ciclos quando o pregao de acoes estiver fechado."""
        return any(i.is_crypto for i in self.items)


def _pct_to_fraction(value) -> Decimal:
    return Decimal(str(value)) / Decimal(100)


def _parse_ladder(raw: dict) -> LadderConfig:
    rungs = [
        LadderRung(drop_pct=_pct_to_fraction(r["drop_pct"]), qty=Decimal(str(r["qty"])))
        for r in raw.get("rungs", [])
    ]
    anchor = raw.get("anchor_price")
    stop = raw.get("stop_loss_pct")
    return LadderConfig(
        anchor_price=Decimal(str(anchor)) if anchor is not None else None,
        rungs=rungs,
        stop_loss_pct=_pct_to_fraction(stop) if stop is not None else None,
    )


def _parse_wheel(raw: dict) -> WheelConfig:
    return WheelConfig(
        otm_pct=_pct_to_fraction(raw["otm_pct"]),
        contracts=int(raw.get("contracts", 1)),
        min_dte=int(raw.get("min_dte", 20)),
        max_dte=int(raw.get("max_dte", 45)),
    )


def load_watchlist(path: Path | str = DEFAULT_WATCHLIST_PATH) -> Watchlist:
    """Le e valida a watchlist do YAML, convertendo % para fracao."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = []
    for entry in raw.get("symbols", []):
        trailing = entry.get("trailing_stop_pct")
        ladder_raw = entry.get("ladder")
        wheel_raw = entry.get("wheel")
        tick = entry.get("tick_size")
        lot = entry.get("lot_size")
        items.append(
            WatchlistItem(
                symbol=entry["symbol"],
                asset_class=entry.get("asset_class", "equity"),
                tick_size=Decimal(str(tick)) if tick is not None else None,
                lot_size=Decimal(str(lot)) if lot is not None else None,
                fractional=bool(entry.get("fractional", False)),
                trailing_stop_pct=_pct_to_fraction(trailing) if trailing is not None else None,
                ladder=_parse_ladder(ladder_raw) if ladder_raw else None,
                wheel=_parse_wheel(wheel_raw) if wheel_raw else None,
            )
        )
    if not items:
        raise ValueError(f"Watchlist vazia ou invalida em {path}")
    return Watchlist(items=items)
