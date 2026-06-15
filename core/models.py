"""Modelos de dominio compartilhados entre agentes, estrategias e broker.

Usamos Pydantic v2 para validacao em runtime nas fronteiras do sistema
(intencoes de ordem geradas por estrategias, sinais externos, posicoes).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OptionType(str, Enum):
    PUT = "put"
    CALL = "call"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OrderIntent(BaseModel):
    """Intencao de ordem produzida por uma estrategia/Planejador.

    E o contrato entre quem decide (Planejador/estrategia) e quem executa
    (Executor). Ainda nao e uma ordem na corretora: passa por validacao,
    kill switch e guard antes de virar uma ordem real.
    """

    symbol: str
    side: OrderSide
    qty: Decimal = Field(..., gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.DAY
    # Estrategia que originou a intencao (auditoria/rastreabilidade).
    strategy: str = "unknown"
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol nao pode ser vazio")
        return v

    @field_validator("limit_price")
    @classmethod
    def _limit_required(cls, v: Decimal | None, info) -> Decimal | None:
        order_type = info.data.get("order_type")
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and v is None:
            raise ValueError(f"limit_price obrigatorio para order_type={order_type}")
        return v

    @field_validator("stop_price")
    @classmethod
    def _stop_required(cls, v: Decimal | None, info) -> Decimal | None:
        order_type = info.data.get("order_type")
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and v is None:
            raise ValueError(f"stop_price obrigatorio para order_type={order_type}")
        return v


class Position(BaseModel):
    """Posicao atual em um ativo, conforme reportado pela corretora."""

    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    current_price: Decimal | None = None

    @property
    def market_value(self) -> Decimal | None:
        if self.current_price is None:
            return None
        return self.qty * self.current_price


class Signal(BaseModel):
    """Sinal/sugestao gerado por um SignalProvider (Nivel 2).

    Sinais NUNCA executam automaticamente: sao apenas input para o Planejador.
    """

    symbol: str
    side: OrderSide
    source: str  # ex: "congress", "smart_money"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    note: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class OptionContract(BaseModel):
    """Contrato de opcao selecionado na corretora."""

    # Simbolo OCC do contrato (ex: AAPL250117P00190000), quando disponivel.
    occ_symbol: str
    underlying: str
    option_type: OptionType
    strike: Decimal = Field(..., gt=0)
    expiration: date

    @field_validator("underlying")
    @classmethod
    def _norm_underlying(cls, v: str) -> str:
        return v.strip().upper()


class OptionOrderIntent(BaseModel):
    """Intencao de ordem de OPCOES (Nivel 3 — Wheel Strategy).

    Distinta de OrderIntent (acoes): o Executor despacha por tipo. Como a Wheel
    so vende premio (cash-secured puts e covered calls), o lado e sempre SELL
    no MVP, mas o campo e explicito para clareza/auditoria.
    """

    contract: OptionContract
    side: OrderSide = OrderSide.SELL
    qty: Decimal = Field(..., gt=0)  # numero de contratos (1 contrato = 100 acoes)
    strategy: str = "wheel"
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def underlying(self) -> str:
        return self.contract.underlying


class OrderResult(BaseModel):
    """Resultado da submissao de uma ordem na corretora."""

    broker_order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    filled_qty: Decimal = Decimal(0)
    filled_avg_price: Decimal | None = None
    status: str = "accepted"
    submitted_at: datetime = Field(default_factory=_utcnow)
