"""Modelos de dominio compartilhados entre agentes, estrategias e broker.

Usamos Pydantic v2 para validacao em runtime nas fronteiras do sistema
(intencoes de ordem geradas por estrategias, sinais externos, posicoes).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from core.idempotency import make_client_order_id


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
    TRAILING_STOP = "trailing_stop"


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
    # Trailing stop nativo (Alpaca): percentual de trail. Quando definido com
    # order_type=TRAILING_STOP, o broker rastreia o high-water e move o stop.
    trail_percent: Decimal | None = Field(default=None, gt=0)
    # Estrategia que originou a intencao (auditoria/rastreabilidade).
    strategy: str = "unknown"
    created_at: datetime = Field(default_factory=_utcnow)
    # client_order_id determinístico (idempotencia). Calculado do evento de
    # decisao se nao informado. Ver core/idempotency.py.
    client_order_id: str | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol nao pode ser vazio")
        return v

    @model_validator(mode="after")
    def _ensure_client_order_id(self) -> "OrderIntent":
        if self.client_order_id is None:
            self.client_order_id = make_client_order_id(
                self.strategy, self.symbol, self.side.value, self.created_at.isoformat()
            )
        return self

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
    # MARKET (default) ou LIMIT. Em opcoes ilíquidas, prefira LIMIT p/ controlar
    # o premio e evitar slippage; o limit_price e o premio-alvo POR ACAO.
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = Field(default=None, gt=0)
    strategy: str = "wheel"
    created_at: datetime = Field(default_factory=_utcnow)
    client_order_id: str | None = None

    @property
    def underlying(self) -> str:
        return self.contract.underlying

    @model_validator(mode="after")
    def _limit_requires_price(self) -> "OptionOrderIntent":
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price obrigatorio para order_type=LIMIT em opcoes")
        return self

    @model_validator(mode="after")
    def _ensure_client_order_id(self) -> "OptionOrderIntent":
        if self.client_order_id is None:
            self.client_order_id = make_client_order_id(
                self.strategy,
                self.contract.occ_symbol,
                self.side.value,
                self.created_at.isoformat(),
            )
        return self



def is_crypto_symbol(symbol: str) -> bool:
    """Cripto na Alpaca usa par com barra (ex.: 'BTC/USD'); acoes/ETFs nao tem '/'.

    Unica fonte de verdade para discriminar o venue de dados (stock vs crypto) e
    o pool de poder de compra (marginavel vs nao-marginavel). Duplicado antes em
    broker/alpaca_broker.py e agents/executor.py — centralizado aqui para evitar
    dessincronizacao silenciosa entre os dois gates."""
    return "/" in symbol


class OrderResult(BaseModel):
    """Resultado da submissao de uma ordem na corretora."""

    broker_order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    filled_qty: Decimal = Decimal(0)
    filled_avg_price: Decimal | None = None
    status: str = "accepted"
    # Liga o resultado a decisao que o originou (loop de feedback usa para casar
    # o fill REAL com a decisao registrada). Pode ser None em ordens externas.
    client_order_id: str | None = None
    submitted_at: datetime = Field(default_factory=_utcnow)
