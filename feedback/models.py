"""Modelos do loop de feedback (decisao, contexto e resultado).

Pydantic v2, no mesmo espirito de core/models.py: validacao nas fronteiras.
Estes modelos sao deliberadamente desacoplados de OrderIntent/OrderResult para
que a Camada 0 possa registrar QUALQUER decisao (inclusive 'skip'/'hold'), nao
so as que viram ordem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MarketRegime(str, Enum):
    """Regime de mercado no momento da decisao (classificador em regime.py).

    Saber em qual regime a decisao foi tomada e meia inteligencia: a mesma
    estrategia que ganha em tendencia costuma sangrar em lateralizacao.
    """

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


class DecisionAction(str, Enum):
    """O que o sistema decidiu — incluindo decidir NAO operar."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"  # tinha posicao/oportunidade e decidiu manter
    SKIP = "skip"  # avaliou e decidiu nao operar (sem edge)


class OutcomeStatus(str, Enum):
    """Desfecho de uma decisao."""

    OPEN = "open"  # virou trade, ainda aberto
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    SKIPPED = "skipped"  # decisao skip/hold, sem trade associado
    CANCELED = "canceled"


class Decision(BaseModel):
    """Uma decisao do sistema, com o contexto que a originou.

    Gravada NO MOMENTO da decisao. O resultado (Outcome) e anexado depois,
    quando conhecido — e esse par decisao->resultado que vira o dataset
    proprietario do qual o sistema aprende.
    """

    strategy: str
    symbol: str
    action: DecisionAction
    regime: MarketRegime = MarketRegime.UNKNOWN
    # Preco de referencia no instante da decisao (para calcular retorno depois).
    reference_price: Decimal | None = Field(default=None, gt=0)
    # Confianca agregada dos sinais que motivaram a decisao (0..1).
    signal_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    # Features/indicadores/sinais que dispararam — JSON livre para auditoria e
    # para virar features de ML no futuro (Camada 1).
    context: dict = Field(default_factory=dict)
    # Liga a decisao a ordem resultante (quando virou ordem).
    client_order_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Outcome(BaseModel):
    """Resultado realizado de uma decisao, anexado quando o trade fecha."""

    status: OutcomeStatus
    entry_price: Decimal | None = Field(default=None, gt=0)
    exit_price: Decimal | None = Field(default=None, gt=0)
    # P&L realizado na moeda da conta (pode ser negativo).
    realized_pnl: Decimal | None = None
    # Retorno percentual do trade (ex: 0.012 = +1,2%).
    return_pct: float | None = None
    closed_at: datetime | None = None
    note: str = ""
