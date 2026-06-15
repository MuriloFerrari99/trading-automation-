"""Sinais de movimentacao de grandes fundos ("smart money").

Baseado em mudancas de posicao reportadas por grandes gestores (ex: 13F).
A FONTE fica atras de `SmartMoneyDataSource` para troca de provedor.

Assim como os sinais de congressistas, estes sao apenas SUGESTOES: nao viram
ordens automaticamente. A confianca pondera o tamanho relativo da mudanca de
posicao (quanto maior a variacao, maior a conviccao implicita).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from pydantic import BaseModel, Field

from core.models import OrderSide, Signal
from strategies.signals.base import SignalProvider

SOURCE_NAME = "smart_money"


class FundPositionChange(BaseModel):
    """Mudanca de posicao de um fundo em um ativo."""

    symbol: str
    fund: str
    side: OrderSide  # BUY (aumentou/abriu) ou SELL (reduziu/zerou)
    # Variacao relativa da posicao (0..1+). Ex: 0.5 = +50% na posicao.
    change_ratio: Decimal = Field(..., ge=0)


class SmartMoneyDataSource(ABC):
    @abstractmethod
    def recent_changes(self) -> list[FundPositionChange]:
        """Retorna as mudancas de posicao recentes da fonte."""


class StaticSmartMoneySource(SmartMoneyDataSource):
    """Fonte em memoria (testes/demos e seed manual)."""

    def __init__(self, changes: list[FundPositionChange] | None = None) -> None:
        self._changes = list(changes or [])

    def recent_changes(self) -> list[FundPositionChange]:
        return list(self._changes)


class SmartMoneyProvider(SignalProvider):
    name = SOURCE_NAME

    def __init__(
        self,
        source: SmartMoneyDataSource,
        *,
        min_change_ratio: Decimal = Decimal("0.10"),
    ) -> None:
        self._source = source
        # Ignora mudancas pequenas (ruido).
        self._min_change_ratio = min_change_ratio

    def fetch(self) -> list[Signal]:
        signals: list[Signal] = []
        for c in self._source.recent_changes():
            if c.change_ratio < self._min_change_ratio:
                continue
            # Confianca cresce com o tamanho da mudanca, saturando em 1.0.
            confidence = min(1.0, float(c.change_ratio))
            signals.append(
                Signal(
                    symbol=c.symbol,
                    side=c.side,
                    source=self.name,
                    confidence=round(confidence, 3),
                    note=f"{c.fund} {c.side.value} {c.symbol} (change {c.change_ratio})",
                )
            )
        return signals
