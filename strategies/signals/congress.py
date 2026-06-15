"""Sinais de negociacoes de congressistas dos EUA (STOCK Act / disclosures).

A FONTE de dados fica atras da interface `CongressDataSource`, para trocar de
provedor (ex: Capital Trades, Quiver, dados publicos) sem tocar na logica.

Nota de risco (briefing): o atraso de divulgacao chega a 45 dias, o que reduz
muito o "edge". Por isso a confianca do sinal DECAI com a idade da divulgacao,
e divulgacoes muito antigas sao descartadas. Tratar como um sinal entre outros,
nunca como tese principal — e, em todo caso, sinais nao executam sozinhos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timezone

from pydantic import BaseModel

from core.models import OrderSide, Signal
from strategies.signals.base import SignalProvider

SOURCE_NAME = "congress"

# Divulgacoes mais antigas que isto sao consideradas sem edge e descartadas.
MAX_DISCLOSURE_AGE_DAYS = 45


class CongressDisclosure(BaseModel):
    """Registro bruto de uma negociacao divulgada por um congressista."""

    symbol: str
    side: OrderSide
    politician: str
    transaction_date: date
    disclosure_date: date
    amount_note: str = ""


class CongressDataSource(ABC):
    """Fonte de divulgacoes de congressistas (swappable)."""

    @abstractmethod
    def recent_disclosures(self) -> list[CongressDisclosure]:
        """Retorna as divulgacoes recentes da fonte."""


class StaticCongressSource(CongressDataSource):
    """Fonte em memoria (testes/demos e seed manual)."""

    def __init__(self, disclosures: list[CongressDisclosure] | None = None) -> None:
        self._disclosures = list(disclosures or [])

    def recent_disclosures(self) -> list[CongressDisclosure]:
        return list(self._disclosures)


class CongressTradingProvider(SignalProvider):
    """Converte divulgacoes de congressistas em sinais ponderados pela idade."""

    name = SOURCE_NAME

    def __init__(
        self,
        source: CongressDataSource,
        *,
        max_age_days: int = MAX_DISCLOSURE_AGE_DAYS,
        today: date | None = None,
    ) -> None:
        self._source = source
        self._max_age_days = max_age_days
        # `today` injetavel para testes deterministicos.
        self._today = today

    def _reference_date(self) -> date:
        return self._today or datetime.now(timezone.utc).date()

    def fetch(self) -> list[Signal]:
        today = self._reference_date()
        signals: list[Signal] = []
        for d in self._source.recent_disclosures():
            age = (today - d.disclosure_date).days
            if age < 0 or age > self._max_age_days:
                continue  # sem edge: muito antiga (ou data futura invalida)
            # Confianca decai linearmente com a idade da divulgacao:
            # divulgada hoje -> ~1.0; no limite de idade -> ~0.0.
            confidence = max(0.0, 1.0 - age / self._max_age_days)
            signals.append(
                Signal(
                    symbol=d.symbol,
                    side=d.side,
                    source=self.name,
                    confidence=round(confidence, 3),
                    note=(
                        f"{d.politician} {d.side.value} {d.symbol} "
                        f"(tx {d.transaction_date}, disclosed {d.disclosure_date}, "
                        f"age {age}d) {d.amount_note}".strip()
                    ),
                )
            )
        return signals
