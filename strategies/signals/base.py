"""Interface de provedores de sinais (Nivel 2) e agregador.

Um SignalProvider isola a FONTE de dados (congressistas, grandes fundos, etc.)
atras de uma interface unica, para que o provedor possa ser trocado sem mexer
no Planejador.

REGRA INEGOCIAVEL: sinais sao apenas SUGESTOES para o Planejador. Eles nunca
sao convertidos em ordens automaticamente. O SignalService apenas coleta e
entrega os sinais; quem decide o que fazer (e que NAO os executa) e o Planner.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.models import Signal

logger = logging.getLogger("signals")


class SignalProvider(ABC):
    #: identificador curto da fonte (vai no campo `source` do Signal)
    name: str = "base"

    @abstractmethod
    def fetch(self) -> list[Signal]:
        """Busca e retorna os sinais atuais da fonte (pode ser vazio)."""


class SignalService:
    """Agrega varios SignalProviders, isolando falhas por provedor."""

    def __init__(self, providers: list[SignalProvider] | None = None) -> None:
        self._providers = providers or []

    def collect(self) -> list[Signal]:
        signals: list[Signal] = []
        for provider in self._providers:
            try:
                produced = provider.fetch()
            except Exception:  # um provedor com erro nao derruba os demais
                logger.exception("SignalProvider '%s' falhou ao buscar", provider.name)
                continue
            if produced:
                logger.info("Provider '%s' gerou %d sinal(is)", provider.name, len(produced))
            signals.extend(produced)
        return signals
