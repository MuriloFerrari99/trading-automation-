"""Interface do broker.

Toda a logica de negocio (agentes/estrategias) depende apenas desta interface,
nunca da Alpaca diretamente. Isso permite testar com FakeBroker sem tocar a
corretora e, no futuro, trocar de corretora sem reescrever a logica.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderResult,
    Position,
)


class AccountInfo:
    """Resumo minimo da conta usado pela logica de negocio."""

    def __init__(
        self,
        *,
        cash: Decimal,
        buying_power: Decimal,
        equity: Decimal,
        currency: str = "USD",
        options_level: int = 0,
    ) -> None:
        self.cash = cash
        self.buying_power = buying_power
        self.equity = equity
        self.currency = currency
        # Nivel de aprovacao de opcoes (0 = sem permissao). Usado para gate da
        # Wheel Strategy (Nivel 3) antes de habilita-la.
        self.options_level = options_level

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"AccountInfo(cash={self.cash}, buying_power={self.buying_power}, "
            f"equity={self.equity}, options_level={self.options_level})"
        )


class BrokerClient(ABC):
    """Contrato que toda implementacao de corretora deve cumprir."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Retorna informacoes da conta."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Retorna as posicoes abertas."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Retorna a posicao de um ativo, ou None se nao houver."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> Decimal:
        """Retorna o ultimo preco negociado do ativo (quote/trade)."""

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Submete uma ordem a partir de uma intencao ja validada."""

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancela todas as ordens abertas. Retorna quantas foram canceladas."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """True se o mercado esta aberto agora (clock da corretora)."""

    # --- Opcoes (Nivel 3) ---------------------------------------------------
    @abstractmethod
    def select_option_contract(
        self,
        underlying: str,
        option_type: OptionType,
        target_strike: Decimal,
        *,
        min_dte: int,
        max_dte: int,
    ) -> OptionContract | None:
        """Seleciona o contrato mais proximo do strike-alvo dentro da janela de
        vencimento (DTE = dias ate expirar). Retorna None se nada elegivel."""

    @abstractmethod
    def submit_option_order(self, intent: OptionOrderIntent) -> OrderResult:
        """Submete uma ordem de opcoes a partir de uma intencao ja validada."""
