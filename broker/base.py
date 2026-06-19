"""Interface do broker.

Toda a logica de negocio (agentes/estrategias) depende apenas desta interface,
nunca da Alpaca diretamente. Isso permite testar com FakeBroker sem tocar a
corretora e, no futuro, trocar de corretora sem reescrever a logica.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderResult,
    Position,
)


@dataclass
class MarketClockInfo:
    """Estado do relogio de mercado (fonte de verdade: clock da corretora).

    `is_open` ja considera feriados e early-close (a corretora resolve isso via
    calendar). next_open/next_close permitem agendar/logar com precisao.
    """

    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None
    timestamp: datetime | None = None


@dataclass
class BrokerOrder:
    """Visao de uma ordem no broker (para reconciliacao broker=verdade)."""

    broker_order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    qty: Decimal
    filled_qty: Decimal
    status: str  # status cru do broker (ex: new, filled, partially_filled, canceled)
    order_type: str
    filled_avg_price: Decimal | None = None  # preco medio do fill (p/ P&L/reconcile)


class AccountInfo:
    """Resumo minimo da conta usado pela logica de negocio.

    DOIS POOLS DE PODER DE COMPRA (semantica REAL da Alpaca):
      - `buying_power`: pool MARGINAVEL (acoes/ETFs). Numa conta de margem Reg-T
        e ~2x o equity (overnight). Cripto NAO consome este pool.
      - `non_marginable_buying_power`: pool NAO-MARGINAVEL (cash-only) contra o
        qual a Alpaca avalia ordens de CRIPTO. E o CAIXA liquidado disponivel —
        cripto e 1x, sem margem. As acoes tomam margem mas tambem CONSOMEM esse
        caixa como colateral, entao a cripto precisa do caixa reservado ANTES.
    Separar os dois e o que permite ao pre-trade validar a cripto contra o pool
    CERTO (caixa), em vez de contra o pool marginavel (que mostra espaco que a
    cripto nao pode usar) — a causa-raiz do leg de cripto rejeitado a 2.0x.
    """

    def __init__(
        self,
        *,
        cash: Decimal,
        buying_power: Decimal,
        equity: Decimal,
        currency: str = "USD",
        options_level: int = 0,
        non_marginable_buying_power: Decimal | None = None,
    ) -> None:
        self.cash = cash
        self.buying_power = buying_power
        self.equity = equity
        self.currency = currency
        # Nivel de aprovacao de opcoes (0 = sem permissao). Usado para gate da
        # Wheel Strategy (Nivel 3) antes de habilita-la.
        self.options_level = options_level
        # Pool NAO-MARGINAVEL (caixa p/ cripto cash-only). Default = `cash` quando
        # o broker nao expoe o campo separado (compat.): o caixa liquidado e a
        # melhor aproximacao do non_marginable_buying_power da Alpaca.
        self.non_marginable_buying_power = (
            non_marginable_buying_power
            if non_marginable_buying_power is not None
            else cash
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"AccountInfo(cash={self.cash}, buying_power={self.buying_power}, "
            f"equity={self.equity}, options_level={self.options_level}, "
            f"non_marginable_buying_power={self.non_marginable_buying_power})"
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
    def get_bars(self, symbol: str, limit: int = 60, *, timeframe: str = "1Day") -> list[Decimal]:
        """Fechamentos recentes (ordem cronologica) p/ classificacao de regime.

        `timeframe` e uma string portavel ("1Min", "5Min", "1Hour", "1Day", ...);
        cada broker a traduz para o seu proprio enum. Default diario (legado)."""

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> OrderResult:
        """Submete uma ordem a partir de uma intencao ja validada."""

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancela todas as ordens abertas. Retorna quantas foram canceladas."""

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]:
        """Ordens abertas no broker (para reconciliacao)."""

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Busca uma ordem pelo client_order_id (idempotencia em retry)."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """True se o mercado esta aberto agora (clock da corretora)."""

    @abstractmethod
    def get_clock(self) -> MarketClockInfo:
        """Relogio de mercado completo (is_open + next_open/next_close)."""

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
