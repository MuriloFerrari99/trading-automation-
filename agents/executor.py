"""Agente Executor.

Traduz intencoes em ordens na corretora. Antes de cada submissao:
1. Checa o kill switch (aborta tudo se engajado).
2. Valida a intencao (qtd, buying power para compras).
3. Submete com retry/backoff em erros transitorios.
4. Registra o trade na auditoria (TradeLogger).
"""

from __future__ import annotations

import logging
import time

from broker.base import BrokerClient
from core.kill_switch import KillSwitch
from core.models import OrderIntent, OrderResult, OrderSide
from data.trade_logger import TradeLogger

logger = logging.getLogger("agent.executor")


class OrderValidationError(ValueError):
    """Intencao reprovada na validacao pre-execucao."""


class Executor:
    def __init__(
        self,
        broker: BrokerClient,
        trade_logger: TradeLogger,
        kill_switch: KillSwitch,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._broker = broker
        self._logger = trade_logger
        self._kill_switch = kill_switch
        self._max_retries = max_retries
        self._backoff = backoff_seconds

    def execute(self, intent: OrderIntent) -> OrderResult | None:
        """Executa uma intencao. Retorna None se barrada por kill switch/validacao."""
        # 1) Kill switch — barreira central antes de qualquer ordem.
        try:
            self._kill_switch.ensure_clear()
        except Exception as exc:
            logger.warning("Ordem barrada pelo kill switch: %s", exc)
            return None

        # 2) Validacao.
        try:
            self._validate(intent)
        except OrderValidationError as exc:
            logger.warning("Intencao reprovada (%s %s): %s", intent.side.value, intent.symbol, exc)
            return None

        # 3) Submissao com retry/backoff.
        result = self._submit_with_retry(intent)
        if result is None:
            return None

        # 4) Auditoria.
        self._logger.log_execution(intent, result)
        return result

    def execute_many(self, intents: list[OrderIntent]) -> list[OrderResult]:
        results = []
        for intent in intents:
            res = self.execute(intent)
            if res is not None:
                results.append(res)
        return results

    def _validate(self, intent: OrderIntent) -> None:
        if intent.qty <= 0:
            raise OrderValidationError("quantidade deve ser > 0")
        if intent.side == OrderSide.BUY:
            account = self._broker.get_account()
            price = intent.limit_price or self._safe_last_price(intent.symbol)
            if price is not None:
                estimated_cost = intent.qty * price
                if estimated_cost > account.buying_power:
                    raise OrderValidationError(
                        f"buying power insuficiente: custo~{estimated_cost} > "
                        f"{account.buying_power}"
                    )

    def _safe_last_price(self, symbol: str):
        try:
            return self._broker.get_last_price(symbol)
        except Exception:
            return None  # sem preco => pula a checagem de custo

    def _submit_with_retry(self, intent: OrderIntent) -> OrderResult | None:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._broker.submit_order(intent)
            except Exception as exc:  # erro transitorio da API
                last_exc = exc
                logger.warning(
                    "Falha ao submeter %s %s (tentativa %d/%d): %s",
                    intent.side.value, intent.symbol, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._backoff * attempt)
        logger.error(
            "Ordem %s %s falhou apos %d tentativas: %s",
            intent.side.value, intent.symbol, self._max_retries, last_exc,
        )
        return None
