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
from core.models import OptionOrderIntent, OrderIntent, OrderResult, OrderSide
from data.trade_logger import TradeLogger

logger = logging.getLogger("agent.executor")

# Nivel minimo de opcoes p/ executar ordens de opcoes (defesa em profundidade;
# a estrategia Wheel ja aplica o mesmo gate antes de gerar a intencao).
REQUIRED_OPTIONS_LEVEL = 1


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

    def execute(self, intent: OrderIntent | OptionOrderIntent) -> OrderResult | None:
        """Executa uma intencao. Retorna None se barrada por kill switch/validacao."""
        # 1) Kill switch — barreira central antes de qualquer ordem.
        try:
            self._kill_switch.ensure_clear()
        except Exception as exc:
            logger.warning("Ordem barrada pelo kill switch: %s", exc)
            return None

        # 2) Despacho por tipo de instrumento.
        if isinstance(intent, OptionOrderIntent):
            return self._execute_option(intent)
        return self._execute_equity(intent)

    def _execute_equity(self, intent: OrderIntent) -> OrderResult | None:
        try:
            self._validate(intent)
        except OrderValidationError as exc:
            logger.warning("Intencao reprovada (%s %s): %s", intent.side.value, intent.symbol, exc)
            return None

        result = self._submit_with_retry(lambda: self._broker.submit_order(intent), intent.symbol, intent.side)
        if result is None:
            return None
        self._logger.log_execution(intent, result)
        return result

    def _execute_option(self, intent: OptionOrderIntent) -> OrderResult | None:
        # Gate defensivo: reverifica o nivel de opcoes da conta.
        account = self._broker.get_account()
        if account.options_level < REQUIRED_OPTIONS_LEVEL:
            logger.warning(
                "Ordem de opcoes barrada: nivel da conta (%d) < requerido (%d) — %s",
                account.options_level, REQUIRED_OPTIONS_LEVEL, intent.contract.occ_symbol,
            )
            return None
        if intent.qty <= 0:
            logger.warning("Ordem de opcoes reprovada: qty <= 0")
            return None

        result = self._submit_with_retry(
            lambda: self._broker.submit_option_order(intent),
            intent.contract.occ_symbol,
            intent.side,
        )
        if result is None:
            return None
        # Auditoria: registra com o equivalente em acoes (contratos * 100).
        self._log_option_execution(intent, result)
        return result

    def _log_option_execution(self, intent: OptionOrderIntent, result: OrderResult) -> None:
        from core.models import OrderIntent as _OI

        audit_intent = _OI(
            symbol=intent.contract.occ_symbol,
            side=intent.side,
            qty=intent.qty,
            strategy=intent.strategy,
        )
        self._logger.log_execution(audit_intent, result)

    def execute_many(self, intents: list[OrderIntent | OptionOrderIntent]) -> list[OrderResult]:
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

    def _submit_with_retry(self, submit, symbol: str, side) -> OrderResult | None:
        """Executa `submit()` com retry/backoff em erros transitorios da API."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return submit()
            except Exception as exc:  # erro transitorio da API
                last_exc = exc
                logger.warning(
                    "Falha ao submeter %s %s (tentativa %d/%d): %s",
                    side.value, symbol, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._backoff * attempt)
        logger.error(
            "Ordem %s %s falhou apos %d tentativas: %s",
            side.value, symbol, self._max_retries, last_exc,
        )
        return None
