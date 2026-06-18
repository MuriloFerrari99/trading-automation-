"""Agente Executor.

E o UNICO componente que escreve no broker. Pipeline de cada intencao:
1. Kill switch (aborta tudo se engajado).
2. Validacao (qtd, buying power; gate de opcoes p/ contratos).
3. Idempotencia: consulta o broker pelo client_order_id; se a ordem ja existe,
   nao reenvia (evita duplicar em retry/reconexao).
4. Persiste a ordem como PENDING_SUBMIT ANTES da chamada de rede (se cair, a
   reconciliacao sabe que existia uma ordem).
5. Submete com retry/backoff; marca SUBMITTED e o fill (FILLED/PARTIALLY_FILLED),
   usando filled_qty REAL (nunca assume fill total).
6. Auditoria (audit_log com ator + TradeLogger).
"""

from __future__ import annotations

import logging
import time

from broker.base import BrokerClient
from core.kill_switch import KillSwitch
from core.models import OptionOrderIntent, OrderIntent, OrderResult, OrderSide, is_crypto_symbol
from data.audit_log import EXECUTOR, AuditLog
from data.order_repo import (
    FILLED,
    PARTIALLY_FILLED,
    REJECTED,
    SUBMITTED,
    OrderRepository,
)
from data.trade_logger import TradeLogger

logger = logging.getLogger("agent.executor")

# Nivel minimo de opcoes p/ executar ordens de opcoes (defesa em profundidade;
# a estrategia Wheel ja aplica o mesmo gate antes de gerar a intencao).
REQUIRED_OPTIONS_LEVEL = 1

# Status crus do broker que indicam fill total.
_FILLED_STATUSES = {"filled"}
_PARTIAL_STATUSES = {"partially_filled"}
_REJECTED_STATUSES = {"rejected", "canceled", "expired"}


class OrderValidationError(ValueError):
    """Intencao reprovada na validacao pre-execucao."""


class Executor:
    def __init__(
        self,
        broker: BrokerClient,
        trade_logger: TradeLogger,
        kill_switch: KillSwitch,
        *,
        order_repo: OrderRepository | None = None,
        audit: AuditLog | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        max_orders_per_cycle: int = 25,
    ) -> None:
        self._broker = broker
        self._logger = trade_logger
        self._kill_switch = kill_switch
        self._orders = order_repo
        self._audit = audit
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        # Anti-rajada: teto de ordens enviadas num unico ciclo. Protege contra um
        # bug/loop que gere um enxame de intencoes — corta antes de inundar o broker.
        self._max_orders_per_cycle = max_orders_per_cycle

    def execute(self, intent: OrderIntent | OptionOrderIntent) -> OrderResult | None:
        """Executa uma intencao. Retorna None se barrada por kill switch/validacao."""
        # 1) Kill switch — barreira central antes de qualquer ordem.
        try:
            self._kill_switch.ensure_clear()
        except Exception as exc:
            logger.warning("Ordem barrada pelo kill switch: %s", exc)
            self._write_audit("order_blocked_killswitch", intent, {"reason": str(exc)})
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
            self._write_audit("order_rejected_validation", intent, {"reason": str(exc)})
            return None
        return self._submit(intent, lambda: self._broker.submit_order(intent), intent.symbol)

    def _execute_option(self, intent: OptionOrderIntent) -> OrderResult | None:
        # Gate defensivo: reverifica o nivel de opcoes da conta.
        account = self._broker.get_account()
        if account.options_level < REQUIRED_OPTIONS_LEVEL:
            logger.warning(
                "Ordem de opcoes barrada: nivel da conta (%d) < requerido (%d) — %s",
                account.options_level, REQUIRED_OPTIONS_LEVEL, intent.contract.occ_symbol,
            )
            self._write_audit("option_blocked_level", intent, {"level": account.options_level})
            return None
        if intent.qty <= 0:
            logger.warning("Ordem de opcoes reprovada: qty <= 0")
            return None
        return self._submit(
            intent, lambda: self._broker.submit_option_order(intent), intent.contract.occ_symbol
        )

    def _submit(self, intent, submit_fn, symbol: str) -> OrderResult | None:
        cid = intent.client_order_id

        # 3) Idempotencia: a ordem ja existe no broker? Nao reenvia.
        existing = self._safe_get_existing(cid)
        if existing is not None:
            logger.info("Ordem %s ja existe no broker (idempotencia) — nao reenviada.", cid)
            self._write_audit("order_idempotent_skip", intent, {"client_order_id": cid})
            return None

        # 4) Persiste PENDING_SUBMIT ANTES da rede.
        if self._orders is not None:
            self._orders.record_pending(intent)
        self._write_audit("order_pending", intent, {"client_order_id": cid})

        # Re-checagem do kill switch IMEDIATAMENTE antes da rede: se foi engajado
        # durante a preparacao (idempotencia/persistencia), nao envia. A ordem
        # fica PENDING_SUBMIT e a reconciliacao resolve (broker nao a conhece).
        try:
            self._kill_switch.ensure_clear()
        except Exception as exc:
            logger.warning("Ordem barrada pelo kill switch antes da rede: %s", exc)
            if self._orders is not None:
                self._orders.mark_rejected(cid)
            self._write_audit("order_blocked_killswitch", intent, {"reason": str(exc)})
            return None

        # 5) Submissao com retry/backoff.
        result = self._submit_with_retry(submit_fn, symbol, intent.side)
        if result is None:
            if self._orders is not None:
                self._orders.mark_rejected(cid)
            self._write_audit("order_submit_failed", intent, {"client_order_id": cid})
            return None

        self._persist_result(intent, result)
        return result

    def _persist_result(self, intent, result: OrderResult) -> None:
        cid = intent.client_order_id
        status = self._lifecycle_status(result.status, result.filled_qty, intent.qty)
        if self._orders is not None:
            self._orders.mark_submitted(cid, result.broker_order_id, status=status)
            if result.filled_qty and result.filled_qty > 0:
                self._orders.mark_fill(
                    cid,
                    filled_qty=result.filled_qty,
                    filled_avg_price=result.filled_avg_price,
                    status=status,
                )
        # Auditoria do fill (so registra trade quando houve preenchimento).
        if result.filled_qty and result.filled_qty > 0:
            self._log_fill(intent, result)
        self._write_audit(
            "order_submitted", intent,
            {"broker_order_id": result.broker_order_id, "status": status,
             "filled_qty": str(result.filled_qty)},
        )

    @staticmethod
    def _lifecycle_status(broker_status: str, filled_qty, requested_qty) -> str:
        s = (broker_status or "").lower()
        if s in _FILLED_STATUSES or (filled_qty and filled_qty >= requested_qty):
            return FILLED
        if s in _PARTIAL_STATUSES or (filled_qty and filled_qty > 0):
            return PARTIALLY_FILLED
        if s in _REJECTED_STATUSES:
            return REJECTED
        return SUBMITTED

    def _log_fill(self, intent, result: OrderResult) -> None:
        if isinstance(intent, OptionOrderIntent):
            from core.models import OrderIntent as _OI

            audit_intent = _OI(
                symbol=intent.contract.occ_symbol,
                side=intent.side,
                qty=intent.qty,
                strategy=intent.strategy,
            )
            self._logger.log_execution(audit_intent, result)
        else:
            self._logger.log_execution(intent, result)

    def execute_many(self, intents: list[OrderIntent | OptionOrderIntent]) -> list[OrderResult]:
        if len(intents) > self._max_orders_per_cycle:
            logger.error(
                "Anti-rajada: %d intencoes excedem o teto de %d por ciclo; "
                "executando apenas as %d primeiras e descartando o resto.",
                len(intents), self._max_orders_per_cycle, self._max_orders_per_cycle,
            )
            self._write_audit(
                "cycle_order_cap_hit", intents[0],
                {"requested": len(intents), "cap": self._max_orders_per_cycle},
            )
            intents = intents[: self._max_orders_per_cycle]
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
                # DOIS POOLS (semantica REAL da Alpaca): CRIPTO e cash-only (1x) e
                # so pode ser financiada pelo pool NAO-MARGINAVEL (caixa); ACAO/ETF
                # usa o pool MARGINAVEL (buying_power). Antes o Executor checava TUDO
                # contra o pool marginavel — entao nao conseguia barrar a cripto cedo
                # (a Alpaca a rejeitava server-side) e a paridade quebrava silenciosa.
                # Agora a cripto e validada contra o caixa: o pre-trade barra a sobra
                # ANTES da rede, em vez de descobrir na rejeicao da corretora.
                if is_crypto_symbol(intent.symbol):
                    available = account.non_marginable_buying_power
                    if estimated_cost > available:
                        raise OrderValidationError(
                            f"caixa nao-marginavel insuficiente p/ cripto: "
                            f"custo~{estimated_cost} > {available} (cash-only 1x)"
                        )
                elif estimated_cost > account.buying_power:
                    raise OrderValidationError(
                        f"buying power insuficiente: custo~{estimated_cost} > "
                        f"{account.buying_power}"
                    )

    def _safe_last_price(self, symbol: str):
        try:
            return self._broker.get_last_price(symbol)
        except Exception as exc:
            # sem preco => pula a checagem de custo, mas NAO em silencio.
            logger.warning("Falha ao obter preco de %s p/ validacao: %s", symbol, exc)
            return None

    def _safe_get_existing(self, cid: str | None):
        if cid is None:
            return None
        try:
            return self._broker.get_order_by_client_id(cid)
        except Exception as exc:
            # falha na consulta de idempotencia => segue o fluxo, mas loga.
            logger.warning("Falha na consulta de idempotencia (%s): %s", cid, exc)
            return None

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

    def _write_audit(self, event: str, intent, payload: dict) -> None:
        if self._audit is None:
            return
        symbol = (
            intent.contract.occ_symbol
            if isinstance(intent, OptionOrderIntent)
            else intent.symbol
        )
        self._audit.write(EXECUTOR, event, symbol=symbol, payload={**payload, "strategy": intent.strategy})
