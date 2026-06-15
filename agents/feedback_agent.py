"""FeedbackAgent — fecha o loop de aprendizado (doc 08 / Camada 0).

Consome `fills` e, quando um trade FECHA (uma venda que zera/reduz um long),
anexa o `Outcome` (win/loss/breakeven) as decisoes de COMPRA abertas daquele
ativo — ligando decisao -> resultado, que e o dataset do qual a DecisionPolicy
aprende.

Matching v1 (documentado, refinavel): por SIMBOLO. Uma venda de S fecha as
decisoes de compra ainda OPEN de S; o retorno e medido do reference_price
(entrada) ate o preco do fill (saida). Pareamento por lote/FIFO e melhoria
futura.
"""

from __future__ import annotations

from decimal import Decimal

from agents.base import BaseAgent
from feedback.decision_log import DecisionLog
from feedback.models import Outcome, OutcomeStatus
from orchestration.bus import Message, MessageBus

FILLS = "fills"


class FeedbackAgent(BaseAgent):
    def __init__(self, decision_log: DecisionLog) -> None:
        super().__init__("feedback", inbox=FILLS)
        self._log = decision_log

    def handle(self, msg: Message, bus: MessageBus) -> None:
        fill = msg.payload
        # So saidas (venda de long) fecham trades neste matching v1.
        if str(fill.get("side", "")).lower() != "sell":
            return
        self._close_open_longs(
            symbol=str(fill["symbol"]).upper(),
            exit_price=Decimal(str(fill["fill_price"])),
            qty=Decimal(str(fill.get("filled_qty", 0))),
        )

    def _close_open_longs(self, symbol: str, exit_price: Decimal, qty: Decimal) -> None:
        for dec in self._log.open_decisions():
            if dec["symbol"].upper() != symbol or dec["action"] != "buy":
                continue
            ref = dec.get("reference_price")
            if ref is None:
                continue  # sem entrada registrada => nao da p/ avaliar
            entry = Decimal(str(ref))
            if entry <= 0:
                continue
            realized = (exit_price - entry) * qty if qty > 0 else (exit_price - entry)
            return_pct = float((exit_price - entry) / entry)
            status = (
                OutcomeStatus.WIN if exit_price > entry
                else OutcomeStatus.LOSS if exit_price < entry
                else OutcomeStatus.BREAKEVEN
            )
            self._log.attach_outcome(
                int(dec["id"]),
                Outcome(
                    status=status,
                    entry_price=entry,
                    exit_price=exit_price,
                    realized_pnl=realized,
                    return_pct=return_pct,
                    note=f"fechado por venda @ {exit_price}",
                ),
            )
            self.log.info(
                "Loop fechado %s: %s entry=%s exit=%s ret=%.2f%%",
                symbol, status.value, entry, exit_price, return_pct * 100,
            )
