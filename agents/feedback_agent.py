"""FeedbackAgent — fecha o loop de aprendizado (doc 08 / Camada 0).

Consome `fills` e, quando um trade FECHA (uma venda que zera/reduz um long),
anexa o `Outcome` (win/loss/breakeven) as decisoes de COMPRA abertas daquele
ativo — ligando decisao -> resultado, que e o dataset do qual a DecisionPolicy
e o ML aprendem.

Matching v2 — FIFO por LOTE. Cada compra registra sua qty (no fill); uma venda
de S acoes abate `remaining_qty` das compras abertas em ordem cronologica
(mais antiga primeiro), fechando cada lote so quando totalmente consumido. Isso
atribui o P&L a decisao certa (antes, uma venda fechava TODOS os longs do
simbolo de uma vez, contaminando o dataset). Lotes sem qty conhecida (legado/
decisoes sem fill de compra) caem no fechamento integral (comportamento antigo).

Aproximacao conhecida: um lote fechado ao longo de varias vendas usa o preco da
venda que o ZEROU (nao uma media ponderada) — suficiente para o rotulo win/loss
e muito mais fiel que o matching v1.
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
        side = str(fill.get("side", "")).lower()
        # Compra: fixa o entry_price REAL (fill) e a qty (lote) na decisao aberta,
        # para o P&L ser medido do preco efetivo de entrada e o matching ser FIFO.
        if side == "buy":
            coid = fill.get("client_order_id")
            if coid is not None:
                self._log.set_entry_price_by_client_order_id(
                    coid, Decimal(str(fill["fill_price"]))
                )
                qty = fill.get("filled_qty")
                if qty is not None:
                    self._log.set_qty_by_client_order_id(coid, Decimal(str(qty)))
            return
        # Venda (saida de long): abate os lotes de compra abertos em ordem FIFO.
        if side != "sell":
            return
        self._match_sell_fifo(
            symbol=str(fill["symbol"]).upper(),
            exit_price=Decimal(str(fill["fill_price"])),
            sell_qty=Decimal(str(fill.get("filled_qty", 0))),
        )

    def _match_sell_fifo(
        self, symbol: str, exit_price: Decimal, sell_qty: Decimal
    ) -> None:
        to_match = sell_qty
        for lot in self._log.open_buy_lots(symbol):
            ref = lot.get("entry_price") or lot.get("reference_price")
            if ref is None:
                continue  # sem entrada registrada => nao da p/ avaliar
            entry = Decimal(str(ref))
            if entry <= 0:
                continue

            lot_remaining = lot.get("remaining_qty")
            # Lote sem qty conhecida (legado) OU venda sem qty: fecha integral
            # (comportamento v1 para esses casos — nao da p/ abater por lote).
            if lot_remaining is None or sell_qty <= 0:
                self._close_lot(lot, entry, exit_price, Decimal(str(lot.get("qty") or 0)))
                continue

            if to_match <= 0:
                break  # venda ja totalmente casada

            lot_remaining = Decimal(str(lot_remaining))
            take = min(lot_remaining, to_match)
            to_match -= take
            new_remaining = lot_remaining - take
            if new_remaining <= 0:
                # lote totalmente consumido => fecha e rotula.
                self._close_lot(
                    lot, entry, exit_price, Decimal(str(lot.get("qty") or take))
                )
            else:
                # consumo parcial => abate e mantem aberto (fecha numa venda futura).
                self._log.reduce_remaining(int(lot["id"]), new_remaining)
                self.log.info(
                    "Lote parcial %s id=%s: -%s, resta %s",
                    symbol, lot["id"], take, new_remaining,
                )

    def _close_lot(
        self, lot: dict, entry: Decimal, exit_price: Decimal, qty: Decimal
    ) -> None:
        realized = (exit_price - entry) * qty if qty > 0 else (exit_price - entry)
        return_pct = float((exit_price - entry) / entry)
        status = (
            OutcomeStatus.WIN if exit_price > entry
            else OutcomeStatus.LOSS if exit_price < entry
            else OutcomeStatus.BREAKEVEN
        )
        self._log.attach_outcome(
            int(lot["id"]),
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
            lot["symbol"], status.value, entry, exit_price, return_pct * 100,
        )
