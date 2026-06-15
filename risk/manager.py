"""RiskManager — camada de decisao/risco (doc 02 §0 risk-first, §4-§5).

Avalia cada intencao ANTES de ir para execucao:
- Atualiza o circuit breaker com o equity atual (daily loss / max drawdown).
- Saidas e protecoes (reduzem risco) sempre passam — fail-safe nao tranca a porta de saida.
- Ordens que AUMENTAM risco (compra de acao; venda de put) so passam se:
  nao houver halt, a exposicao por simbolo e o portfolio heat respeitarem os
  limites, e houver dado de preco (sem preco => veta, fail-safe).
- Pode REDUZIR a quantidade de uma compra para respeitar o teto por simbolo.

Nota: "portfolio heat" e o RISCO-ATE-O-STOP somado (nao a exposicao bruta).
Sem stop especifico por posicao, aproximamos: heat = exposicao * assumed_stop_pct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from config.risk import RiskSettings
from core.models import OptionOrderIntent, OptionType, OrderIntent, OrderSide, Position
from risk.portfolio_guard import PortfolioRiskGuard
from risk.sizing import fixed_fractional_qty, max_qty_for_exposure

logger = logging.getLogger("risk.manager")

TradeIntent = OrderIntent | OptionOrderIntent
SHARES_PER_CONTRACT = Decimal(100)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    intent: TradeIntent | None  # possivelmente com qty ajustada


class RiskManager:
    def __init__(self, settings: RiskSettings, guard: PortfolioRiskGuard) -> None:
        self._s = settings
        self._guard = guard

    @property
    def guard(self) -> PortfolioRiskGuard:
        return self._guard

    def begin_cycle(self, equity: Decimal) -> str | None:
        """Atualiza o circuit breaker; retorna o motivo do halt ou None."""
        return self._guard.update(equity)

    @staticmethod
    def is_risk_increasing(intent: TradeIntent) -> bool:
        if isinstance(intent, OptionOrderIntent):
            # Venda de PUT (cash-secured) abre exposicao; covered call reduz/neutro.
            return intent.contract.option_type == OptionType.PUT and intent.side == OrderSide.SELL
        return intent.side == OrderSide.BUY

    def assess(
        self,
        intent: TradeIntent,
        equity: Decimal,
        positions: list[Position],
        price: Decimal | None,
        *,
        fractional: bool = False,
    ) -> RiskDecision:
        # Saidas/protecoes sempre passam.
        if not self.is_risk_increasing(intent):
            return RiskDecision(True, "reduz/neutraliza risco", intent)

        # Halt global barra qualquer risco novo.
        if self._guard.trading_halted:
            return RiskDecision(False, f"halt: {self._guard.halt_reason}", None)

        # Fail-safe: sem dado de preco nao se abre risco.
        if price is None or price <= 0 or equity <= 0:
            return RiskDecision(False, "sem preco/equity confiavel (fail-safe)", None)

        by_symbol = {p.symbol: p for p in positions}

        if isinstance(intent, OptionOrderIntent):
            return self._assess_option(intent, equity, by_symbol, price)
        return self._assess_equity_buy(intent, equity, by_symbol, price, fractional)

    def _assess_equity_buy(self, intent, equity, by_symbol, price, fractional=False) -> RiskDecision:
        existing = by_symbol.get(intent.symbol)
        current_qty = existing.qty if existing else Decimal(0)

        # Cap por exposicao do simbolo.
        max_add = max_qty_for_exposure(
            equity, self._s.max_per_symbol_pct, price, current_qty, fractional=fractional
        )
        if max_add <= 0:
            return RiskDecision(False, "exposicao por simbolo no limite", None)
        final_qty = min(intent.qty, max_add)

        # Cap por risco-por-trade (fixed fractional ate o stop assumido): nunca
        # arrisca mais que risk_per_trade_pct do equity num unico trade. So
        # REDUZ a quantidade — alinha o sizing ao stop protetor que sera emitido.
        stop_price = price * (Decimal(1) - self._s.assumed_stop_pct)
        qty_risk = fixed_fractional_qty(
            equity, self._s.risk_per_trade_pct, price, stop_price, fractional=fractional
        )
        if qty_risk > 0:
            final_qty = min(final_qty, qty_risk)
        if final_qty <= 0:
            return RiskDecision(False, "qty <= 0 apos limite de risco-por-trade", None)

        # Heat agregado apos a adicao.
        heat = self._heat_after(by_symbol, equity, intent.symbol, final_qty, price)
        symbol_exposure = (current_qty + final_qty) * price / equity
        ok, reason = self._guard.can_open(symbol_exposure, heat)
        if not ok:
            return RiskDecision(False, reason, None)

        if final_qty < intent.qty:
            adjusted = intent.model_copy(update={"qty": final_qty})
            return RiskDecision(
                True, f"qty reduzida p/ limites de risco ({final_qty})", adjusted
            )
        return RiskDecision(True, "ok", intent)

    def _assess_option(self, intent, equity, by_symbol, price) -> RiskDecision:
        # Colateral do cash-secured put ~ strike * 100 * contratos.
        collateral = intent.contract.strike * SHARES_PER_CONTRACT * intent.qty
        symbol_exposure = collateral / equity
        heat = self._heat(by_symbol, equity) + symbol_exposure * self._s.assumed_stop_pct
        ok, reason = self._guard.can_open(symbol_exposure, heat)
        if not ok:
            return RiskDecision(False, reason, None)
        return RiskDecision(True, "ok", intent)

    def _gross_exposure(self, by_symbol: dict[str, Position], equity: Decimal) -> Decimal:
        if equity <= 0:
            return Decimal(0)
        gross = Decimal(0)
        for p in by_symbol.values():
            px = p.current_price if p.current_price is not None else p.avg_entry_price
            gross += abs(p.qty) * px
        return gross / equity

    def _heat(self, by_symbol: dict[str, Position], equity: Decimal) -> Decimal:
        # Risco-ate-o-stop ~ exposicao bruta * stop assumido.
        return self._gross_exposure(by_symbol, equity) * self._s.assumed_stop_pct

    def _heat_after(self, by_symbol, equity, symbol, added_qty, price) -> Decimal:
        if equity <= 0:
            return Decimal(0)
        added_exposure = added_qty * price / equity
        return (self._gross_exposure(by_symbol, equity) + added_exposure) * self._s.assumed_stop_pct
