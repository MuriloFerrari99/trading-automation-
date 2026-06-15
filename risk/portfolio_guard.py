"""Circuit breakers de portfolio (doc 02 §5).

Defesa em camadas no nivel agregado: limite de perda diaria, max drawdown,
exposicao por simbolo e portfolio heat. Quando um limite e atingido, o bot
PARA de abrir risco novo (fail-safe), mas saidas/protecoes continuam permitidas.
"""

from __future__ import annotations

import logging
from decimal import Decimal

logger = logging.getLogger("risk.portfolio_guard")


class PortfolioRiskGuard:
    def __init__(
        self,
        start_equity: Decimal,
        *,
        daily_loss_pct: Decimal = Decimal("0.03"),
        max_dd_pct: Decimal = Decimal("0.20"),
        max_per_symbol_pct: Decimal = Decimal("0.20"),
        max_heat_pct: Decimal = Decimal("0.10"),
        peak_equity: Decimal | None = None,
    ) -> None:
        self.start_equity = start_equity
        self.peak_equity = peak_equity or start_equity
        self.daily_loss_pct = daily_loss_pct
        self.max_dd_pct = max_dd_pct
        self.max_per_symbol_pct = max_per_symbol_pct
        self.max_heat_pct = max_heat_pct
        self.trading_halted = False
        self.halt_reason: str | None = None

    def update(self, equity: Decimal) -> str | None:
        """Atualiza com o equity atual; engaja halt se um limite foi atingido."""
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.start_equity > 0:
            daily_pl = (equity - self.start_equity) / self.start_equity
            if daily_pl <= -self.daily_loss_pct:
                return self._halt(f"daily loss limit atingido ({daily_pl:.2%})")

        if self.peak_equity > 0:
            drawdown = (equity - self.peak_equity) / self.peak_equity
            if drawdown <= -self.max_dd_pct:
                return self._halt(f"max drawdown atingido ({drawdown:.2%})")

        return self.halt_reason  # mantem halt se ja engajado

    def _halt(self, reason: str) -> str:
        if not self.trading_halted:
            logger.critical("HALT de portfolio: %s", reason)
        self.trading_halted = True
        self.halt_reason = reason
        return reason

    def can_open(
        self, symbol_exposure_pct: Decimal, current_heat_pct: Decimal
    ) -> tuple[bool, str]:
        """Pode ABRIR risco novo? (saidas nao passam por aqui)."""
        if self.trading_halted:
            return False, f"trading halted: {self.halt_reason}"
        if symbol_exposure_pct > self.max_per_symbol_pct:
            return False, (
                f"exposicao por simbolo excedida "
                f"({symbol_exposure_pct:.2%} > {self.max_per_symbol_pct:.2%})"
            )
        if current_heat_pct > self.max_heat_pct:
            return False, (
                f"portfolio heat excedido "
                f"({current_heat_pct:.2%} > {self.max_heat_pct:.2%})"
            )
        return True, "ok"
