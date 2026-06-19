"""Trava de risco ISOLADA do sleeve de cripto (Sistema 2).

Parede de fogo entre o Sistema 1 (buy-and-hold, equity Alpaca) e o Sistema 2
(caca de alpha intradiario em cripto perp na Binance). Este guard:

- So conhece o capital ISOLADO do sleeve. NUNCA toca o equity do Sistema 1 —
  o sizing e dimensionado contra `sleeve_capital_usd`, nao contra o equity total.
- Aplica um CEILING RIGIDO de alavancagem (LEVERAGE_HARD_CEILING) mesmo que a
  config peca mais. Perp permite 50x; a trava existe p/ impedir o suicidio.
- Roda kills INDEPENDENTES do Sistema 1: perda diaria e drawdown do sleeve.
- Limita posicoes concorrentes e exposicao bruta do sleeve.
- Respeita o kill switch GLOBAL do projeto (core.kill_switch) — reutiliza, nao
  recria. Se o kill global estiver engajado, nada abre.

Strategy-agnostic: o metodo `allowed_size` recebe (preco, stop, sinal) e devolve
o tamanho permitido OU bloqueia. Serve p/ momentum, liquidacao ou market-making.

Fail-safe: quando um limite e atingido, o sleeve PARA de abrir risco novo. Como
no PortfolioRiskGuard, saidas/protecoes nao passam por aqui.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from config.crypto_sleeve import LEVERAGE_HARD_CEILING, CryptoSleeveSettings
from core.kill_switch import KillSwitch
from risk.sizing import fixed_fractional_qty

logger = logging.getLogger("risk.crypto_sleeve")


@dataclass(frozen=True)
class SizingDecision:
    """Resultado de um pedido de sizing ao sleeve."""

    allowed: bool
    qty: Decimal
    reason: str
    # Notional efetivo (qty * preco) e alavancagem implicada, p/ telemetria.
    notional_usd: Decimal = Decimal(0)
    implied_leverage: Decimal = Decimal(0)


class CryptoSleeveGuard:
    """Trava de risco isolada do sleeve de cripto (Sistema 2).

    Conservador por padrao. Todos os limites tem teto rigido; o de alavancagem
    nao pode ser ultrapassado nem por configuracao absurda.
    """

    def __init__(
        self,
        settings: CryptoSleeveSettings | None = None,
        *,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._settings = settings or CryptoSleeveSettings()  # type: ignore[call-arg]
        self._kill = kill_switch or KillSwitch()

        s = self._settings
        if s.sleeve_capital_usd <= 0:
            raise ValueError("sleeve_capital_usd deve ser > 0")

        # Capital ISOLADO do sleeve. Esta e a unica base de capital que o guard
        # conhece — a parede de fogo contra o Sistema 1.
        self.sleeve_capital_usd: Decimal = s.sleeve_capital_usd

        # Alavancagem EFETIVA = min(config, ceiling rigido). Dupla trava: a
        # config ja clampa, e aqui re-aplicamos para nao depender disso.
        self.leverage_cap: Decimal = min(
            max(s.leverage_cap, Decimal("1")), LEVERAGE_HARD_CEILING
        )

        self.risk_per_trade_pct: Decimal = s.risk_per_trade_pct
        self.daily_loss_limit_pct: Decimal = s.daily_loss_limit_pct
        self.max_drawdown_pct: Decimal = s.max_drawdown_pct
        self.max_concurrent_positions: int = s.max_concurrent_positions
        # Exposicao bruta nunca pode exceder a alavancagem efetiva.
        self.max_gross_exposure_x: Decimal = min(s.max_gross_exposure_x, self.leverage_cap)

        # Estado de equity do sleeve (alimentado por update()).
        self.start_capital: Decimal = s.sleeve_capital_usd
        self.peak_equity: Decimal = s.sleeve_capital_usd
        self.trading_halted: bool = False
        self.halt_reason: str | None = None

    # --- Circuit breakers do sleeve (independentes do Sistema 1) ------------
    def update(self, sleeve_equity: Decimal) -> str | None:
        """Atualiza com o equity ATUAL do sleeve; engaja halt se um limite caiu.

        `sleeve_equity` e o valor corrente do capital isolado do sleeve (nunca o
        equity do Sistema 1). Mantem high-water p/ o gate de drawdown.
        """
        if sleeve_equity > self.peak_equity:
            self.peak_equity = sleeve_equity

        if self.start_capital > 0:
            daily_pl = (sleeve_equity - self.start_capital) / self.start_capital
            if daily_pl <= -self.daily_loss_limit_pct:
                return self._halt(f"daily loss do sleeve atingido ({daily_pl:.2%})")

        if self.peak_equity > 0:
            drawdown = (sleeve_equity - self.peak_equity) / self.peak_equity
            if drawdown <= -self.max_drawdown_pct:
                return self._halt(f"max drawdown do sleeve atingido ({drawdown:.2%})")

        return self.halt_reason  # mantem halt se ja engajado

    def reset_day(self, sleeve_equity: Decimal | None = None) -> None:
        """Reinicia a base do kill diario (novo dia). Drawdown (pico) persiste."""
        if sleeve_equity is not None:
            self.start_capital = sleeve_equity
            if sleeve_equity > self.peak_equity:
                self.peak_equity = sleeve_equity
        else:
            self.start_capital = self.peak_equity
        # Reabre apenas se o halt era do limite diario; drawdown segue valendo.
        self.trading_halted = False
        self.halt_reason = None

    def _halt(self, reason: str) -> str:
        if not self.trading_halted:
            logger.critical("HALT do sleeve cripto: %s", reason)
        self.trading_halted = True
        self.halt_reason = reason
        return reason

    # --- Sizing strategy-agnostic -------------------------------------------
    def allowed_size(
        self,
        *,
        price: Decimal,
        stop_price: Decimal,
        signal: Decimal | float = Decimal("1"),
        open_positions: int = 0,
        current_gross_notional_usd: Decimal = Decimal(0),
    ) -> SizingDecision:
        """Dado (preco, stop, sinal), devolve o tamanho permitido OU bloqueia.

        `signal`: forca/conviccao opcional em [0, 1] (1 = sinal pleno). Strategy
        -agnostic: a estrategia decide o que e o sinal; aqui so escalamos o risco.
        Bloqueios checados em ordem (fail-safe):
          1. kill switch GLOBAL do projeto;
          2. halt do sleeve (perda diaria / drawdown);
          3. limite de posicoes concorrentes;
          4. inputs invalidos (preco/stop);
          5. cap de exposicao bruta do sleeve;
          6. cap de alavancagem efetiva (ceiling rigido).
        """
        sig = Decimal(str(signal))
        if sig < 0:
            sig = Decimal(0)
        if sig > 1:
            sig = Decimal(1)

        # 1. Kill switch GLOBAL — reutilizado, nao recriado.
        if self._kill.is_engaged():
            return SizingDecision(False, Decimal(0), f"kill switch global: {self._kill.reason()}")

        # 2. Halt do sleeve.
        if self.trading_halted:
            return SizingDecision(False, Decimal(0), f"sleeve halted: {self.halt_reason}")

        # 3. Posicoes concorrentes.
        if open_positions >= self.max_concurrent_positions:
            return SizingDecision(
                False,
                Decimal(0),
                f"max posicoes concorrentes atingido ({open_positions} >= "
                f"{self.max_concurrent_positions})",
            )

        # 4. Inputs.
        if price <= 0 or stop_price <= 0 or sig <= 0:
            return SizingDecision(False, Decimal(0), "inputs invalidos (preco/stop/sinal)")

        # Sizing por risco contra o CAPITAL DO SLEEVE (parede de fogo): risco em
        # USD = sleeve_capital * risk_per_trade * sinal. Fractional (cripto).
        risk_pct = self.risk_per_trade_pct * sig
        qty = fixed_fractional_qty(
            self.sleeve_capital_usd, risk_pct, price, stop_price, fractional=True
        )
        if qty <= 0:
            return SizingDecision(False, Decimal(0), "sizing resultou em qty 0")

        # 5. Cap de exposicao bruta do sleeve (notional somado / capital).
        max_gross_notional = self.sleeve_capital_usd * self.max_gross_exposure_x
        room = max_gross_notional - current_gross_notional_usd
        if room <= 0:
            return SizingDecision(
                False, Decimal(0), "exposicao bruta do sleeve esgotada"
            )
        max_qty_gross = room / price
        if qty > max_qty_gross:
            qty = max_qty_gross

        # 6. Cap de alavancagem EFETIVA (ceiling rigido). Notional total apos
        # este trade nao pode implicar leverage > leverage_cap.
        max_notional_lev = self.sleeve_capital_usd * self.leverage_cap
        room_lev = max_notional_lev - current_gross_notional_usd
        if room_lev <= 0:
            return SizingDecision(False, Decimal(0), "cap de alavancagem do sleeve esgotado")
        max_qty_lev = room_lev / price
        if qty > max_qty_lev:
            qty = max_qty_lev

        if qty <= 0:
            return SizingDecision(False, Decimal(0), "qty bloqueada pelos caps")

        notional = qty * price
        implied_lev = (current_gross_notional_usd + notional) / self.sleeve_capital_usd
        return SizingDecision(
            allowed=True,
            qty=qty,
            reason="ok",
            notional_usd=notional,
            implied_leverage=implied_lev,
        )
