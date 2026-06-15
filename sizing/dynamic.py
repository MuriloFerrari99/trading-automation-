"""DynamicSizer — tamanho da posicao em funcao do edge/confianca.

Pipeline:
  confianca (P(win)) + razao R:R  --Kelly fracionario-->  risco% do equity
  risco% + equity + entry/stop     --risk/sizing--------->  quantidade (qty)
  qty                              --cap de exposicao----->  qty final

Sem edge (P(win) abaixo do breakeven do R:R) -> Kelly = 0 -> qty 0: o sistema
NAO opera quando nao tem vantagem — a coisa mais inteligente na maioria dos dias.
"""

from __future__ import annotations

from decimal import Decimal

from risk.sizing import fixed_fractional_qty, max_qty_for_exposure


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Fracao de Kelly cheia. b = payoff (ganho/perda); p = P(win).

    f* = (p*b - (1-p)) / b. Negativa (sem edge) -> 0; limitada a [0, 1].
    Edge exige p > 1/(1+b) (ex.: R:R 2 -> p > 0.333).
    """
    b = win_loss_ratio
    if b <= 0:
        return 0.0
    p = max(0.0, min(1.0, win_prob))
    f = (p * b - (1.0 - p)) / b
    return max(0.0, min(1.0, f))


class DynamicSizer:
    def __init__(
        self,
        *,
        kelly_cap: float = 0.25,        # fracao do Kelly cheio (1/4 Kelly)
        max_risk_pct: float = 0.02,     # teto de risco por trade (2% do equity)
        min_risk_pct: float = 0.0025,   # piso quando HA edge (evita "po")
        confidence_floor: float = 0.5,  # abaixo disso, nao opera
        default_win_loss_ratio: float = 2.0,
    ) -> None:
        if not (0 < kelly_cap <= 1):
            raise ValueError("kelly_cap deve estar em (0, 1]")
        self.kelly_cap = kelly_cap
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.confidence_floor = confidence_floor
        self.default_b = default_win_loss_ratio

    def risk_pct(self, confidence: float, win_loss_ratio: float | None = None) -> float:
        """Fracao do equity a arriscar neste trade (0 = nao operar)."""
        if confidence < self.confidence_floor:
            return 0.0
        b = win_loss_ratio if win_loss_ratio is not None else self.default_b
        f = kelly_fraction(confidence, b)
        if f <= 0:
            return 0.0
        risk = f * self.kelly_cap
        risk = min(risk, self.max_risk_pct)
        return max(risk, self.min_risk_pct)

    def qty(
        self,
        *,
        confidence: float,
        equity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal,
        win_loss_ratio: float | None = None,
        max_per_symbol_pct: Decimal | None = None,
        current_qty: Decimal = Decimal(0),
    ) -> Decimal:
        """Quantidade dimensionada pela confianca, com cap opcional de exposicao."""
        rp = self.risk_pct(confidence, win_loss_ratio)
        if rp <= 0:
            return Decimal(0)
        qty = fixed_fractional_qty(
            equity, Decimal(str(rp)), entry_price, stop_price
        )
        if max_per_symbol_pct is not None:
            qty = min(
                qty,
                max_qty_for_exposure(
                    equity, max_per_symbol_pct, entry_price, current_qty
                ),
            )
        return max(qty, Decimal(0))
