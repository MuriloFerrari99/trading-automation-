"""Modelo de custos de transacao para backtest honesto.

Custo e o inimigo nº1 de qualquer edge. Em cripto na Alpaca a comissao e 0,25% POR TRADE
(~50 bps ida-e-volta antes de slippage) — 10x a fricção de equities, e mata estrategias de
alto giro. Toda avaliacao de edge deve rodar no cenario BASE e no ESTRESSADO (2x) — um edge
que so sobrevive com custo otimista nao e edge.

Refs: docs.alpaca.markets/us/docs/crypto-fees (0,25%); equities Alpaca = comissao zero,
custo vem do spread/slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CostModel:
    """Custo por LADO (fill), em basis points. round_trip = 2x (compra + venda)."""

    commission_bps: float
    slippage_bps: float
    name: str = "custom"

    @property
    def per_side_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_side_bps

    def stressed(self, factor: float = 2.0) -> "CostModel":
        """Cenario de estresse: multiplica comissao e slippage (default 2x)."""
        return replace(
            self,
            commission_bps=self.commission_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            name=f"{self.name}_stress{factor:g}x",
        )


# Equities Alpaca: comissao zero; custo = spread/slippage de execucao.
EQUITY_BASE = CostModel(commission_bps=0.0, slippage_bps=3.0, name="equity_base")

# Cripto Alpaca: 0,25% por trade (25 bps) + slippage em moeda liquida.
CRYPTO_BASE = CostModel(commission_bps=25.0, slippage_bps=10.0, name="crypto_base")

PRESETS = {p.name: p for p in (EQUITY_BASE, CRYPTO_BASE)}
