"""Visoes (Views) — a unidade de entrada da sintese.

Cada FONTE (estrategia, smart money, ML, regime, noticia) emite uma View:
direcao + confianca + peso + racional. O sintetizador as combina. Fontes sao
ortogonais de proposito — diversidade e o que da inteligencia ao conjunto.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.models import OrderSide, Signal


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

    def sign(self) -> int:
        return {Direction.LONG: 1, Direction.SHORT: -1, Direction.NEUTRAL: 0}[self]


@dataclass(frozen=True)
class View:
    """Visao de UMA fonte sobre um ativo."""

    source: str
    direction: Direction
    confidence: float = 0.5  # 0..1
    weight: float = 1.0      # importancia relativa da fonte
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))
        object.__setattr__(self, "weight", max(0.0, self.weight))


# ----------------------------- adaptadores ----------------------------- #

def view_from_signal(signal: Signal, *, weight: float = 1.0) -> View:
    """Converte um Signal de smart money em View."""
    direction = Direction.LONG if signal.side == OrderSide.BUY else Direction.SHORT
    return View(
        source=signal.source,
        direction=direction,
        confidence=signal.confidence,
        weight=weight,
        rationale=signal.note or f"sinal {signal.source}",
    )


def view_from_proba(
    source: str, p_win: float, side: OrderSide, *, weight: float = 1.0
) -> View:
    """Converte um P(win) de modelo (ML) + lado pretendido em View.

    p_win e a confianca; a direcao vem do lado do trade que o modelo avalia.
    """
    direction = Direction.LONG if side == OrderSide.BUY else Direction.SHORT
    return View(
        source=source,
        direction=direction,
        confidence=p_win,
        weight=weight,
        rationale=f"{source} P(win)={p_win:.2f}",
    )
