"""Synthesizer — combina visoes ortogonais numa conviccao unificada.

Pondera cada View por (peso x confianca x direcao) e produz:
- direcao agregada (long/short/neutral) com uma banda neutra,
- conviccao 0..1 (forca do consenso liquido),
- agreement: consenso ponderado COM a direcao final (detecta conflito),
- racional (via Reasoner, deterministico por padrao).

E uma camada de SCORE/VISAO: informa/modula decisoes; nunca executa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthesis.reasoner import Reasoner, TemplateReasoner
from synthesis.views import Direction, View


@dataclass
class SynthesisResult:
    symbol: str
    direction: Direction
    conviction: float                 # 0..1
    agreement: float                  # 0..1 (consenso com a direcao final)
    n_views: int
    conflict: bool
    contributions: list[tuple[str, float]] = field(default_factory=list)
    rationale: str = ""


def synthesize(
    symbol: str,
    views: list[View],
    *,
    reasoner: Reasoner | None = None,
    neutral_band: float = 0.05,
    conflict_threshold: float = 0.6,
) -> SynthesisResult:
    reasoner = reasoner or TemplateReasoner()

    if not views:
        return SynthesisResult(
            symbol=symbol, direction=Direction.NEUTRAL, conviction=0.0,
            agreement=0.0, n_views=0, conflict=False,
            rationale=f"{symbol}: sem visoes.",
        )

    total_w = sum(v.weight for v in views) or 1.0
    net = sum(v.weight * v.confidence * v.direction.sign() for v in views) / total_w

    if net > neutral_band:
        direction = Direction.LONG
    elif net < -neutral_band:
        direction = Direction.SHORT
    else:
        direction = Direction.NEUTRAL

    conviction = min(1.0, abs(net))

    # agreement: fracao da "massa" (peso x confianca) que aponta na direcao final.
    total_cw = sum(v.weight * v.confidence for v in views) or 1.0
    if direction == Direction.NEUTRAL:
        agreement = 0.0
    else:
        agree_cw = sum(
            v.weight * v.confidence for v in views if v.direction == direction
        )
        agreement = agree_cw / total_cw

    conflict = direction != Direction.NEUTRAL and agreement < conflict_threshold
    contributions = [
        (v.source, round(v.weight * v.confidence * v.direction.sign(), 4))
        for v in views
    ]
    rationale = reasoner.explain(
        symbol, views, direction=direction,
        conviction=conviction, agreement=agreement,
    )
    return SynthesisResult(
        symbol=symbol, direction=direction, conviction=conviction,
        agreement=agreement, n_views=len(views), conflict=conflict,
        contributions=contributions, rationale=rationale,
    )


class Synthesizer:
    """Wrapper com config fixa (peso default por fonte, reasoner)."""

    def __init__(
        self,
        *,
        reasoner: Reasoner | None = None,
        neutral_band: float = 0.05,
        conflict_threshold: float = 0.6,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self._reasoner = reasoner or TemplateReasoner()
        self._neutral_band = neutral_band
        self._conflict_threshold = conflict_threshold
        self._weights = source_weights or {}

    def synthesize(self, symbol: str, views: list[View]) -> SynthesisResult:
        # aplica peso por fonte (se configurado) sobre o peso da view
        weighted = [
            View(
                source=v.source, direction=v.direction, confidence=v.confidence,
                weight=v.weight * self._weights.get(v.source, 1.0),
                rationale=v.rationale,
            )
            for v in views
        ]
        return synthesize(
            symbol, weighted, reasoner=self._reasoner,
            neutral_band=self._neutral_band,
            conflict_threshold=self._conflict_threshold,
        )
