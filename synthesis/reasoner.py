"""Reasoner — gera o RACIONAL legivel da sintese.

`Reasoner` e uma interface (Protocol): o nucleo deterministico usa o
`TemplateReasoner`; um `LLMReasoner` opcional pode gerar narrativa qualitativa
(noticias, earnings) chamando um LLM que VOCE injeta — sem acoplar o loop de
trading a nenhuma API. Se a chamada do LLM falhar, cai no template (nunca
derruba o ciclo).
"""

from __future__ import annotations

from typing import Callable, Protocol

from synthesis.views import Direction, View


class Reasoner(Protocol):
    def explain(
        self,
        symbol: str,
        views: list[View],
        *,
        direction: Direction,
        conviction: float,
        agreement: float,
    ) -> str: ...


def _fmt_views(views: list[View], direction: Direction) -> tuple[str, str]:
    favor = [v for v in views if v.direction == direction and direction != Direction.NEUTRAL]
    contra = [v for v in views if v.direction.sign() == -direction.sign() and direction != Direction.NEUTRAL]
    fav = ", ".join(f"{v.source}({v.confidence:.2f})" for v in favor) or "—"
    con = ", ".join(f"{v.source}({v.confidence:.2f})" for v in contra) or "—"
    return fav, con


class TemplateReasoner:
    """Racional deterministico (sem dependencia)."""

    def explain(
        self,
        symbol: str,
        views: list[View],
        *,
        direction: Direction,
        conviction: float,
        agreement: float,
    ) -> str:
        fav, con = _fmt_views(views, direction)
        return (
            f"{symbol}: {direction.value.upper()} "
            f"(convicção {conviction:.2f}, consenso {agreement * 100:.0f}%). "
            f"A favor: {fav}. Contra: {con}."
        )


class LLMReasoner:
    """Racional via LLM injetado. `call_fn(prompt) -> str`. Fallback no template."""

    def __init__(self, call_fn: Callable[[str], str]) -> None:
        self._call = call_fn
        self._fallback = TemplateReasoner()

    def _prompt(self, symbol, views, direction, conviction, agreement) -> str:
        linhas = "\n".join(
            f"- {v.source}: {v.direction.value} conf={v.confidence:.2f} "
            f"peso={v.weight:.2f} — {v.rationale}"
            for v in views
        )
        return (
            f"Sintetize uma visao de trading para {symbol}.\n"
            f"Direcao agregada: {direction.value} | conviccao {conviction:.2f} | "
            f"consenso {agreement:.2f}.\nVisoes das fontes:\n{linhas}\n"
            f"Responda em 1-2 frases, objetivo, citando os principais drivers e conflitos."
        )

    def explain(
        self,
        symbol: str,
        views: list[View],
        *,
        direction: Direction,
        conviction: float,
        agreement: float,
    ) -> str:
        try:
            return self._call(
                self._prompt(symbol, views, direction, conviction, agreement)
            ).strip()
        except Exception:  # LLM indisponivel/erro -> nunca derruba o ciclo
            return self._fallback.explain(
                symbol, views, direction=direction,
                conviction=conviction, agreement=agreement,
            )
