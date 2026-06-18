"""Reasoner — gera o RACIONAL legivel da sintese.

`Reasoner` e uma interface (Protocol): o nucleo deterministico usa o
`TemplateReasoner`; um `LLMReasoner` opcional pode gerar narrativa qualitativa
(noticias, earnings) chamando um LLM que VOCE injeta — sem acoplar o loop de
trading a nenhuma API. Se a chamada do LLM falhar, cai no template (nunca
derruba o ciclo).
"""

from __future__ import annotations

from typing import Callable, Protocol

from pydantic import BaseModel, Field

from synthesis.observability import observe_generation
from synthesis.structured import structured_call
from synthesis.views import Direction, View


class NarrativaSintese(BaseModel):
    """Racional do LLM TIPADO e validado (Pydantic) — auditavel, sem prosa solta.

    `narrativa` e o texto curto consumido a jusante; `drivers`/`conflitos` deixam
    explicito o porque, e o que o LLM viu de conflitante entre as fontes.
    """

    narrativa: str = Field(description="Visao de trading em 1-2 frases, objetiva.")
    drivers: list[str] = Field(default_factory=list, description="Principais drivers da direcao.")
    conflitos: list[str] = Field(
        default_factory=list, description="Conflitos/discordancias entre as fontes."
    )


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
        # Env-gated: traca a geracao no Langfuse quando ha chaves; senao, no-op.
        self._call = observe_generation("synthesis.llm_reasoner")(call_fn)
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


class StructuredLLMReasoner:
    """Como o LLMReasoner, mas a saida do LLM e TIPADA e VALIDADA (Pydantic).

    Forca JSON -> valida contra NarrativaSintese (narrativa + drivers + conflitos),
    com reparo e fallback no template (nunca quebra o ciclo). `explain` segue
    devolvendo a string (narrativa) para o downstream; `explain_structured` expoe
    o objeto auditavel completo.
    """

    def __init__(self, call_fn: Callable[[str], str]) -> None:
        self._call = observe_generation("synthesis.llm_reasoner.structured")(call_fn)
        self._fallback = TemplateReasoner()

    def explain_structured(
        self,
        symbol: str,
        views: list[View],
        *,
        direction: Direction,
        conviction: float,
        agreement: float,
    ) -> NarrativaSintese:
        prompt = LLMReasoner._prompt(self, symbol, views, direction, conviction, agreement)
        default = NarrativaSintese(
            narrativa=self._fallback.explain(
                symbol, views, direction=direction,
                conviction=conviction, agreement=agreement,
            )
        )
        return structured_call(self._call, prompt, NarrativaSintese, default=default)

    def explain(
        self,
        symbol: str,
        views: list[View],
        *,
        direction: Direction,
        conviction: float,
        agreement: float,
    ) -> str:
        return self.explain_structured(
            symbol, views, direction=direction,
            conviction=conviction, agreement=agreement,
        ).narrativa.strip()
