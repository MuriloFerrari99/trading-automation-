"""Politica de decisao — o "gate" que usa o track record para filtrar trades.

Pura e sem I/O (recebe as estatisticas ja calculadas): facil de testar e de
raciocinar. A regra central, conservadora de proposito:

  - SEM dados suficientes (poucos trades naquele combo estrategia@regime) ->
    NAO bloqueia (inocente ate prova em contrario); confianca = baseline.
  - COM amostra suficiente E expectancy < limiar -> BLOQUEIA (registra como
    'skip'). E o sistema aprendendo a NAO repetir o que vem perdendo.

A confianca final (`score`) mistura o win-rate historico do combo com a forca
do sinal de smart money alinhado (quando houver). Score so prioriza/loga; quem
decide operar/vetar e o `allow`.
"""

from __future__ import annotations

from dataclasses import dataclass

from feedback.evaluation import GroupStats


@dataclass(frozen=True)
class PolicyVerdict:
    allow: bool
    score: float  # 0..1 — confianca final (apenas prioriza/loga)
    reason: str


class DecisionPolicy:
    def __init__(
        self,
        *,
        min_samples: int = 12,
        block_if_expectancy_below: float = 0.0,
        baseline_score: float = 0.5,
    ) -> None:
        # Quantos trades fechados num combo antes de confiar no historico dele.
        self._min_samples = min_samples
        # Expectancy (P&L medio/trade) abaixo da qual o combo e vetado.
        self._block_below = block_if_expectancy_below
        self._baseline = baseline_score

    def evaluate_combo(
        self,
        combo: GroupStats | None,
        *,
        signal_strength: float = 0.0,
    ) -> PolicyVerdict:
        """Decide se um combo estrategia@regime pode operar, dado o historico.

        `combo` e o GroupStats historico daquele (estrategia, regime), ou None
        se nunca houve trade fechado nele. `signal_strength` (0..1) e a forca do
        sinal de smart money alinhado, usada apenas no score.
        """
        if combo is None or combo.n_trades < self._min_samples:
            n = 0 if combo is None else combo.n_trades
            score = _blend(self._baseline, signal_strength)
            return PolicyVerdict(
                allow=True,
                score=score,
                reason=f"amostra insuficiente (n={n}<{self._min_samples}); permitido",
            )

        if combo.expectancy < self._block_below:
            return PolicyVerdict(
                allow=False,
                score=_blend(combo.win_rate, signal_strength),
                reason=(
                    f"VETADO: expectancy {combo.expectancy:.2f} < "
                    f"{self._block_below:.2f} em {combo.n_trades} trades "
                    f"(win%={combo.win_rate * 100:.0f})"
                ),
            )

        return PolicyVerdict(
            allow=True,
            score=_blend(combo.win_rate, signal_strength),
            reason=(
                f"OK: expectancy {combo.expectancy:.2f} em {combo.n_trades} trades "
                f"(win%={combo.win_rate * 100:.0f})"
            ),
        )


def _blend(base: float, signal_strength: float) -> float:
    """Mistura conservadora: o sinal de smart money desloca a confianca base
    em ate +/-0.2, sem nunca dominar a decisao. Resultado limitado a [0,1]."""
    score = base + 0.4 * (signal_strength - 0.5)
    return max(0.0, min(1.0, score))
