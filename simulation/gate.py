"""FrozenRegimeGate — gate de regime CONGELADO, treinado in-sample.

Treina-se a camada de decisao no periodo IN-SAMPLE (acumulando track record por
estrategia@regime via DecisionLog), congela-se o conjunto de combos com edge
negativo comprovado, e avalia-se OUT-OF-SAMPLE. Sem re-treino no OOS => sem
look-ahead e rapido (O(1) por consulta), viabilizando dezenas de milhares de
backtests.
"""

from __future__ import annotations

from feedback.decision_log import DecisionLog
from feedback.evaluation import evaluate


class FrozenRegimeGate:
    def __init__(self, vetoed: set[tuple[str, str]]) -> None:
        self._vetoed = set(vetoed)

    @property
    def vetoed(self) -> set[tuple[str, str]]:
        return set(self._vetoed)

    def allow(self, strategy: str, regime: str) -> bool:
        """Permite operar este combo estrategia@regime? (veta os de edge negativo)."""
        return (strategy, regime) not in self._vetoed

    @classmethod
    def from_decision_log(
        cls,
        dlog: DecisionLog,
        *,
        min_samples: int = 12,
        max_expectancy: float = 0.0,
    ) -> "FrozenRegimeGate":
        """Constroi o gate a partir do track record aprendido no IN-SAMPLE.

        Veta combos com amostra suficiente (>= min_samples trades fechados) E
        expectancy abaixo do limiar (default < 0).
        """
        report = evaluate(dlog.all_records())
        vetoed = {
            key
            for key, stats in report.by_strategy_regime.items()
            if stats.n_trades >= min_samples and stats.expectancy < max_expectancy
        }
        return cls(vetoed)

    def describe(self) -> str:
        if not self._vetoed:
            return "(nenhum combo vetado)"
        return ", ".join(f"{s}@{r}" for (s, r) in sorted(self._vetoed))
