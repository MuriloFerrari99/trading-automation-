"""Motor de decisao: aplica regime + politica + registro a cada ciclo.

Fica ENTRE o Planejador (que gera intents) e o Executor (que executa). Para
cada intent:
  1. classifica o regime do ativo (a partir de um buffer de precos rolante);
  2. mede a forca do sinal de smart money ALINHADO (mesmo ativo e lado);
  3. consulta o historico do combo estrategia@regime e aplica a DecisionPolicy;
  4. registra a decisao (executada -> open; vetada -> skip) no DecisionLog;
  5. devolve apenas os intents permitidos.

Sinais NUNCA criam trades aqui (a regra do projeto e respeitada): eles apenas
modulam a confianca de intents que uma estrategia ja produziu.

Desacoplado do broker: `price_provider` e injetado (opcional). Sem ele, o
regime fica UNKNOWN e a politica opera so na dimensao do historico.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from core.models import OptionOrderIntent, OrderIntent, OrderSide, Signal
from feedback.decision_log import DecisionLog
from feedback.evaluation import evaluate
from feedback.models import Decision, DecisionAction, MarketRegime
from feedback.regime import classify_regime
from intelligence.decision_policy import DecisionPolicy, PolicyVerdict
from synthesis.synthesizer import synthesize
from synthesis.views import Direction, View, view_from_signal

logger = logging.getLogger("intelligence.engine")

# Visao fraca derivada do regime (entra na sintese como mais uma fonte).
_REGIME_VIEW = {
    MarketRegime.TREND_UP: Direction.LONG,
    MarketRegime.TREND_DOWN: Direction.SHORT,
}

TradeIntent = OrderIntent | OptionOrderIntent
PriceProvider = Callable[[str], Decimal]

_ACTION = {OrderSide.BUY: DecisionAction.BUY, OrderSide.SELL: DecisionAction.SELL}


@dataclass
class ProcessResult:
    allowed: list[TradeIntent]
    verdicts: list[PolicyVerdict] = field(default_factory=list)
    blocked_count: int = 0


def _extract(intent: TradeIntent) -> tuple[str, OrderSide, str, str | None]:
    """(symbol, side, strategy, client_order_id) para acoes e opcoes."""
    if isinstance(intent, OptionOrderIntent):
        return intent.underlying, intent.side, intent.strategy, intent.client_order_id
    return intent.symbol, intent.side, intent.strategy, intent.client_order_id


class DecisionIntelligence:
    def __init__(
        self,
        decision_log: DecisionLog,
        policy: DecisionPolicy | None = None,
        *,
        price_provider: PriceProvider | None = None,
        history_len: int = 60,
        regime_kwargs: dict | None = None,
    ) -> None:
        self._log = decision_log
        self._policy = policy or DecisionPolicy()
        self._price_provider = price_provider
        self._regime_kwargs = regime_kwargs or {}
        # Buffer de precos por simbolo (amostrado a cada ciclo). Sem API de
        # barras: o regime emerge das amostras acumuladas ao longo dos ticks.
        self._history: dict[str, deque[float]] = {}
        self._history_len = history_len

    # ------------------------------------------------------------------ #

    def _regime_for(
        self, symbol: str, market_data: dict[str, list] | None
    ) -> tuple[MarketRegime, Decimal | None]:
        """Classifica o regime do simbolo.

        Preferencia: barras reais do `market_data` (IngestionAgent) — uma serie
        completa por ciclo, regime confiavel ja no 1o tick. Sem barras, cai no
        fallback de amostragem (1 preco/ciclo acumulado no buffer)."""
        bars = market_data.get(symbol) if market_data else None
        if bars:
            closes = [float(b) for b in bars]
            price = Decimal(str(bars[-1]))
            return classify_regime(closes, **self._regime_kwargs), price
        return self._update_and_regime(symbol)

    def _update_and_regime(self, symbol: str) -> tuple[MarketRegime, Decimal | None]:
        price: Decimal | None = None
        if self._price_provider is not None:
            try:
                price = self._price_provider(symbol)
            except Exception:  # provider de preco nao deve derrubar o ciclo
                logger.exception("price_provider falhou para %s", symbol)
                price = None
        if price is not None:
            buf = self._history.setdefault(symbol, deque(maxlen=self._history_len))
            buf.append(float(price))
        buf = self._history.get(symbol)
        if not buf:
            return MarketRegime.UNKNOWN, price
        return classify_regime(list(buf), **self._regime_kwargs), price

    @staticmethod
    def _signal_strength(
        symbol: str, side: OrderSide, signals: list[Signal]
    ) -> float:
        """Maior confianca de um sinal ALINHADO (mesmo ativo e lado). Sinais de
        lado oposto sao ignorados aqui (nao reduzem; so nao somam)."""
        aligned = [
            s.confidence
            for s in signals
            if s.symbol.upper() == symbol.upper() and s.side == side
        ]
        return max(aligned) if aligned else 0.0

    @staticmethod
    def _synthesis_context(
        symbol: str, signals: list[Signal], regime: MarketRegime
    ) -> dict:
        """Sintese (Camada 3) das visoes disponiveis -> contexto p/ o log.

        ADITIVO: enriquece o `context` com conviccao/conflito/racional. NAO
        dimensiona (sizing e responsabilidade do RiskManager no Planner) e NAO
        veta — quem veta e a DecisionPolicy. So da observabilidade e prepara o
        terreno para o ML/sizing usarem a sintese depois."""
        views = [
            view_from_signal(s) for s in signals if s.symbol.upper() == symbol.upper()
        ]
        rdir = _REGIME_VIEW.get(regime)
        if rdir is not None:
            views.append(View("regime", rdir, 0.55, 0.5, f"regime {regime.value}"))
        if not views:
            return {}
        syn = synthesize(symbol, views)
        return {
            "synthesis_direction": syn.direction.value,
            "conviction": round(syn.conviction, 4),
            "conflict": syn.conflict,
            "rationale": syn.rationale,
        }

    def process(
        self,
        intents: list[TradeIntent],
        signals: list[Signal] | None = None,
        market_data: dict[str, list] | None = None,
    ) -> ProcessResult:
        signals = signals or []
        if not intents:
            return ProcessResult(allowed=[])

        # Historico avaliado UMA vez por ciclo (estado do track record).
        report = evaluate(self._log.all_records())

        # Regime calculado uma vez por simbolo (evita reamostrar o preco).
        # Usa barras reais do market_data quando disponiveis (regime confiavel
        # ja no 1o ciclo); senao cai no fallback de amostragem por preco.
        regimes: dict[str, tuple[MarketRegime, Decimal | None]] = {}
        for intent in intents:
            symbol, _, _, _ = _extract(intent)
            if symbol not in regimes:
                regimes[symbol] = self._regime_for(symbol, market_data)

        allowed: list[TradeIntent] = []
        verdicts: list[PolicyVerdict] = []
        blocked = 0

        for intent in intents:
            symbol, side, strategy, coid = _extract(intent)
            regime, price = regimes[symbol]
            strength = self._signal_strength(symbol, side, signals)
            combo = report.combo(strategy, regime.value)
            verdict = self._policy.evaluate_combo(combo, signal_strength=strength)
            verdicts.append(verdict)

            action = _ACTION[side] if verdict.allow else DecisionAction.SKIP
            context = {
                "regime": regime.value,
                "policy_reason": verdict.reason,
                "score": round(verdict.score, 4),
                "signal_strength": round(strength, 4),
            }
            context.update(self._synthesis_context(symbol, signals, regime))
            self._log.record(
                Decision(
                    strategy=strategy,
                    symbol=symbol,
                    action=action,
                    regime=regime,
                    reference_price=price,
                    signal_strength=strength,
                    context=context,
                    client_order_id=coid,
                )
            )

            if verdict.allow:
                allowed.append(intent)
            else:
                blocked += 1
                logger.info(
                    "VETO [%s %s @%s] %s", strategy, symbol, regime.value, verdict.reason
                )

        if blocked:
            logger.info(
                "Camada de decisao: %d permitido(s), %d vetado(s).",
                len(allowed), blocked,
            )
        return ProcessResult(allowed=allowed, verdicts=verdicts, blocked_count=blocked)
