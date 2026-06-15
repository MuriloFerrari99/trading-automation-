"""DecisionEnricher — compoe FimatheEngine + sintese + ML + sizing.

Dado um intent (symbol/side/strategy + entry/stop), o regime e o que estiver
disponivel (barras OHLC, sinais de smart money, modelo de ML promovido),
produz:
  - contexto enriquecido (features da FimatheEngine + resumo da sintese) p/ log,
  - confianca [0..1] alinhada ao lado pretendido,
  - quantidade dimensionada por essa confianca (DynamicSizer).

Degrada com graca: sem barras -> sem features FimatheEngine; sem modelo
promovido -> sem P(win). NUNCA cria trades: enriquece/score/sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from core.models import OrderSide, Signal
from feedback.models import MarketRegime
from fimathe.engine import FimatheEngine
from ml.dataset import context_to_features
from ml.setup_classifier import SetupClassifier
from sizing.dynamic import DynamicSizer
from synthesis.synthesizer import SynthesisResult, Synthesizer
from synthesis.views import Direction, View, view_from_proba, view_from_signal

# Features da FimatheEngine que entram no contexto/ML (subconjunto de ML_FEATURES).
_FIM_KEYS = ("pcm_score", "dist_to_zn", "breakout_strength", "rsi", "adx")


@dataclass
class EnrichedDecision:
    confidence: float
    qty: Decimal
    context: dict
    synthesis: SynthesisResult
    p_win: float | None = None
    fimathe_signal: int | None = None


def _side_sign(side: OrderSide) -> int:
    return 1 if side == OrderSide.BUY else -1


class DecisionEnricher:
    def __init__(
        self,
        *,
        fimathe: FimatheEngine | None = None,
        synthesizer: Synthesizer | None = None,
        sizer: DynamicSizer | None = None,
        classifier: SetupClassifier | None = None,
        regime_view_weight: float = 0.5,
        min_bars: int | None = None,
    ) -> None:
        self.fimathe = fimathe or FimatheEngine()
        self.synth = synthesizer or Synthesizer()
        self.sizer = sizer or DynamicSizer()
        # SetupClassifier PROMOVIDO (champion/challenger). None = ML em shadow.
        self.classifier = classifier
        self.regime_view_weight = regime_view_weight
        self.min_bars = min_bars or (self.fimathe.swing_period + 5)

    # ------------------------------------------------------------------ #

    def _fimathe(self, ohlc: pd.DataFrame | None) -> tuple[tuple[int, float] | None, dict]:
        if ohlc is None or len(ohlc) < self.min_bars:
            return None, {}
        out = self.fimathe.process(ohlc)
        last = out.iloc[-1]
        feats = {
            k: float(last[k])
            for k in _FIM_KEYS
            if k in out.columns and pd.notna(last[k])
        }
        sig = int(last["signal"]) if pd.notna(last["signal"]) else 0
        strength = float(last["signal_strength"]) if pd.notna(last["signal_strength"]) else 0.0
        return (sig, strength), feats

    def _regime_view(self, regime: MarketRegime) -> View | None:
        if regime == MarketRegime.TREND_UP:
            return View("regime", Direction.LONG, 0.55, self.regime_view_weight, "regime trend_up")
        if regime == MarketRegime.TREND_DOWN:
            return View("regime", Direction.SHORT, 0.55, self.regime_view_weight, "regime trend_down")
        return None

    def enrich(
        self,
        *,
        symbol: str,
        side: OrderSide,
        strategy: str,
        regime: MarketRegime,
        equity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal,
        signals: list[Signal] | None = None,
        ohlc: pd.DataFrame | None = None,
        signal_strength: float = 0.0,
        win_loss_ratio: float = 2.0,
        max_per_symbol_pct: Decimal | None = None,
        current_qty: Decimal = Decimal(0),
    ) -> EnrichedDecision:
        signals = signals or []
        fim, fim_feats = self._fimathe(ohlc)

        # ---- visoes ortogonais ----
        views = [
            view_from_signal(s) for s in signals if s.symbol.upper() == symbol.upper()
        ]
        if fim and fim[0] != 0:
            d = Direction.LONG if fim[0] > 0 else Direction.SHORT
            views.append(View("fimathe", d, fim[1], 1.0, "FimatheEngine"))
        rv = self._regime_view(regime)
        if rv:
            views.append(rv)

        # ---- P(win) do ML (so se PROMOVIDO); features pre-decisao, sem leak ----
        p_win: float | None = None
        if self.classifier is not None and self.classifier.is_trained:
            ml_ctx = {"signal_strength": signal_strength, **fim_feats}
            x = context_to_features(ml_ctx, regime.value)
            p_win = float(self.classifier.predict_proba(x)[0])
            views.append(view_from_proba("ml", p_win, side))

        # ---- sintese ----
        syn = self.synth.synthesize(symbol, views)

        # ---- confianca alinhada ao lado pretendido ----
        sign = _side_sign(side)
        if syn.direction == Direction.NEUTRAL:
            syn_conf = 0.5
        elif syn.direction.sign() == sign:
            syn_conf = 0.5 + 0.5 * syn.conviction        # a favor: 0.5..1
        else:
            syn_conf = 0.5 - 0.5 * syn.conviction        # contra: 0..0.5
        if p_win is not None:
            confidence = 0.5 * syn_conf + 0.5 * p_win    # blend com o ML promovido
        else:
            confidence = syn_conf
        confidence = max(0.0, min(1.0, confidence))

        # ---- sizing pela confianca ----
        qty = self.sizer.qty(
            confidence=confidence,
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            win_loss_ratio=win_loss_ratio,
            max_per_symbol_pct=max_per_symbol_pct,
            current_qty=current_qty,
        )

        context = {
            **fim_feats,
            "signal_strength": round(signal_strength, 4),
            "synthesis_direction": syn.direction.value,
            "conviction": round(syn.conviction, 4),
            "conflict": syn.conflict,
            "confidence": round(confidence, 4),
            "p_win": None if p_win is None else round(p_win, 4),
            "rationale": syn.rationale,
        }
        return EnrichedDecision(
            confidence=confidence,
            qty=qty,
            context=context,
            synthesis=syn,
            p_win=p_win,
            fimathe_signal=(fim[0] if fim else None),
        )
