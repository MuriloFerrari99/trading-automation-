"""Classificador de regime de mercado — deterministico e explicavel.

Recebe uma serie de precos de fechamento e devolve um MarketRegime. E
proposital que seja rule-based (sem ML): serve como baseline explicavel e como
SEMENTE da Camada 2 (deteccao de regime via ML pode vir depois e ser comparada
contra este baseline). Sem dependencias externas (so a stdlib `statistics`).

Heuristica:
  1. Volatilidade recente >> volatilidade historica  -> HIGH_VOL
  2. SMA rapida > SMA lenta e tendencia para cima      -> TREND_UP
  3. SMA rapida < SMA lenta e tendencia para baixo     -> TREND_DOWN
  4. caso contrario                                    -> RANGE
Dados insuficientes -> UNKNOWN.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from feedback.models import MarketRegime


def _sma(values: Sequence[float], window: int) -> float:
    return statistics.fmean(values[-window:])


def _returns(closes: Sequence[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev:
            out.append((cur - prev) / prev)
    return out


def classify_regime(
    closes: Sequence[float],
    *,
    fast: int = 10,
    slow: int = 30,
    vol_window: int = 20,
    trend_threshold: float = 0.005,
    high_vol_ratio: float = 1.6,
) -> MarketRegime:
    """Classifica o regime a partir dos fechamentos (ordem cronologica).

    Parametros sao configuraveis por timeframe/par (ver doc 08, "Adaptabilidade").
    `trend_threshold`: variacao minima (fracao) ao longo de `slow` para contar
    como tendencia. `high_vol_ratio`: quao maior a vol recente precisa ser vs a
    historica para marcar HIGH_VOL.
    """
    closes = [float(c) for c in closes]
    if len(closes) < slow + 1:
        return MarketRegime.UNKNOWN

    rets = _returns(closes)
    if len(rets) < vol_window + 1:
        return MarketRegime.UNKNOWN

    recent_vol = statistics.pstdev(rets[-vol_window:])
    hist_vol = statistics.pstdev(rets)
    if hist_vol > 0 and recent_vol > high_vol_ratio * hist_vol:
        return MarketRegime.HIGH_VOL

    fast_sma = _sma(closes, fast)
    slow_sma = _sma(closes, slow)
    # Inclinacao normalizada ao longo da janela lenta.
    ref = closes[-slow]
    slope = (closes[-1] - ref) / ref if ref else 0.0

    if fast_sma > slow_sma and slope > trend_threshold:
        return MarketRegime.TREND_UP
    if fast_sma < slow_sma and slope < -trend_threshold:
        return MarketRegime.TREND_DOWN
    return MarketRegime.RANGE
