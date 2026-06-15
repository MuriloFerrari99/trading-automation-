"""Testes do classificador de regime (deterministico)."""

from __future__ import annotations

from feedback.regime import classify_regime
from feedback.models import MarketRegime


def test_dados_insuficientes_retorna_unknown():
    assert classify_regime([100, 101, 102]) == MarketRegime.UNKNOWN


def test_tendencia_de_alta():
    closes = [100 + i for i in range(60)]  # subida linear forte
    assert classify_regime(closes) == MarketRegime.TREND_UP


def test_tendencia_de_baixa():
    closes = [200 - i for i in range(60)]  # queda linear forte
    assert classify_regime(closes) == MarketRegime.TREND_DOWN


def test_mercado_lateral():
    # oscila em torno de 100 sem direcao -> RANGE
    closes = [100 + (1 if i % 2 == 0 else -1) for i in range(60)]
    assert classify_regime(closes) == MarketRegime.RANGE


def test_alta_volatilidade_tem_prioridade():
    # serie longa e calma e, no fim, um choque que preenche a janela de vol (20)
    closes = [100 + (0.1 if i % 2 == 0 else -0.1) for i in range(100)]
    closes += [80, 170] * 12  # 24 pontos de altissima volatilidade no fim
    assert classify_regime(closes) == MarketRegime.HIGH_VOL
