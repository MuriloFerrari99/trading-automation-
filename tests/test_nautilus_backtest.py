"""Fase 2 do Nautilus — estrategia/dados REAIS + decomposicao do motor.

Nao testamos "paridade dentro de X pp" (passaria por sorte e esconderia o ponto).
Testamos o INVARIANTE: depois de alinhar a suposicao de execucao, os dois motores
(Nautilus e SimBroker caseiro) batem ao centavo. Concretamente:

  - entrada IDENTICA (mesma barra, mesmo preco, mesma qty);
  - todo o gap restante = custo da suposicao D1 (close[gatilho] vs open[gatilho+1]);
  - residual ~= 0  =>  nenhum dos motores tem bug; a diferenca e a granularidade.

Pula limpo se o cache de SPY ainda nao existe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simulation.nautilus_backtest import crypto_minute, decompose

_SPY_CACHE = Path("data/cache/SPY_1d.csv")
_skip = pytest.mark.skipif(not _SPY_CACHE.exists(), reason="cache de SPY ausente (rode o fetch primeiro)")

_BTC_CACHE = Path("data/binance_1m/BTCUSDT/BTCUSDT-1m-2024-06.csv.gz")
_skip_crypto = pytest.mark.skipif(not _BTC_CACHE.exists(), reason="cache de klines 1min ausente")


@_skip
def test_nautilus_roda_estrategia_real():
    """Smoke: a estrategia portada negocia de fato sobre dados reais."""
    d = decompose("SPY", quiet=True)
    assert d.bars > 100
    assert any(f.side == "BUY" for f in d.nautilus_fills), "Nautilus nao entrou"


@_skip
def test_entrada_identica_entre_motores():
    d = decompose("SPY", quiet=True)
    nt_buy = next(f for f in d.nautilus_fills if f.side == "BUY")
    hm_buy = next(f for f in d.homemade_fills if f.side == "BUY")
    assert nt_buy.bar == hm_buy.bar, "off-by-one: motores entram em barras diferentes"
    assert nt_buy.qty == hm_buy.qty, "arredondamento de qty divergente"
    assert abs(nt_buy.price - hm_buy.price) < 1e-6, "preco de entrada divergente"


@_skip
def test_gap_e_so_a_suposicao_de_execucao():
    """O gap entre motores e 100% atribuivel a suposicao de fill D1 (residual ~0)."""
    d = decompose("SPY", quiet=True)
    assert abs(d.residual_pp) < 1e-4, (
        f"residual {d.residual_pp:.6f}pp != 0 => ha divergencia ALEM da suposicao de "
        f"execucao (possivel bug de motor). gap={d.gap_pp:.4f} custo={d.assumption_cost_pp:.4f}"
    )
    # E o custo da suposicao deve ser material aqui (gatilho na COVID), nao trivial.
    assert d.trigger_bar is not None and abs(d.assumption_cost_pp) > 0.5


@_skip_crypto
def test_intraday_colapsa_a_suposicao_de_execucao():
    """A TESE: trocar barra diaria por 1min encolhe o custo de execucao ~1000x+.

    Em cripto 24/7, close[gatilho] ~= open[gatilho+1] (sem gap overnight), entao a
    suposicao de fill some — e os motores batem ao centavo (residual ~0) mesmo com
    263k barras. Custo zerado dos dois lados (fee do instrumento cripto zerada).
    """
    d = decompose("BTCUSDT", market=crypto_minute("BTCUSDT", months=["2024-06"]), quiet=True)
    assert d.bars > 40_000, "esperado ~43k barras de 1min em 1 mes"
    assert d.trigger_bar is not None, "trailing deveria disparar em junho/2024 (BTC caiu >10%)"
    # residual 0 => sem bug de motor; custo de execucao << 0.5pp do caso diario.
    assert abs(d.residual_pp) < 1e-4, f"residual {d.residual_pp:.6f}pp (fee nao zerada? bug?)"
    assert abs(d.assumption_cost_pp) < 0.01, (
        f"custo de execucao {d.assumption_cost_pp:.4f}pp nao colapsou como esperado em 1min"
    )
