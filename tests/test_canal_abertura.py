"""Testes do motor CANAL DE ABERTURA (FIMATHE intraday) — invariantes mecanicos.

Cobre: (1) a caixa = extremos das 4 primeiras velas M15 da sessao; (2) direcao e
niveis do rompimento (stop fora da ZN, alvos por clone 1x/2x); (3) SEM LOOK-AHEAD
(sinal so APOS a 4a vela e NUNCA na ultima barra da sessao); (4) fronteiras de
sessao (dia UTC) e gatilho de liquidacao intraday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fimathe.canal_abertura import CanalAbertura, CanalAberturaParams


def _session_m15(day: str, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Constroi um dia de M15 (UTC) a partir de tuplas (open,high,low,close)."""
    idx = pd.date_range(f"{day} 00:00", periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def test_caixa_e_extremos_das_4_primeiras_velas():
    # 4 velas de abertura com extremos conhecidos -> caixa [100, 105]
    bars = [
        (101, 103, 100, 102),  # low 100
        (102, 105, 101, 104),  # high 105
        (104, 104, 102, 103),
        (103, 104, 102, 103),
        (103, 110, 103, 109),  # 5a: rompe pra cima (fecha 109 > 105)
        (109, 111, 108, 110),
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    fixed = out[out.ca_fixed]
    assert fixed.ca_top.iloc[0] == pytest.approx(105.0)
    assert fixed.ca_bottom.iloc[0] == pytest.approx(100.0)
    assert fixed.ca_range.iloc[0] == pytest.approx(5.0)


def test_rompimento_de_alta_niveis_corretos():
    bars = [
        (101, 103, 100, 102),
        (102, 105, 101, 104),
        (104, 104, 102, 103),
        (103, 104, 102, 103),
        (103, 110, 103, 109),  # rompe acima de 105
        (109, 111, 108, 110),
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    sig = out[out.signal != 0]
    assert len(sig) == 1
    r = sig.iloc[0]
    assert int(r.signal) == 1  # compra
    # stop = base - 1*range = 100 - 5 = 95 (fora da ZN, abaixo)
    assert r.stop_loss == pytest.approx(95.0)
    # alvos a partir da borda rompida (top=105): tp1=110 (+1x), tp2=115 (+2x)
    assert r.take_profit_1 == pytest.approx(110.0)
    assert r.take_profit_2 == pytest.approx(115.0)


def test_rompimento_de_baixa_niveis_corretos():
    bars = [
        (104, 105, 102, 103),
        (103, 104, 100, 101),  # low 100
        (101, 102, 100, 101),
        (101, 102, 100, 101),
        (101, 101, 94, 95),    # rompe abaixo de 100 (fecha 95 < 100)
        (95, 96, 93, 94),
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    sig = out[out.signal != 0]
    assert len(sig) == 1
    r = sig.iloc[0]
    assert int(r.signal) == -1  # venda
    # caixa: top=105 (1a vela), bottom=100 -> range 5
    # stop = top + range = 105 + 5 = 110 (fora da ZN, acima)
    assert r.stop_loss == pytest.approx(110.0)
    # alvos a partir da borda rompida (bottom=100): tp1=95 (-1x), tp2=90 (-2x)
    assert r.take_profit_1 == pytest.approx(95.0)
    assert r.take_profit_2 == pytest.approx(90.0)


def test_nao_opera_durante_formacao_do_CA():
    # rompimento "falso" DENTRO das 4 primeiras velas nao pode gerar sinal
    bars = [
        (100, 120, 100, 119),  # vela 0 ja "rompe" mas e parte do CA
        (119, 121, 118, 120),
        (120, 122, 119, 121),
        (121, 123, 120, 122),
        (122, 124, 121, 123),  # so a partir daqui pode haver sinal
        (123, 125, 122, 124),
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    # nenhum sinal nas 4 primeiras barras (bar_in_session 0..3)
    early = out[out.bar_in_session < 4]
    assert int((early.signal != 0).sum()) == 0


def test_uma_entrada_por_sessao():
    # multiplos fechamentos acima da caixa -> apenas 1 sinal (o primeiro)
    bars = [
        (101, 103, 100, 102),
        (102, 105, 101, 104),
        (104, 104, 102, 103),
        (103, 104, 102, 103),
        (103, 110, 103, 109),  # 1o rompimento
        (109, 112, 108, 111),  # ainda acima — nao deve re-sinalizar
        (111, 113, 110, 112),
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    assert int((out.signal != 0).sum()) == 1


def test_nao_sinaliza_na_ultima_barra_da_sessao():
    # rompimento so na ULTIMA barra: sem barra seguinte na mesma sessao -> sem entrada
    bars = [
        (101, 103, 100, 102),
        (102, 105, 101, 104),
        (104, 104, 102, 103),
        (103, 104, 102, 103),
        (103, 104, 102, 103),
        (103, 110, 103, 109),  # ULTIMA barra rompe -> motor nao sinaliza (range j-1)
    ]
    out = CanalAbertura().process(_session_m15("2024-01-02", bars))
    assert int((out.signal != 0).sum()) == 0
    assert bool(out.is_session_last.iloc[-1]) is True


def test_sessoes_por_dia_utc():
    d1 = _session_m15("2024-01-02", [(100, 101, 99, 100)] * 6)
    d2 = _session_m15("2024-01-03", [(100, 101, 99, 100)] * 6)
    out = CanalAbertura().process(pd.concat([d1, d2]))
    assert out.session_id.nunique() == 2
    # ultima barra de cada sessao marcada
    assert int(out.is_session_last.sum()) == 2


def test_sem_look_ahead_a_caixa_nao_muda_com_dados_futuros():
    """A caixa/sinais da sessao do dia D nao podem mudar quando velas de D+1 sao
    adicionadas: cada sessao e auto-contida (causal por construcao)."""
    d1 = _session_m15(
        "2024-01-02",
        [
            (101, 103, 100, 102),
            (102, 105, 101, 104),
            (104, 104, 102, 103),
            (103, 104, 102, 103),
            (103, 110, 103, 109),
            (109, 111, 108, 110),
        ],
    )
    d2 = _session_m15("2024-01-03", [(200, 201, 199, 200)] * 6)
    eng = CanalAbertura()
    only_d1 = eng.process(d1.copy())
    with_d2 = eng.process(pd.concat([d1, d2]).copy()).loc[d1.index]
    for col in ("ca_top", "ca_bottom", "signal", "stop_loss", "take_profit_1"):
        a = only_d1[col].to_numpy(float)
        b = with_d2[col].to_numpy(float)
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), col


def test_mark_on_invalido_levanta():
    with pytest.raises(ValueError):
        CanalAberturaParams(mark_on="nope")
    with pytest.raises(ValueError):
        CanalAberturaParams(opening_bars=0)
