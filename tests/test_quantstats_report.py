"""Testes do tearsheet QuantStats — conversao NAV->retornos e tabela de metricas.

Nao dependem do SQLite nem de rede: montam uma NavSeries sintetica. Provam que a
curva de equity vira retornos diarios corretos e que a tabela de metricas sai
populada (Sharpe & cia). A geracao de HTML em si (matplotlib) nao e exercitada
aqui para manter o teste rapido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reporting.quantstats_report import metrics_table, nav_to_returns
from reporting.track_record import NavSeries


def _serie_sintetica(n: int = 120, drift: float = 0.0008, seed: int = 7) -> NavSeries:
    rng = np.random.default_rng(seed)
    rets = drift + 0.01 * rng.standard_normal(n)
    equity = 100_000 * np.cumprod(1 + rets)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-01", periods=n)]
    bench = 100_000 * np.cumprod(1 + (0.0005 + 0.009 * rng.standard_normal(n)))
    return NavSeries(dates=dates, equity=equity, bench_spy=bench, bench_6040=bench)


def test_nav_to_returns_alinha_serie():
    s = _serie_sintetica(n=10)
    r = nav_to_returns(s)
    assert isinstance(r, pd.Series)
    assert len(r) == 9  # n-1 retornos
    assert isinstance(r.index, pd.DatetimeIndex)
    # primeiro retorno = equity[1]/equity[0]-1
    esperado = s.equity[1] / s.equity[0] - 1
    assert abs(r.iloc[0] - esperado) < 1e-12


def test_nav_to_returns_ignora_equity_invalida():
    s = NavSeries(
        dates=["2025-01-01", "2025-01-02", "2025-01-03"],
        equity=np.array([0.0, 100.0, 110.0]),
        bench_spy=np.array([np.nan, np.nan, np.nan]),
        bench_6040=np.array([np.nan, np.nan, np.nan]),
    )
    r = nav_to_returns(s)
    assert len(r) == 1  # so o retorno 100->110 sobrevive
    assert abs(r.iloc[0] - 0.1) < 1e-12


def test_metrics_table_popula():
    s = _serie_sintetica()
    r = nav_to_returns(s)
    tabela = metrics_table(r)
    assert tabela is not None and len(tabela) > 0
    texto = " ".join(str(i) for i in tabela.index)
    assert "Sharpe" in texto
