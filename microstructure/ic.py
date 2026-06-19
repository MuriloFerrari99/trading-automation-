"""Information Coefficient (IC) de sinais de order-flow vs retorno forward.

O coracao da Fase 1. Para cada sinal s e horizonte h, mede a correlacao entre o
sinal em t e o retorno forward t->t+h, SEM look-ahead (a grade ja garante isso).

Metricas:
  - IC Spearman (rank) e Pearson, POR DIA. Spearman e o headline (robusto a
    nao-linearidade e caudas; padrao em microestrutura).
  - Decaimento: IC em funcao do horizonte (microestrutura some rapido).
  - Estabilidade entre dias: IC medio +- desvio, t-stat (mean/se), e fracao de
    dias com o MESMO sinal do IC medio ("hit-rate" de consistencia). Ruido => IC
    medio ~0, dispersao alta, hit-rate ~50%.

Honestidade (impressa no report): IC alto NAO e lucro. O sinal ainda tem que
sobreviver ao FILL (adverse selection na latencia de varejo: quando o sinal e
forte, o preco ja andou e voce e o ultimo a entrar) e ao CUSTO (fee + spread) na
Fase 2. IC e condicao NECESSARIA, nao suficiente.

Sem scipy: Spearman/Pearson implementados em numpy (ties por average-rank).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microstructure.data import TradeArrays, load_all_days
from microstructure.signals import (
    DEFAULT_HORIZONS_S,
    DEFAULT_WINDOWS_S,
    TimeGrid,
    build_trade_signals,
    forward_log_return,
)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks com empates por media (equivalente a scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks 1-based
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


@dataclass
class DayIC:
    """IC de um (sinal, horizonte) em UM dia."""

    spearman: float
    pearson: float
    n: int


@dataclass
class SignalHorizonIC:
    """IC agregado de um (sinal, horizonte) entre dias."""

    signal: str
    horizon_s: int
    n_days: int
    n_obs_total: int
    ic_spearman_mean: float
    ic_spearman_std: float
    ic_spearman_tstat: float  # mean / (std/sqrt(n_days))
    ic_pearson_mean: float
    hit_rate: float  # fracao de dias com sign(IC_dia) == sign(IC_medio)
    per_day_spearman: list[float]


def _clean_pair(sig: np.ndarray, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mantem so posicoes com sinal e retorno finitos."""
    good = np.isfinite(sig) & np.isfinite(ret)
    return sig[good], ret[good]


def compute_day_ic(grid: TimeGrid, horizons_s: tuple[int, ...]) -> dict[tuple[str, int], DayIC]:
    """IC por (sinal, horizonte) para a grade de UM dia."""
    out: dict[tuple[str, int], DayIC] = {}
    fwd_cache: dict[int, np.ndarray] = {
        h: forward_log_return(grid.price, grid.step_s, h) for h in horizons_s
    }
    for name, sig in grid.signals.items():
        for h in horizons_s:
            s, r = _clean_pair(sig, fwd_cache[h])
            # descarta sinais degenerados (constantes) — IC indefinido
            if s.size < 30 or np.all(s == s[0]):
                out[(name, h)] = DayIC(float("nan"), float("nan"), int(s.size))
                continue
            out[(name, h)] = DayIC(_spearman(s, r), _pearson(s, r), int(s.size))
    return out


def aggregate_ic(
    day_ics: list[dict[tuple[str, int], DayIC]],
) -> dict[tuple[str, int], SignalHorizonIC]:
    """Agrega ICs diarios em estatistica entre dias por (sinal, horizonte)."""
    if not day_ics:
        return {}
    keys = sorted({k for d in day_ics for k in d})
    out: dict[tuple[str, int], SignalHorizonIC] = {}
    for (name, h) in keys:
        sp = np.array(
            [d[(name, h)].spearman for d in day_ics if (name, h) in d], dtype=np.float64
        )
        pe = np.array(
            [d[(name, h)].pearson for d in day_ics if (name, h) in d], dtype=np.float64
        )
        nobs = int(sum(d[(name, h)].n for d in day_ics if (name, h) in d))
        sp_valid = sp[np.isfinite(sp)]
        pe_valid = pe[np.isfinite(pe)]
        nd = int(sp_valid.size)
        if nd == 0:
            out[(name, h)] = SignalHorizonIC(
                name, h, 0, nobs, float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), [],
            )
            continue
        mean = float(sp_valid.mean())
        std = float(sp_valid.std(ddof=1)) if nd > 1 else 0.0
        se = std / np.sqrt(nd) if (nd > 1 and std > 0) else float("nan")
        tstat = float(mean / se) if (se and np.isfinite(se)) else float("nan")
        sign = np.sign(mean) if mean != 0 else 1.0
        hit = float(np.mean(np.sign(sp_valid) == sign)) if nd > 0 else float("nan")
        out[(name, h)] = SignalHorizonIC(
            signal=name, horizon_s=h, n_days=nd, n_obs_total=nobs,
            ic_spearman_mean=mean, ic_spearman_std=std, ic_spearman_tstat=tstat,
            ic_pearson_mean=float(pe_valid.mean()) if pe_valid.size else float("nan"),
            hit_rate=hit, per_day_spearman=[float(x) for x in sp_valid],
        )
    return out


@dataclass
class ICStudy:
    """Resultado completo do estudo de IC para um simbolo."""

    symbol: str
    market: str
    step_s: int
    windows_s: tuple[int, ...]
    horizons_s: tuple[int, ...]
    days: list[str]
    table: dict[tuple[str, int], SignalHorizonIC]


def run_ic_study(
    symbol: str,
    *,
    market: str = "um",
    step_s: int = 1,
    windows_s: tuple[int, ...] = DEFAULT_WINDOWS_S,
    horizons_s: tuple[int, ...] = DEFAULT_HORIZONS_S,
    trades_by_day: list[TradeArrays] | None = None,
) -> ICStudy | None:
    """Roda o estudo de IC completo p/ um simbolo (todos os dias em cache).

    trades_by_day: injetavel p/ teste; senao carrega de data/microstructure_cache/.
    """
    days_data = trades_by_day if trades_by_day is not None else load_all_days(symbol, market=market)
    if not days_data:
        return None
    day_ics: list[dict[tuple[str, int], DayIC]] = []
    days: list[str] = []
    for ta in days_data:
        grid = build_trade_signals(ta, step_s=step_s, windows_s=windows_s)
        if grid.n < 30:
            continue
        day_ics.append(compute_day_ic(grid, horizons_s))
        days.append(ta.day)
    if not day_ics:
        return None
    return ICStudy(
        symbol=symbol, market=market, step_s=step_s, windows_s=windows_s,
        horizons_s=horizons_s, days=sorted(days), table=aggregate_ic(day_ics),
    )
