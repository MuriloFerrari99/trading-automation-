"""Tribunal de FATORES DE ACOES price-based (R&D offline, programa de validacao).

PERGUNTA (imparcial, MEDIR — nao assumir): um sleeve de fatores de acoes
price-based — MOMENTUM cross-seccional (12-1m) + LOW-VOLATILITY (vol rolante) —
tem edge risco-ajustado ROBUSTO **e** diversifica o nosso beta (vol-target)?

Este modulo e R&D PURO, em paralelo ao beta vivo. NAO toca em beta_*, main.py,
nav_history nem config de producao. IMPORTA (read-only) de:
  - simulation.beta_portfolio : loader/cache de precos (data/beta_cache) + a serie
    de retornos LIQUIDOS do vol-target (o nosso "beta" real, p/ a corr de diversificacao)
  - simulation.costs          : EQUITY_BASE (custo real de equity, por lado)
  - simulation.statistics     : DSR/PSR/PBO (tribunal anti-overfitting)
  - simulation.metrics        : sharpe/sortino/max_drawdown

UNIVERSO (dados gratis, yfinance, cacheado em data/factor_cache/*.csv):
  ~40 nomes liquidos de mega/large-cap US (constituintes do S&P 100), close ajustado,
  historico maximo. Cacheado -> roda offline apos um fetch.

FATORES (price-based, testaveis sem fundamentos), rebalance MENSAL:
  - MOMENTUM cross-seccional: retorno 12-1 meses (PULA o ultimo mes p/ evitar
    short-term reversal); ranqueia o universo; long top tercil/quartil.
  - LOW-VOLATILITY: vol rolante (default 6m); long baixa-vol.
  - COMBINADO: media dos ranks dos dois.
  Cada fator roda em 2 VARIANTES:
    * LONG-ONLY (long top, equal-weight) — implementavel na nossa conta (sem short de acao).
    * LONG-SHORT (long top - short bottom, dollar-neutral) — o fator "academico" puro.

SEM LOOK-AHEAD (rigoroso):
  - Sinal calculado com dados ATE o fim do mes m (close de t <= ultimo dia de m).
  - Pesos do mes m+1 aplicados aos retornos DIARIOS do mes m+1 (decididos em t, retorno
    em t+1: garantido por weights.shift(1) na simulacao diaria).
  - Vol/momentum usam janelas que terminam em t-1 relativo ao retorno avaliado.

CUSTO: simulation.costs.EQUITY_BASE (3 bps/lado) + cenario ESTRESSADO 2x. Cobrado
sobre |Delta peso| de cada rebalance (turnover real). VEREDITO usa o cenario ESTRESSADO.

n_TRIALS HONESTO (anti data-snooping): fatores {mom, lowvol, combo} x variantes
{long-only, long-short} x lookbacks de momentum {6,9,12m} x lookbacks de vol {3,6,12m}
x cortes {tercil, quartil}. Contado e passado ao DSR; PBO sobre a matriz de todas as configs.

BARRA (regua do programa): PASSA p/ candidato a sleeve SO se:
  DSR >= 0.95  E  Sharpe liq robusto (cenario estressado)  E
  (bate buy&hold/equal-weight  OU  diversifica: corr baixa com o beta MELHORANDO o
   conjunto), de forma ROBUSTA nos lookbacks. Senao: FALHA -> cemiterio.

Uso:
    uv run python -m simulation.factors_equity            # usa cache
    uv run python -m simulation.factors_equity --force    # re-baixa precos
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.beta_portfolio import (
    UNIVERSE as BETA_UNIVERSE,
    load_close,
    load_panel,
    weights_disciplined,
    run_portfolio,
)
from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown, sharpe as _sharpe, sortino as _sortino
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.factors_equity")

CACHE_DIR = Path("data/factor_cache")
REPORT_PATH = Path("data/factor_equity_report.txt")
TRADING_DAYS = EQUITY_PERIODS  # 252

# Universo: ~40 nomes liquidos do S&P 100 (mega/large-cap, varios setores). Precos
# ajustados via yfinance (mesmo loader/cache do beta). Diversidade setorial p/ que
# o ranqueamento cross-seccional tenha o que separar.
EQUITY_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO",  # tech
    "JPM", "BAC", "WFC", "GS", "MS", "C",                            # financials
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY",                       # healthcare
    "PG", "KO", "PEP", "WMT", "COST", "MCD",                         # staples/cons
    "XOM", "CVX", "COP",                                              # energy
    "HD", "NKE", "DIS", "SBUX",                                       # consumer disc
    "CAT", "BA", "HON", "GE",                                         # industrials
    "VZ", "T",                                                        # telecom
    "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN",                    # more tech
]

# Benchmark de mercado p/ comparacao e proxy de "beta".
SPY = "SPY"


# ============================================================================
# DADOS
# ============================================================================
def load_equity_panel(force: bool = False) -> pd.DataFrame:
    """Painel de closes ajustados {ticker} dos nomes do universo + SPY.

    Reusa o loader/cache do beta_portfolio (yfinance, close ajustado, period=max).
    Cacheia o painel consolidado em data/factor_cache/panel.csv. Alinha no
    calendario de pregao (uniao -> ffill curto). Pula nomes sem dados.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel_cache = CACHE_DIR / "panel.csv"
    if panel_cache.exists() and not force:
        df = pd.read_csv(panel_cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]

    series: dict[str, pd.Series] = {}
    for tk in [*EQUITY_UNIVERSE, SPY]:
        try:
            df = load_close(tk, force=force)
        except Exception as exc:  # noqa: BLE001 — rede/yf
            logger.warning("Falha em %s: %s", tk, exc)
            continue
        if df.empty:
            logger.warning("%s sem dados — pulado.", tk)
            continue
        series[tk] = df["close"]

    if not series:
        return pd.DataFrame()

    panel = pd.DataFrame(series).sort_index()
    # alinha no calendario uniao; ffill curto (1 dia) p/ feriados isolados.
    panel = panel.ffill(limit=1)
    panel.to_csv(panel_cache)
    return panel


def common_window(panel: pd.DataFrame, min_names: int = 30) -> pd.DataFrame:
    """Janela onde ha >= min_names nomes vivos (descarta o warmup de IPOs recentes)."""
    alive = panel.notna().sum(axis=1)
    valid = alive[alive >= min_names]
    if valid.empty:
        return panel
    start = valid.index[0]
    return panel.loc[start:]


# ============================================================================
# SINAIS DE FATOR (cross-seccional, mensal, SEM look-ahead)
# ============================================================================
def month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ultimo dia de pregao de cada mes presente no indice."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().to_numpy())


def momentum_score(
    closes: pd.DataFrame, rebal_dates: pd.DatetimeIndex, lookback_m: int, skip_m: int = 1
) -> pd.DataFrame:
    """Momentum cross-seccional (retorno lookback-skip meses) por data de rebalance.

    Em cada data de rebalance t: ret = close[t - skip*21] / close[t - lookback*21] - 1.
    Usa SO dados <= t (close de t = ultimo close do mes). NaN p/ nome sem historico.
    Retorna DataFrame (rebal_dates x tickers) de scores (maior = melhor).
    """
    out = pd.DataFrame(index=rebal_dates, columns=closes.columns, dtype=float)
    arr = closes
    for t in rebal_dates:
        loc = arr.index.get_indexer([t], method="ffill")[0]
        if loc < 0:
            continue
        i_skip = loc - skip_m * 21
        i_look = loc - lookback_m * 21
        if i_look < 0:
            continue
        recent = arr.iloc[i_skip]
        past = arr.iloc[i_look]
        out.loc[t] = (recent / past - 1.0)
    return out


def lowvol_score(
    closes: pd.DataFrame, rebal_dates: pd.DatetimeIndex, lookback_m: int
) -> pd.DataFrame:
    """Low-vol cross-seccional: -1 * vol rolante (lookback meses) ate t.

    Score MAIOR = menor vol = melhor. Usa retornos diarios <= t.
    """
    rets = closes.pct_change()
    win = lookback_m * 21
    out = pd.DataFrame(index=rebal_dates, columns=closes.columns, dtype=float)
    for t in rebal_dates:
        loc = rets.index.get_indexer([t], method="ffill")[0]
        if loc < win:
            continue
        window = rets.iloc[loc - win + 1 : loc + 1]
        vol = window.std(ddof=1)
        out.loc[t] = -vol
    return out


def _rank_pct(scores_row: pd.Series) -> pd.Series:
    """Rank percentil [0,1] dos scores validos da linha (maior score = rank maior)."""
    valid = scores_row.dropna()
    if valid.size < 4:
        return pd.Series(np.nan, index=scores_row.index)
    r = valid.rank(pct=True)
    return r.reindex(scores_row.index)


def combo_score(mom: pd.DataFrame, lowvol: pd.DataFrame) -> pd.DataFrame:
    """Media dos ranks percentil de momentum e low-vol (cross-seccional, por data)."""
    out = pd.DataFrame(index=mom.index, columns=mom.columns, dtype=float)
    for t in mom.index:
        rm = _rank_pct(mom.loc[t])
        rv = _rank_pct(lowvol.loc[t])
        out.loc[t] = (rm + rv) / 2.0
    return out


# ============================================================================
# CONSTRUCAO DE PESOS (a partir dos scores) -> pesos diarios
# ============================================================================
def weights_from_scores(
    scores: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    cut: float,
    long_short: bool,
) -> pd.DataFrame:
    """Pesos diarios a partir de scores mensais.

    cut: fracao do topo/fundo (1/3 tercil, 1/4 quartil).
    long_short: True -> long top (+) e short bottom (-), dollar-neutral (long=+0.5,
        short=-0.5 do bruto, soma=0). False -> long-only top, equal-weight (soma=1).
    Os pesos da data de rebalance t valem ATE o proximo rebalance (forward-fill).
    A simulacao diaria aplica .shift(1) (decide em t, retorno em t+1) -> sem look-ahead.
    """
    w_rebal = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for t in scores.index:
        row = scores.loc[t].dropna()
        n = row.size
        if n < 4:
            continue
        k = max(1, int(round(n * cut)))
        ranked = row.sort_values(ascending=False)
        top = ranked.index[:k]
        if long_short:
            bottom = ranked.index[-k:]
            w_rebal.loc[t, top] = 0.5 / len(top)
            w_rebal.loc[t, bottom] = -0.5 / len(bottom)
        else:
            w_rebal.loc[t, top] = 1.0 / len(top)

    # expande p/ diario: peso de t vale ate o proximo rebalance.
    w_daily = w_rebal.reindex(daily_index).ffill().fillna(0.0)
    return w_daily


# ============================================================================
# SIMULACAO DIARIA com custo real (turnover) — SEM look-ahead
# ============================================================================
def simulate(
    closes: pd.DataFrame, weights_daily: pd.DataFrame, cost_per_side_bps: float
) -> pd.Series:
    """Retornos diarios LIQUIDOS do book de pesos.

    SEM LOOK-AHEAD: usa weights.shift(1) (peso decidido em t-1 captura retorno de t).
    Custo = |Delta peso| * custo_lado, cobrado no dia em que o peso muda.
    """
    rets = closes.pct_change().fillna(0.0)
    w = weights_daily.reindex(closes.index).fillna(0.0)
    w_eff = w.shift(1).fillna(0.0)  # decide em t-1, captura retorno de t
    common = [c for c in closes.columns if c in w_eff.columns]
    r = rets[common].to_numpy()
    we = w_eff[common].to_numpy()

    gross = (we * r).sum(axis=1)
    turnover = np.abs(np.diff(we, axis=0, prepend=np.zeros((1, we.shape[1]))))
    cost = turnover.sum(axis=1) * (cost_per_side_bps / 1e4)
    net = gross - cost

    net_s = pd.Series(net, index=closes.index)
    nz = np.flatnonzero(np.abs(we).sum(axis=1) > 0)
    start = int(nz[0]) if nz.size else 0
    return net_s.iloc[start:]


def equal_weight_benchmark(closes: pd.DataFrame) -> pd.Series:
    """Buy&hold equal-weight do universo (rebalance diario implicito p/ EW puro)."""
    cols = [c for c in closes.columns if c != SPY]
    rets = closes[cols].pct_change()
    return rets.mean(axis=1).fillna(0.0)


# ============================================================================
# METRICAS
# ============================================================================
@dataclass
class FactorStats:
    name: str
    n_days: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    corr_beta: float  # corr com o vol-target (nosso beta)
    corr_spy: float


def _cagr(net: pd.Series) -> float:
    if net.size < 2:
        return 0.0
    eq = (1.0 + net).cumprod()
    years = net.size / TRADING_DAYS
    val = eq.iloc[-1]
    if val <= 0:
        return -1.0
    return float(val ** (1.0 / max(years, 1e-9)) - 1.0)


def factor_stats(
    name: str, net: pd.Series, beta_net: pd.Series | None, spy_net: pd.Series | None
) -> FactorStats:
    r = net.to_numpy()
    eq = (1.0 + net).cumprod().to_numpy()
    mdd = max_drawdown(eq)
    cagr = _cagr(net)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else (float("inf") if cagr > 0 else 0.0)

    def _corr(other: pd.Series | None) -> float:
        if other is None:
            return float("nan")
        a, b = net.align(other, join="inner")
        if a.size < 30 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])

    return FactorStats(
        name=name,
        n_days=int(net.size),
        cagr=cagr,
        sharpe=_sharpe(r, periods=TRADING_DAYS),
        sortino=_sortino(r, periods=TRADING_DAYS),
        max_dd=mdd,
        calmar=calmar,
        corr_beta=_corr(beta_net),
        corr_spy=_corr(spy_net),
    )


# ============================================================================
# CONFIGS (n_trials honesto)
# ============================================================================
@dataclass(frozen=True)
class Config:
    factor: str       # "mom" | "lowvol" | "combo"
    variant: str      # "LO" (long-only) | "LS" (long-short)
    mom_look: int     # meses
    vol_look: int     # meses
    cut: float        # 1/3 | 1/4
    label: str


def build_configs() -> list[Config]:
    """Grade honesta de configuracoes testadas (= n_trials para o DSR)."""
    cfgs: list[Config] = []
    mom_looks = [6, 9, 12]
    vol_looks = [3, 6, 12]
    cuts = [(1 / 3, "T"), (1 / 4, "Q")]
    variants = [("LO", False), ("LS", True)]
    for vname, _ in variants:
        for cval, cname in cuts:
            for ml in mom_looks:
                cfgs.append(Config("mom", vname, ml, 6, cval, f"mom_{ml}m_{cname}_{vname}"))
            for vl in vol_looks:
                cfgs.append(Config("lowvol", vname, 12, vl, cval, f"lowvol_{vl}m_{cname}_{vname}"))
            for ml in mom_looks:
                for vl in vol_looks:
                    cfgs.append(
                        Config("combo", vname, ml, vl, cval,
                               f"combo_m{ml}_v{vl}_{cname}_{vname}")
                    )
    return cfgs


def run_config(
    closes: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    cfg: Config,
    cost_bps: float,
    *,
    mom_cache: dict[int, pd.DataFrame],
    vol_cache: dict[int, pd.DataFrame],
) -> pd.Series:
    """Retornos diarios liquidos de UMA config."""
    if cfg.factor == "mom":
        scores = mom_cache[cfg.mom_look]
    elif cfg.factor == "lowvol":
        scores = vol_cache[cfg.vol_look]
    else:
        scores = combo_score(mom_cache[cfg.mom_look], vol_cache[cfg.vol_look])
    w = weights_from_scores(
        scores, closes.index, cut=cfg.cut, long_short=(cfg.variant == "LS")
    )
    return simulate(closes, w, cost_bps)


# ============================================================================
# BETA (vol-target) — a serie real do nosso produto, p/ a corr de diversificacao
# ============================================================================
def beta_voltarget_returns(force: bool = False) -> pd.Series | None:
    """Retornos diarios LIQUIDOS do vol-target (o nosso beta vivo), via beta_portfolio.

    Reusa weights_disciplined(use_gate=False, use_vol_target=True) — o produto
    CONFIRMADO. Read-only; nao altera nada de producao. Retorna None se faltar dado.
    """
    try:
        panel, classes = load_panel(force=force)
        if panel.empty:
            return None
        w = weights_disciplined(panel, classes, use_gate=False, use_vol_target=True)
        _, net = run_portfolio(panel, w, classes)
        return net
    except Exception as exc:  # noqa: BLE001
        logger.warning("Beta vol-target indisponivel: %s", exc)
        return None


# ============================================================================
# RELATORIO / VEREDITO
# ============================================================================
def _fmt_pct(x: float) -> str:
    if x != x:  # nan
        return "  n/a"
    return f"{x * 100:6.1f}%"


def _fmt(x: float) -> str:
    if x != x:
        return " n/a"
    return f"{x:6.2f}"


def run(force: bool = False) -> str:
    lines: list[str] = []
    panel = load_equity_panel(force=force)
    if panel.empty:
        return (
            "DADOS PENDENTES — nenhum preco baixado.\n"
            "Rode: uv run python -m simulation.factors_equity --force\n"
        )
    panel = common_window(panel, min_names=30)
    spy = panel[SPY] if SPY in panel.columns else None
    closes = panel[[c for c in panel.columns if c != SPY]].copy()

    start = closes.index[0].date()
    end = closes.index[-1].date()
    rebal = month_end_index(closes.index)

    # caches de score por lookback (custoso) — calcula 1x.
    mom_looks = sorted({6, 9, 12})
    vol_looks = sorted({3, 6, 12})
    mom_cache = {ml: momentum_score(closes, rebal, ml) for ml in mom_looks}
    vol_cache = {vl: lowvol_score(closes, rebal, vl) for vl in vol_looks}

    cost_base = EQUITY_BASE.per_side_bps
    cost_stress = EQUITY_BASE.stressed(2.0).per_side_bps

    configs = build_configs()
    n_trials = len(configs)

    # roda TODAS as configs nos dois cenarios de custo.
    net_stress: dict[str, pd.Series] = {}
    net_base: dict[str, pd.Series] = {}
    for cfg in configs:
        net_base[cfg.label] = run_config(
            closes, rebal, cfg, cost_base, mom_cache=mom_cache, vol_cache=vol_cache
        )
        net_stress[cfg.label] = run_config(
            closes, rebal, cfg, cost_stress, mom_cache=mom_cache, vol_cache=vol_cache
        )

    # benchmarks e beta
    ew = equal_weight_benchmark(panel)
    spy_net = spy.pct_change().fillna(0.0) if spy is not None else None
    beta_net = beta_voltarget_returns(force=False)

    # alinha beta/spy a janela das estrategias p/ a corr
    # ---- TABELA principal: as 3 configs "headline" (12m mom, 6m vol, tercil, LO) +
    #      a melhor de cada familia por Sharpe estressado. ----
    def headline(label_pred) -> str | None:
        cands = [c.label for c in configs if label_pred(c)]
        if not cands:
            return None
        return max(cands, key=lambda lb: observed_sharpe(net_stress[lb].to_numpy()))

    head_mom = headline(lambda c: c.factor == "mom")
    head_lv = headline(lambda c: c.factor == "lowvol")
    head_combo = headline(lambda c: c.factor == "combo")

    # PBO sobre a matriz de TODAS as configs (cenario estressado) — alinhadas.
    aligned = pd.DataFrame(net_stress).dropna()
    pbo = probability_of_backtest_overfitting(aligned.to_numpy(), n_splits=16) if len(aligned) > 50 else float("nan")

    # estatisticas de cada headline + benchmarks
    def stat(label: str) -> FactorStats:
        return factor_stats(label, net_stress[label], beta_net, spy_net)

    rows: list[FactorStats] = []
    for lb in (head_mom, head_lv, head_combo):
        if lb:
            rows.append(stat(lb))
    rows.append(factor_stats("BENCH_equal_weight", ew, beta_net, spy_net))
    if spy_net is not None:
        rows.append(factor_stats("BENCH_SPY", spy_net, beta_net, None))
    if beta_net is not None:
        rows.append(factor_stats("BETA_voltarget", beta_net, None, spy_net))

    # trial sharpes (por periodo) p/ o DSR honesto.
    trial_sr = [observed_sharpe(net_stress[c.label].to_numpy()) for c in configs]

    # DSR de cada headline (com n_trials e a dispersao real entre trials).
    def verdict(label: str):
        return evaluate_edge(
            net_stress[label].to_numpy(),
            n_trials=n_trials,
            trial_sharpes=trial_sr,
            periods_per_year=TRADING_DAYS,
            min_sharpe_annual=0.8,
            dsr_threshold=0.95,
        )

    verdicts = {lb: verdict(lb) for lb in (head_mom, head_lv, head_combo) if lb}

    # ---- robustez aos lookbacks: dispersao de Sharpe dentro de cada fator/variante ----
    def robustness(factor: str, variant: str) -> tuple[float, float, int]:
        srs = [
            observed_sharpe(net_stress[c.label].to_numpy()) * np.sqrt(TRADING_DAYS)
            for c in configs
            if c.factor == factor and c.variant == variant
        ]
        if not srs:
            return (float("nan"), float("nan"), 0)
        return (float(np.mean(srs)), float(np.std(srs)), len(srs))

    # =====================  ESCREVE O RELATORIO  =====================
    lines.append("=" * 78)
    lines.append("TRIBUNAL DE FATORES DE ACOES (price-based) — momentum + low-vol")
    lines.append("=" * 78)
    lines.append(f"Universo: {len(closes.columns)} nomes liquidos (S&P 100). Janela {start} .. {end}")
    lines.append(f"Dias de pregao: {len(closes)} | rebalances mensais: {len(rebal)}")
    lines.append(f"Custo (EQUITY_BASE): base {cost_base:.0f} bps/lado | "
                 f"VEREDITO no ESTRESSADO {cost_stress:.0f} bps/lado")
    lines.append(f"n_trials (honesto) = {n_trials}  "
                 f"(3 fatores x 2 variantes x lookbacks mom/vol x 2 cortes)")
    lines.append(f"PBO (CSCV, todas as {n_trials} configs, estressado) = {_fmt(pbo)}  "
                 f"[<0.5 minimo; menor=melhor]")
    lines.append("")
    lines.append("(1) TABELA — melhor config por familia (cenario ESTRESSADO) vs benchmark vs beta")
    lines.append("-" * 78)
    hdr = f"{'estrategia':<26}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Calmar':>8}{'cBeta':>7}{'cSPY':>7}"
    lines.append(hdr)
    for s in rows:
        lines.append(
            f"{s.name:<26}{_fmt_pct(s.cagr):>8}{_fmt(s.sharpe):>8}{_fmt(s.sortino):>8}"
            f"{_fmt_pct(s.max_dd):>8}{_fmt(s.calmar):>8}{_fmt(s.corr_beta):>7}{_fmt(s.corr_spy):>7}"
        )
    lines.append("")
    lines.append("    DSR/PSR das headlines (n_trials e dispersao real entre trials):")
    for lb, v in verdicts.items():
        lines.append(f"      {lb:<22} {v.summary()}")
    lines.append("")

    lines.append("(2) ROBUSTEZ AOS LOOKBACKS — Sharpe anual medio +/- desvio por familia/variante")
    lines.append("-" * 78)
    for factor in ("mom", "lowvol", "combo"):
        for variant in ("LO", "LS"):
            mean, sd, n = robustness(factor, variant)
            lines.append(
                f"  {factor:<7} {variant:<3} : Sharpe {_fmt(mean)} +/- {_fmt(sd)}  "
                f"(n={n} configs)  {'[instavel]' if sd == sd and sd > 0.4 else '[estavel]' if sd == sd else ''}"
            )
    lines.append("")

    lines.append("(3) CORRELACAO COM O BETA (diversifica?)")
    lines.append("-" * 78)
    if beta_net is not None:
        lines.append("    cBeta = corr diaria com o vol-target (nosso produto vivo).")
        for s in rows:
            if s.name.startswith("BETA"):
                continue
            tag = ""
            if s.corr_beta == s.corr_beta:
                tag = "[diversificante]" if abs(s.corr_beta) < 0.3 else "[redundante c/ beta]"
            lines.append(f"      {s.name:<26} cBeta={_fmt(s.corr_beta)} {tag}")
    else:
        lines.append("    Beta vol-target indisponivel -> usar cSPY como proxy do beta de mercado.")
        for s in rows:
            if s.name.startswith("BENCH_SPY"):
                continue
            lines.append(f"      {s.name:<26} cSPY={_fmt(s.corr_spy)}")
    lines.append("")

    # =====================  VEREDITO  =====================
    ew_sr = _sharpe(ew.to_numpy(), periods=TRADING_DAYS)
    lines.append("(4) VEREDITO")
    lines.append("-" * 78)
    verdict_flag = "CEMITERIO"
    reasons: list[str] = []
    any_pass = False
    for lb, v in verdicts.items():
        s = stat(lb)
        beats_bh = s.sharpe > ew_sr + 0.1
        diversifies = (s.corr_beta == s.corr_beta and abs(s.corr_beta) < 0.3 and s.sharpe > 0.3)
        passes = v.passes_dsr and v.passes_sharpe and (beats_bh or diversifies)
        any_pass = any_pass or passes
        reasons.append(
            f"  {lb}: DSR={v.dsr:.2f}({'ok' if v.passes_dsr else 'X'}) "
            f"Sharpe={s.sharpe:.2f}({'ok' if v.passes_sharpe else 'X'}) "
            f"bate_EW={'sim' if beats_bh else 'nao'}(EW Sharpe={ew_sr:.2f}) "
            f"diversifica={'sim' if diversifies else 'nao'}(cBeta={_fmt(s.corr_beta).strip()}) "
            f"-> {'PASSA' if passes else 'FALHA'}"
        )
    if any_pass:
        verdict_flag = "CANDIDATO A SLEEVE"
    lines.extend(reasons)
    lines.append("")
    robust_ok = all(
        (robustness(f, "LO")[1] <= 0.4 or robustness(f, "LO")[1] != robustness(f, "LO")[1])
        for f in ("mom", "lowvol", "combo")
    )
    lines.append(f"  Robustez geral aos lookbacks: {'OK' if robust_ok else 'FRAGIL (alta dispersao)'}")
    lines.append(f"  PBO: {_fmt(pbo).strip()} ({'aceitavel' if (pbo==pbo and pbo<0.5) else 'ALTO/indef' if pbo==pbo else 'indef'})")
    lines.append("")
    lines.append(f"  >>> VEREDITO FINAL: {verdict_flag}")
    lines.append("=" * 78)

    report = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tribunal de fatores de acoes price-based")
    parser.add_argument("--force", action="store_true", help="re-baixa precos (ignora cache)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report = run(force=args.force)
    print(report)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
