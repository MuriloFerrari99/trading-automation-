"""Tribunal R&D — Style-premia COMBINADO (value + momentum + low-vol + carry, L/S).

PERGUNTA (imparcial, MEDIR): um sleeve multi-estilo cross-seccional num universo
multi-asset de ETFs liquidos — combinando MOMENTUM, LOW-VOL, VALUE (reversao de
longo prazo) e CARRY (yield de distribuicao trailing) — tem Sharpe risco-ajustado
ROBUSTO, melhor que cada estilo isolado, e diversifica/bate o mercado (SPY)?

R&D PURO e ISOLADO. NAO toca beta_*, main.py, nav_history nem config de producao.
Cria arquivo NOVO. Importa (read-only):
  - simulation.statistics : evaluate_edge / DSR / PSR / PBO
  - simulation.metrics    : max_drawdown, sharpe
  - simulation.costs      : EQUITY_BASE (custo real de ETF, por lado) + estresse 2x

UNIVERSO (dados GRATIS via yfinance, cacheado em data/style_premia_cache/*.csv):
  22 ETFs multi-asset, close AJUSTADO + dividendos (p/ carry):
    equity regioes: SPY QQQ IWM EFA EEM VGK EWJ
    setores US:     XLE XLF XLK XLU XLP XLV
    bonds/credito:  TLT IEF LQD HYG TIP EMB
    real/commodity: GLD DBC VNQ

ESTILOS (todos price/dividend-based, sem fundamentos, rebalance MENSAL):
  - MOMENTUM : retorno 12-1m (pula ult. mes, anti short-term reversal). Alto = bom.
  - LOW-VOL  : -vol rolante (default 6m). Baixa vol = bom (sinal = -vol).
  - VALUE    : reversao de LONGO prazo = -(retorno 60-12m). Caro (subiu muito) = ruim.
               proxy de value academico p/ multi-asset (LT reversal, De Bondt-Thaler).
  - CARRY    : yield de distribuicao trailing 12m / preco. Yield alto = bom.
               carry real cross-asset (bond coupon, equity div, REIT yield, commodity ~0).
  - COMBO    : media dos z-scores cross-seccionais dos 4 estilos.

VARIANTES:
  - LONG-SHORT dollar-neutral (long top tercil - short bottom tercil) — premia academica.
  - LONG-ONLY (long top tercil, equal-weight) — implementavel sem short.

SEM LOOK-AHEAD (rigoroso):
  - Sinal calculado com close ATE o fim do mes m (t <= ult. dia de m).
  - Pesos do mes m aplicados aos retornos do mes m+1 via weights.shift(1) na sim diaria.
  - Janelas de mom/vol/value terminam em t-1 relativo ao retorno avaliado.

CUSTO: EQUITY_BASE (3 bps/lado) sobre |Delta peso| (turnover real) + cenario 2x.
VEREDITO usa o cenario ESTRESSADO (honesto).

n_TRIALS HONESTO: estilos isolados {mom,lowvol,value,carry} + combo, x variantes
{LS, LO}, x lookbacks de vol {3,6,12m}, x cortes {tercil,quartil}. Tudo contado no DSR;
PBO sobre a matriz de TODAS as configs.

BARRA: PASSA so se DSR>=0.95 E Sharpe_liq robusto E (bate SPY OU corr baixa ao SPY
melhorando o conjunto) E robusto OOS (PBO baixo + split temporal).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

CACHE_DIR = Path("data/style_premia_cache")
REPORT_PATH = Path("data/style_premia_combo_verdict.txt")

UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ",
    "XLE", "XLF", "XLK", "XLU", "XLP", "XLV",
    "TLT", "IEF", "LQD", "HYG", "TIP", "EMB",
    "GLD", "DBC", "VNQ",
]
SPY = "SPY"
MIN_NAMES = 12  # exige pelo menos 12 ETFs vivos p/ um cross-section honesto


# --------------------------------------------------------------------------- #
# DADOS — yfinance, cacheado por ticker. close ajustado (total return) + yield.
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.csv"


def _fetch_one(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf

    h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if h is None or len(h) == 0:
        return None
    # Adj Close = total return (reinveste dividendos) -> usado p/ retornos.
    # Close + Dividends -> carry (yield trailing). Mantem ambos.
    df = pd.DataFrame(
        {
            "adj_close": h["Adj Close"].astype(float),
            "close": h["Close"].astype(float),
            "dividends": h["Dividends"].astype(float),
        }
    )
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_panels(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (adj_close, close, dividends) alinhados num painel (datas x tickers)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adj, close, div = {}, {}, {}
    for tk in UNIVERSE:
        p = _cache_path(tk)
        if p.exists() and not force:
            d = pd.read_csv(p, index_col=0, parse_dates=True)
        else:
            d = _fetch_one(tk)
            if d is None:
                raise RuntimeError(f"sem dados yfinance p/ {tk}")
            d.to_csv(p)
            time.sleep(0.4)
        adj[tk] = d["adj_close"]
        close[tk] = d["close"]
        div[tk] = d["dividends"]
    adj_df = pd.DataFrame(adj).sort_index()
    close_df = pd.DataFrame(close).sort_index()
    div_df = pd.DataFrame(div).reindex(adj_df.index).fillna(0.0)
    return adj_df, close_df, div_df


def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).last().values


# --------------------------------------------------------------------------- #
# SINAIS — cada um retorna scores (month_end x ticker); maior = mais atraente.
# --------------------------------------------------------------------------- #
def _monthly_close(adj: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    return adj.reindex(me, method="ffill")


def momentum_score(adj: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    mc = _monthly_close(adj, me)
    # retorno 12-1m: preco[t-1] / preco[t-12] - 1  (pula ultimo mes)
    return mc.shift(1) / mc.shift(12) - 1.0


def lowvol_score(adj: pd.DataFrame, me: pd.DatetimeIndex, months: int) -> pd.DataFrame:
    daily = adj.pct_change()
    win = months * 21
    vol = daily.rolling(win, min_periods=int(win * 0.6)).std()
    vol_me = vol.reindex(me, method="ffill").shift(1)  # vol ate t-1
    return -vol_me  # baixa vol = score alto


def value_score(adj: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    """Value cross-asset = reversao de LONGO prazo: -(retorno 60-12m).
    Quem subiu muito em 5y (ex-ult.ano) esta 'caro' -> score baixo (De Bondt-Thaler)."""
    mc = _monthly_close(adj, me)
    lt = mc.shift(12) / mc.shift(60) - 1.0
    return -lt


def carry_score(
    close: pd.DataFrame, div: pd.DataFrame, me: pd.DatetimeIndex
) -> pd.DataFrame:
    """Carry = yield de distribuicao trailing 12m / preco (ate t-1).
    Bond coupon, equity/REIT div, commodity ~0. Sinal cross-asset real."""
    # soma de dividendos dos ultimos ~252 dias uteis, dividido pelo close.
    ttm_div = div.rolling(252, min_periods=60).sum()
    yield_d = ttm_div / close.replace(0.0, np.nan)
    return yield_d.reindex(me, method="ffill").shift(1)


def _zscore_rows(scores: pd.DataFrame) -> pd.DataFrame:
    mu = scores.mean(axis=1)
    sd = scores.std(axis=1, ddof=0).replace(0.0, np.nan)
    return scores.sub(mu, axis=0).div(sd, axis=0)


def combo_score(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Media dos z-scores cross-seccionais. Cada parte ja em score 'maior=melhor'."""
    zs = [_zscore_rows(p) for p in parts]
    # alinha indices/colunas
    base = zs[0]
    stacked = np.stack([z.reindex_like(base).values for z in zs], axis=0)
    valid_any = np.sum(~np.isnan(stacked), axis=0) > 0
    avg = np.full(stacked.shape[1:], np.nan)
    with np.errstate(invalid="ignore"):
        avg[valid_any] = np.nanmean(stacked[:, valid_any], axis=0)
    out = pd.DataFrame(avg, index=base.index, columns=base.columns)
    # exige pelo menos 2 estilos validos por celula
    valid = np.sum(~np.isnan(stacked), axis=0)
    out[valid < 2] = np.nan
    return out


# --------------------------------------------------------------------------- #
# PESOS — long top tercil (+ short bottom no LS). dollar-neutral no LS.
# --------------------------------------------------------------------------- #
def weights_from_scores(
    scores: pd.DataFrame, *, long_short: bool, frac: float
) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        r = row.dropna()
        n = len(r)
        if n < MIN_NAMES:
            continue
        k = max(1, int(round(n * frac)))
        ranked = r.sort_values(ascending=False)
        longs = ranked.index[:k]
        w.loc[dt, longs] = 1.0 / k
        if long_short:
            shorts = ranked.index[-k:]
            w.loc[dt, shorts] = -1.0 / k
    return w


# --------------------------------------------------------------------------- #
# SIMULACAO diaria sem look-ahead. weights mensais -> aplicados ao mes seguinte.
# --------------------------------------------------------------------------- #
def simulate(
    adj: pd.DataFrame, monthly_w: pd.DataFrame, cost_per_side_bps: float
) -> pd.Series:
    daily_ret = adj.pct_change().fillna(0.0)
    # expande pesos mensais para diario, mantendo o peso DECIDIDO no fim do mes
    # ate o proximo rebalance. shift(1) garante: peso decidido em t aplica em t+1.
    w_daily = monthly_w.reindex(daily_ret.index, method="ffill").shift(1)
    w_daily = w_daily.reindex(columns=daily_ret.columns).fillna(0.0)

    gross = (w_daily * daily_ret).sum(axis=1)
    # custo de turnover: |Delta peso| nos dias de rebalance (mudou o peso ffill)
    turnover = (w_daily - w_daily.shift(1)).abs().sum(axis=1)
    cost = turnover * (cost_per_side_bps / 1e4)
    net = gross - cost
    # so vale apos primeiro peso valido
    first = w_daily.abs().sum(axis=1)
    start = first[first > 0].index.min()
    if start is None or (isinstance(start, float) and np.isnan(start)):
        return pd.Series(dtype=float)
    return net.loc[start:]


def spy_returns(adj: pd.DataFrame, start, end) -> pd.Series:
    return adj[SPY].pct_change().loc[start:end].fillna(0.0)


# --------------------------------------------------------------------------- #
# METRICAS
# --------------------------------------------------------------------------- #
@dataclass
class StyleStats:
    name: str
    n_days: int
    sharpe_ann: float
    cagr: float
    maxdd: float
    corr_spy: float
    net: pd.Series


def _cagr(net: pd.Series) -> float:
    if len(net) < 2:
        return 0.0
    eq = (1.0 + net).cumprod()
    yrs = len(net) / EQUITY_PERIODS
    return float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0


def compute_stats(name: str, net: pd.Series, spy: pd.Series) -> StyleStats:
    arr = net.values
    sh = observed_sharpe(arr) * np.sqrt(EQUITY_PERIODS)
    eq = (1.0 + net).cumprod().values
    dd = max_drawdown(eq)
    al = net.align(spy, join="inner")
    corr = float(np.corrcoef(al[0].values, al[1].values)[0, 1]) if len(al[0]) > 5 else float("nan")
    return StyleStats(name, len(net), float(sh), _cagr(net), float(dd), corr, net)


# --------------------------------------------------------------------------- #
# CONFIGS / RUN
# --------------------------------------------------------------------------- #
STYLES = ["mom", "lowvol", "value", "carry", "combo"]
VARIANTS = [("LS", True), ("LO", False)]
VOL_MONTHS = [3, 6, 12]
FRACS = [("tercil", 1 / 3), ("quartil", 0.25)]


def build_signal(
    style: str, adj: pd.DataFrame, close: pd.DataFrame, div: pd.DataFrame,
    me: pd.DatetimeIndex, vol_m: int,
) -> pd.DataFrame:
    mom = momentum_score(adj, me)
    lv = lowvol_score(adj, me, vol_m)
    val = value_score(adj, me)
    car = carry_score(close, div, me)
    if style == "mom":
        return mom
    if style == "lowvol":
        return lv
    if style == "value":
        return val
    if style == "carry":
        return car
    if style == "combo":
        return combo_score([mom, lv, val, car])
    raise ValueError(style)


def n_trials_honest() -> int:
    # estilos x variantes x vol_lookbacks x cortes (vol_lookback so muda lowvol/combo,
    # mas contamos o grid completo de forma conservadora -> anti-snooping honesto)
    return len(STYLES) * len(VARIANTS) * len(VOL_MONTHS) * len(FRACS)


def run(force: bool = False) -> str:
    out: list[str] = []
    def P(s=""):
        out.append(s)

    P("=" * 78)
    P("TRIBUNAL — Style-premia COMBINADO (value+momentum+lowvol+carry, L/S)")
    P("=" * 78)

    adj, close, div = load_panels(force=force)
    # janela comum: exige MIN_NAMES vivos
    alive = adj.notna().sum(axis=1)
    adj = adj.loc[alive >= MIN_NAMES].dropna(how="all", axis=1)
    cols = adj.columns
    close = close[cols].reindex(adj.index)
    div = div[cols].reindex(adj.index).fillna(0.0)
    me = pd.DatetimeIndex(month_end_index(adj.index))

    P(f"Universo: {list(cols)}")
    P(f"Janela: {adj.index[0].date()} -> {adj.index[-1].date()} "
      f"({len(adj)} dias, {len(me)} meses)")
    cost_stress = EQUITY_BASE.stressed(2.0).per_side_bps
    P(f"Custo: base {EQUITY_BASE.per_side_bps:.0f} bps/lado | VEREDITO no estresse "
      f"{cost_stress:.0f} bps/lado")

    nt = n_trials_honest()
    P(f"n_trials honesto (DSR): {nt}")
    P("")

    # ----- roda TODAS as configs (matriz p/ PBO) sob custo estressado ----------
    all_nets: dict[str, pd.Series] = {}
    config_stats: dict[str, StyleStats] = {}
    spy_full = adj[SPY].pct_change().fillna(0.0)

    for style in STYLES:
        for vlabel, vm in [("v6", 6)] if style not in ("lowvol", "combo") else [(f"v{m}", m) for m in VOL_MONTHS]:
            sig = build_signal(style, adj, close, div, me, vm)
            for vname, ls in VARIANTS:
                for flabel, frac in FRACS:
                    w = weights_from_scores(sig, long_short=ls, frac=frac)
                    net = simulate(adj, w, cost_stress)
                    if len(net) < 252:
                        continue
                    key = f"{style}.{vname}.{flabel}.{vlabel}"
                    all_nets[key] = net
                    sp = spy_full.reindex(net.index).fillna(0.0)
                    config_stats[key] = compute_stats(key, net, sp)

    # ----- PBO sobre a matriz de todas as configs (janela comum) ---------------
    common = None
    for net in all_nets.values():
        common = net.index if common is None else common.intersection(net.index)
    mat = np.column_stack([all_nets[k].reindex(common).fillna(0.0).values for k in all_nets])
    pbo = probability_of_backtest_overfitting(mat, n_splits=16)

    # ----- estilos isolados (melhor variante de cada p/ comparar com combo) ----
    P("-" * 78)
    P("ESTILOS ISOLADOS vs COMBO (custo estressado, todas as variantes do grid):")
    P("-" * 78)
    P(f"{'config':<26}{'Sharpe':>8}{'CAGR':>9}{'MaxDD':>9}{'corrSPY':>9}")
    # ordena por sharpe
    for k in sorted(config_stats, key=lambda x: -config_stats[x].sharpe_ann):
        s = config_stats[k]
        P(f"{k:<26}{s.sharpe_ann:>8.2f}{s.cagr*100:>8.1f}%{s.maxdd*100:>8.1f}%{s.corr_spy:>9.2f}")

    # melhor de cada estilo (qualquer variante)
    best_by_style: dict[str, StyleStats] = {}
    for style in STYLES:
        cands = [config_stats[k] for k in config_stats if k.startswith(style + ".")]
        if cands:
            best_by_style[style] = max(cands, key=lambda s: s.sharpe_ann)

    P("")
    P("MELHOR variante por estilo:")
    for style in STYLES:
        if style in best_by_style:
            s = best_by_style[style]
            P(f"  {style:<8} -> {s.name:<24} Sharpe={s.sharpe_ann:.2f} "
              f"CAGR={s.cagr*100:.1f}% corrSPY={s.corr_spy:.2f}")

    # ----- candidato a graduacao: o COMBO (tese da estrategia) -----------------
    # escolhe o melhor combo (mas o n_trials ja penaliza a selecao)
    combo_cands = {k: config_stats[k] for k in config_stats if k.startswith("combo.")}
    if not combo_cands:
        P("\nSEM combo valido — dados insuficientes.")
        Path("data").mkdir(exist_ok=True)
        REPORT_PATH.write_text("\n".join(out))
        return "\n".join(out)

    champ_key = max(combo_cands, key=lambda k: combo_cands[k].sharpe_ann)
    champ = combo_cands[champ_key]
    champ_net = all_nets[champ_key]
    spy_c = spy_full.reindex(champ_net.index).fillna(0.0)

    # tribunal estatistico sobre o COMBO campeao, n_trials honesto
    trial_sharpes = [observed_sharpe(all_nets[k].values) for k in all_nets]
    verdict = evaluate_edge(
        champ_net.values,
        n_trials=nt,
        trial_sharpes=trial_sharpes,
        periods_per_year=EQUITY_PERIODS,
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # SPY buy&hold na mesma janela
    spy_stats = compute_stats("SPY_buyhold", spy_c, spy_c)

    # OOS: split temporal 70/30
    n = len(champ_net)
    cut = int(n * 0.7)
    is_sh = observed_sharpe(champ_net.values[:cut]) * np.sqrt(EQUITY_PERIODS)
    oos_sh = observed_sharpe(champ_net.values[cut:]) * np.sqrt(EQUITY_PERIODS)

    P("")
    P("=" * 78)
    P(f"CANDIDATO A GRADUACAO (COMBO campeao): {champ_key}")
    P("=" * 78)
    P(f"Sharpe_anual liq (estresse): {champ.sharpe_ann:.2f}")
    P(f"CAGR liq: {champ.cagr*100:.1f}%   MaxDD: {champ.maxdd*100:.1f}%")
    P(f"corr ao SPY: {champ.corr_spy:.2f}")
    P("")
    P(f"SPY buy&hold (mesma janela): Sharpe={spy_stats.sharpe_ann:.2f} "
      f"CAGR={spy_stats.cagr*100:.1f}% MaxDD={spy_stats.maxdd*100:.1f}%")
    P("")
    P("TRIBUNAL ESTATISTICO:")
    P("  " + verdict.summary())
    P(f"  PBO (CSCV, {len(all_nets)} configs): {pbo:.3f}  "
      f"({'OK <0.5' if pbo < 0.5 else 'RUIM >=0.5'})")
    P(f"  OOS split 70/30: IS Sharpe={is_sh:.2f}  OOS Sharpe={oos_sh:.2f}")
    P("")

    # ----- BARRA DE GRADUACAO --------------------------------------------------
    beats_spy = champ.sharpe_ann > spy_stats.sharpe_ann
    diversifies = (abs(champ.corr_spy) < 0.3) and champ.sharpe_ann >= 0.5
    robust_oos = oos_sh > 0.0 and (oos_sh >= 0.5 * is_sh if is_sh > 0 else oos_sh > 0)
    sharpe_robust = champ.sharpe_ann >= 0.8

    passed = bool(
        verdict.passes_dsr
        and sharpe_robust
        and (beats_spy or diversifies)
        and robust_oos
        and pbo < 0.5
    )

    P("-" * 78)
    P("BARRA DE GRADUACAO:")
    P(f"  DSR>=0.95 ............ {verdict.passes_dsr}  (DSR={verdict.dsr:.3f})")
    P(f"  Sharpe_liq>=0.8 ...... {sharpe_robust}  ({champ.sharpe_ann:.2f})")
    P(f"  bate SPY ............. {beats_spy}  ({champ.sharpe_ann:.2f} vs {spy_stats.sharpe_ann:.2f})")
    P(f"  OU diversifica ....... {diversifies}  (corr={champ.corr_spy:.2f})")
    P(f"  robusto OOS .......... {robust_oos}  (OOS={oos_sh:.2f} vs IS={is_sh:.2f})")
    P(f"  PBO<0.5 .............. {pbo < 0.5}  ({pbo:.3f})")
    P("")
    P(f"  >>> VEREDITO: {'PASSA' if passed else 'REPROVADO'} <<<")
    P("=" * 78)

    Path("data").mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(out))

    # marca p/ o runner extrair
    P(f"\n__PASSED__={passed}")
    P(f"__DSR__={verdict.dsr:.4f}")
    P(f"__SHARPE__={champ.sharpe_ann:.4f}")
    P(f"__CAGR__={champ.cagr:.4f}")
    P(f"__MAXDD__={champ.maxdd:.4f}")
    P(f"__CORR__={champ.corr_spy:.4f}")
    P(f"__PBO__={pbo:.4f}")
    P(f"__SPYSHARPE__={spy_stats.sharpe_ann:.4f}")
    P(f"__OOS__={oos_sh:.4f}")
    P(f"__IS__={is_sh:.4f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    force = "--force" in (argv or sys.argv[1:])
    print(run(force=force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
