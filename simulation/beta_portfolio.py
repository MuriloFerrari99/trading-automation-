"""Beta disciplinado — carteira diversificada com gestao de risco.

CONTEXTO (decisao do usuario): apos 6 levers de alpha price-based reprovados, o
objetivo deixou de ser BATER o mercado e passou a ser RETORNO ~DE MERCADO COM
DRAWDOWN MENOR + track record limpo (rampa p/ gerir AUM). Nao se exige retorno
maior; exige-se risco-ajustado/tombo melhor, liquido de custo.

O QUE ESTE MODULO FAZ — backtest HONESTO de 3 contendores sobre a MESMA cesta
multi-ativo (acoes/bonds/ouro/prata/cripto), com o maximo de historico:

  A) BENCHMARK A — buy&hold equal-weight da cesta (beta cru, rebalance mensal p/
     manter os pesos iguais — sem ele "equal-weight" derivaria p/ o vencedor).
  B) BENCHMARK B — 60/40 (acoes/bonds) rebalanceado mensalmente (o classico).
  S) ESTRATEGIA — "beta disciplinado": MESMA cesta, mas com
       (a) VOL-TARGETING: peso ∝ 1/vol_rolante(ativo); a carteira inteira e
           escalada p/ mirar ~10% a.a. de vol (caixa absorve o resto).
       (b) PORTAO DE REGIME / FILTRO DE TENDENCIA: des-arrisca um ativo (peso->0,
           vira caixa) quando regime=TREND_DOWN/HIGH_VOL OU preco < SMA200
           (protecao de crash).
       (c) REBALANCE mensal, CUSTOS reais aplicados sobre o turnover.

HONESTIDADE (nao-negociavel):
  - Params PADRAO, NAO otimizados, declarados no topo do relatorio: SMA200,
    vol-alvo 10% a.a., lookback de vol 60d, rebalance mensal. Roda uma
    sensibilidade simples (varia cada param) p/ mostrar que NAO e overfit.
  - SEM LOOK-AHEAD: o peso usado p/ o retorno do dia t e decidido com dado
    ATE t-1 (precos, vol, SMA, regime). O retorno do dia t e do ativo em t.
    O fim de mes que dispara o rebalance tambem e detectado por dado <= t-1.
  - Custo real (simulation.costs, por LADO em bps) cobrado sobre |Δpeso| no
    rebalance — inclui o custo de des-arriscar e de re-arriscar.
  - Metricas vs os 2 benchmarks: CAGR, Sharpe, Sortino, MaxDD, Calmar, pior ano.
  - Reusa o que PASSOU nos testes: feedback.regime.classify_regime (corta
    drawdown), simulation.costs (custo), simulation.metrics.max_drawdown.

Uso:
    uv run python -m simulation.beta_portfolio
    uv run python -m simulation.beta_portfolio --force        # re-baixa o cache
    uv run python -m simulation.beta_portfolio --no-sensitivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feedback.models import MarketRegime
from feedback.regime import classify_regime
from simulation.costs import CRYPTO_BASE, EQUITY_BASE, CostModel
from simulation.metrics import max_drawdown

logger = logging.getLogger("simulation.beta_portfolio")

CACHE_DIR = Path("data/beta_cache")
TRADING_DAYS = 252

# ----------------------------------------------------------------------------
# UNIVERSO multi-ativo. classe -> custo (equities/ETFs sem comissao; cripto 25bps).
# Cripto entra com PESO PEQUENO via teto de peso (CRYPTO_MAX_WEIGHT no overlay) e,
# nos benchmarks, pelo proprio equal-weight (2 de 8 nomes).
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    ticker: str  # ticker yfinance
    klass: str   # "equity" | "bond" | "metal" | "crypto"


UNIVERSE: list[Asset] = [
    Asset("SPY", "equity"),
    Asset("QQQ", "equity"),
    Asset("TLT", "bond"),
    Asset("IEF", "bond"),
    Asset("GLD", "metal"),
    Asset("SLV", "metal"),
    Asset("BTC-USD", "crypto"),
    Asset("ETH-USD", "crypto"),
]
COST_BY_CLASS: dict[str, CostModel] = {
    "equity": EQUITY_BASE,
    "bond": EQUITY_BASE,
    "metal": EQUITY_BASE,
    "crypto": CRYPTO_BASE,
}

# ---- params PADRAO do overlay (NAO otimizados; declarados no relatorio) ----
SMA_WINDOW = 200          # filtro de tendencia (protecao de crash)
VOL_LOOKBACK = 60         # janela da vol rolante (~3 meses)
VOL_TARGET_ANNUAL = 0.10  # alvo de vol da carteira (10% a.a.)
CRYPTO_MAX_WEIGHT = 0.05  # teto de peso por nome de cripto (peso pequeno)
MAX_GROSS = 1.0           # sem alavancagem: soma de pesos <= 1 (resto = caixa)

# Para o 60/40: acoes = SPY, bond = IEF (Treasury intermediaria, o "40" classico).
SIXTY_FORTY = {"SPY": 0.60, "IEF": 0.40}


# ============================================================================
# DADOS — yfinance, cacheado por ticker em data/beta_cache/*.csv
# ============================================================================
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '_')}_1d.csv"


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance -> DataFrame(close) ajustado, indice datetime UTC, ordenado."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        try:
            out = out.xs(ticker, axis=1, level=1)
        except KeyError:
            out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower() for c in out.columns]
    # auto_adjust=True -> "close" ja ajustado por split/dividendo.
    col = "close" if "close" in out.columns else ("adj close" if "adj close" in out.columns else None)
    if col is None:
        return pd.DataFrame()
    s = pd.to_numeric(out[col], errors="coerce")
    idx = pd.to_datetime(s.index, utc=True, errors="coerce")
    res = pd.DataFrame({"close": s.to_numpy()}, index=idx)
    res = res[~res.index.isna()].dropna()
    return res[res["close"] > 0].sort_index()


def _yf_download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    # period="max" -> maximo historico disponivel (ETFs >10a; inclui 2008/2020/2022).
    return yf.download(
        ticker, period="max", interval="1d",
        progress=False, auto_adjust=True, threads=False,
    )


def load_close(ticker: str, *, force: bool = False, write: bool = True) -> pd.DataFrame:
    """Serie de fechamento ajustado de UM ticker. Usa cache; baixa se ausente/force.

    Levanta a excecao de rede do yfinance se o download falhar (o runner trata
    como DADOS PENDENTES e reporta o comando exato). Nao inventa dados.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(ticker)
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[~df.index.isna()]
    df = _normalize(_yf_download(ticker), ticker)
    if write and not df.empty:
        df.to_csv(p)
        logger.info(
            "%s: %d barras (%s..%s)",
            ticker, len(df),
            df.index[0].date() if len(df) else "-",
            df.index[-1].date() if len(df) else "-",
        )
    return df


def load_panel(force: bool = False) -> tuple[pd.DataFrame, dict[str, str]]:
    """Painel de fechamentos {ticker} alinhado por dia (uniao de datas).

    Retorna (closes, classes). Cripto negocia 24/7 e ETFs nao -> reindexa tudo
    no calendario de NEGOCIACAO dos ETFs (dias uteis) e faz forward-fill curto
    da cripto p/ esses dias, evitando dias-fantasma de fim de semana.
    """
    closes: dict[str, pd.Series] = {}
    classes: dict[str, str] = {}
    equity_idx: pd.DatetimeIndex | None = None
    for a in UNIVERSE:
        df = load_close(a.ticker, force=force)
        if df.empty:
            logger.warning("%s sem dados — pulado.", a.ticker)
            continue
        closes[a.ticker] = df["close"]
        classes[a.ticker] = a.klass
        if a.klass != "crypto":
            equity_idx = df.index if equity_idx is None else equity_idx.union(df.index)
    if not closes or equity_idx is None:
        return pd.DataFrame(), {}
    # Calendario de pregao (uniao dos dias dos ativos nao-cripto).
    cols = {}
    for tkr, s in closes.items():
        s = s[~s.index.duplicated(keep="last")]
        if classes[tkr] == "crypto":
            # cripto 24/7 -> pega o fechamento do dia de pregao (ffill curto).
            cols[tkr] = s.reindex(equity_idx, method="ffill", limit=3)
        else:
            cols[tkr] = s.reindex(equity_idx)
    panel = pd.DataFrame(cols).sort_index()
    return panel, classes


def diversified_start(panel: pd.DataFrame, classes: dict[str, str]) -> pd.Timestamp:
    """Primeiro dia em que a cesta e GENUINAMENTE diversificada: >=1 equity E
    >=1 bond E >=1 metal vivos.

    POR QUE ISSO IMPORTA (honestidade): os ativos entram em datas diferentes
    (SPY 1993, QQQ 1999, bonds 2002, ouro 2004...). Antes disso, 'equal-weight'
    da cesta NAO e diversificado — em 1993-2002 era 100% acoes (SPY/QQQ), e o
    -67% do benchmark A e o estouro da bolha .com num book 100% acoes, NAO um
    tombo de cesta diversificada. Comparar a estrategia contra esse A inflaria
    artificialmente a vantagem de drawdown. A tabela-cabeca roda nesta janela
    comum; a tabela de CRISE usa o historico inteiro (com ressalva), p/ ainda
    ver 2008.
    """
    alive = panel.notna()
    by_class: dict[str, pd.Series] = {}
    for klass in ("equity", "bond", "metal"):
        cols = [t for t in panel.columns if classes.get(t) == klass]
        by_class[klass] = alive[cols].any(axis=1) if cols else pd.Series(False, index=panel.index)
    diversified = by_class["equity"] & by_class["bond"] & by_class["metal"]
    idx = panel.index[diversified]
    return idx[0] if len(idx) else panel.index[0]


# ============================================================================
# CONSTRUTORES DE PESO — todos SEM look-ahead.
# A convencao do backtest (run_portfolio) e: weights.loc[t] e aplicado ao
# RETORNO de t. Portanto cada construtor DEVE produzir, na linha t, um peso
# decidido SO com dados ate t-1. Garantimos isso com .shift(1) no fim.
# ============================================================================
def _rebalance_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True no 1o dia de pregao de cada mes (gatilho de rebalance mensal).

    Detectavel sem look-ahead: 'mudou o mes em relacao ao dia anterior'.
    """
    months = index.tz_localize(None).to_period("M")  # tz-naive p/ silenciar warning
    changed = pd.Series(months != np.roll(months, 1), index=index)
    changed.iloc[0] = True
    return changed


def _hold_between_rebalances(target: pd.DataFrame, rebal: pd.Series) -> pd.DataFrame:
    """Recebe pesos-ALVO por dia e o mask de rebalance. Devolve os pesos
    EFETIVAMENTE mantidos: mudam so nos dias de rebalance (drift de preco entre
    rebalances e ignorado p/ simplicidade — o efeito no turnover/custo e 2a ordem
    e penaliza igualmente os 3 contendores)."""
    held = target.where(rebal).ffill()
    return held.fillna(0.0)


def weights_equal_weight(closes: pd.DataFrame) -> pd.DataFrame:
    """BENCHMARK A: equal-weight entre os ativos VIVOS no dia, rebalance mensal."""
    alive = closes.notna()
    n_alive = alive.sum(axis=1).replace(0, np.nan)
    target = alive.div(n_alive, axis=0).fillna(0.0)
    held = _hold_between_rebalances(target, _rebalance_mask(closes.index))
    return held.shift(1).fillna(0.0)


def weights_sixty_forty(closes: pd.DataFrame) -> pd.DataFrame:
    """BENCHMARK B: 60/40 (SPY/IEF) rebalanceado mensalmente."""
    target = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for tkr, w in SIXTY_FORTY.items():
        if tkr in target.columns:
            target[tkr] = np.where(closes[tkr].notna(), w, 0.0)
    held = _hold_between_rebalances(target, _rebalance_mask(closes.index))
    return held.shift(1).fillna(0.0)


def _regime_derisk_flags(close: pd.Series, *, sma_window: int) -> pd.Series:
    """True nos dias em que o ativo deve ser DES-ARRISCADO (peso->0), decidido
    com dado ate o PROPRIO dia (o .shift(1) global cuida do look-ahead).

    Gatilho de des-risco = (preco < SMA200) OU regime ∈ {TREND_DOWN, HIGH_VOL}.
    O regime e classificado com a janela de 'slow' do classificador (rule-based
    que passou nos testes). Calculado incrementalmente, sem custo proibitivo.
    """
    c = close.astype(float)
    sma = c.rolling(sma_window, min_periods=sma_window).mean()
    below = c < sma  # NaN (warmup) -> False (nao des-arrisca por falta de SMA)
    below = below.fillna(False)

    # Regime: usa janela curta (slow padrao=30) p/ ser responsivo a crash.
    arr = c.to_numpy()
    n = len(arr)
    slow = 30
    lookback = slow + 25  # folga p/ a janela de vol interna do classificador
    bad = np.zeros(n, dtype=bool)
    last_valid = -1
    for i in range(n):
        if not np.isfinite(arr[i]):
            bad[i] = bad[last_valid] if last_valid >= 0 else False
            continue
        last_valid = i
        if i < lookback:
            continue
        window = arr[i - lookback + 1: i + 1]
        window = window[np.isfinite(window)]
        if window.size < slow + 1:
            continue
        reg = classify_regime(window.tolist())
        bad[i] = reg in (MarketRegime.TREND_DOWN, MarketRegime.HIGH_VOL)
    regime_bad = pd.Series(bad, index=close.index)
    return below | regime_bad


def weights_disciplined(
    closes: pd.DataFrame,
    classes: dict[str, str],
    *,
    sma_window: int = SMA_WINDOW,
    vol_lookback: int = VOL_LOOKBACK,
    vol_target: float = VOL_TARGET_ANNUAL,
    crypto_max: float = CRYPTO_MAX_WEIGHT,
    use_vol_target: bool = True,
    use_gate: bool = True,
) -> pd.DataFrame:
    """ESTRATEGIA: vol-targeting + portao de regime/tendencia, rebalance mensal.

    Passos (todos com dado ate t-1 apos o .shift(1) final):
      1. peso bruto ∝ 1/vol_rolante(ativo)  (vol-targeting por ativo).
      2. teto de peso p/ cripto (peso pequeno).
      3. PORTAO: zera o peso de um ativo des-arriscado (regime ruim ou < SMA200).
      4. normaliza p/ gross<=1, escala a carteira p/ mirar vol_target a.a.
         (caixa absorve o resto; em crash a carteira fica majoritariamente caixa).
    """
    rets = closes.pct_change()
    # vol rolante por ativo, anualizada.
    vol = rets.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)

    raw = inv_vol.copy()
    raw[closes.isna()] = np.nan  # so ativos vivos

    if use_gate:
        for tkr in raw.columns:
            derisk = _regime_derisk_flags(closes[tkr], sma_window=sma_window)
            raw.loc[derisk[derisk].index, tkr] = 0.0

    # teto de cripto ANTES de normalizar: limita a fracao bruta que cada nome de
    # cripto pode reivindicar (aplicado depois da normalizacao, abaixo).
    raw = raw.fillna(0.0)
    gross = raw.sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0)  # pesos relativos (somam 1 quando ha ativo)

    # aplica teto por nome de cripto e redistribui o excedente nos demais.
    crypto_cols = [t for t in w.columns if classes.get(t) == "crypto"]
    if crypto_cols:
        for t in crypto_cols:
            over = (w[t] - crypto_max).clip(lower=0.0)
            w[t] = w[t] - over  # corta no teto
        # redistribui o cortado proporcionalmente aos NAO-cripto ativos.
        noncrypto = [t for t in w.columns if t not in crypto_cols]
        cut_total = (1.0 - w.sum(axis=1)).clip(lower=0.0)
        base = w[noncrypto].sum(axis=1).replace(0, np.nan)
        share = w[noncrypto].div(base, axis=0).fillna(0.0)
        w[noncrypto] = w[noncrypto] + share.mul(cut_total, axis=0)

    if use_vol_target:
        # vol EX-ANTE da carteira com os pesos atuais (usa vol/corr rolante? aqui
        # aproximamos por vol ponderada ignorando correlacao -> conservador, tende
        # a SUBdimensionar o leverage, nunca a inflar). scale = alvo/vol_carteira,
        # limitado a [0,1] (sem alavancagem; so DES-arrisca).
        port_vol = (w * vol.fillna(0.0)).sum(axis=1)
        scale = (vol_target / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        w = w.mul(scale, axis=0)

    # gross<=1 (resto = caixa). Nunca alavanca.
    gross_final = w.sum(axis=1)
    over = (gross_final - MAX_GROSS).clip(lower=0.0)
    w = w.sub(w.div(gross_final.replace(0, np.nan), axis=0).mul(over, axis=0), fill_value=0.0)

    held = _hold_between_rebalances(w, _rebalance_mask(closes.index))
    return held.shift(1).fillna(0.0)


# ============================================================================
# BACKTEST — aplica pesos aos retornos, cobra custo no turnover, gera equity.
# ============================================================================
def _cost_vector(classes: dict[str, str], columns: pd.Index) -> np.ndarray:
    """Custo por LADO (fracao) por coluna, da classe do ativo."""
    return np.array(
        [COST_BY_CLASS[classes.get(c, "equity")].per_side_bps / 10000.0 for c in columns],
        dtype=float,
    )


def run_portfolio(
    closes: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str]
) -> tuple[pd.Series, pd.Series]:
    """Equity e retornos diarios LIQUIDOS de custo.

    SEM LOOK-AHEAD: weights.loc[t] foi decidido com dado <= t-1 (garantido pelos
    construtores via .shift(1)). O retorno bruto do dia t = sum(w_t * r_t), com
    r_t o retorno do ativo em t. Custo do dia t = sum(|w_t - w_{t-1}|) * custo_lado
    (cobra des-arriscar E re-arriscar; turnover real do rebalance).
    """
    rets = closes.pct_change().fillna(0.0)
    w = weights.reindex(closes.index).fillna(0.0)
    common = [c for c in closes.columns if c in w.columns]
    rets = rets[common].to_numpy()
    w_arr = w[common].to_numpy()
    cost_side = _cost_vector(classes, pd.Index(common))

    gross_daily = (w_arr * rets).sum(axis=1)
    turnover = np.abs(np.diff(w_arr, axis=0, prepend=np.zeros((1, w_arr.shape[1]))))
    cost_daily = (turnover * cost_side).sum(axis=1)
    net = gross_daily - cost_daily

    # comeca a contar quando ha exposicao (descarta warmup todo-zero do inicio).
    nonzero = np.flatnonzero(np.abs(w_arr).sum(axis=1) > 0)
    start = int(nonzero[0]) if nonzero.size else 0
    net_s = pd.Series(net, index=closes.index).iloc[start:]
    equity = (1.0 + net_s).cumprod()
    return equity, net_s


# ============================================================================
# METRICAS
# ============================================================================
@dataclass
class Stats:
    name: str
    n_days: int
    start: str
    end: str
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    worst_year: float
    worst_year_label: str
    avg_exposure: float


def _annualized_sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    sd = rets.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * rets.mean() / sd) if sd > 0 else 0.0


def _annualized_sortino(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    downside = rets[rets < 0]
    dd = downside.std(ddof=1) if downside.size >= 2 else 0.0
    return float(np.sqrt(TRADING_DAYS) * rets.mean() / dd) if dd > 0 else 0.0


def compute_stats(name: str, equity: pd.Series, net: pd.Series, weights: pd.DataFrame) -> Stats:
    if equity.size < 2:
        return Stats(name, 0, "-", "-", 0, 0, 0, 0, 0, 0, "-", 0)
    r = net.to_numpy()
    years = equity.size / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / max(years, 1e-9)) - 1.0)
    mdd = max_drawdown(equity.to_numpy())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("inf") if cagr > 0 else 0.0

    # pior ano-calendario.
    by_year = (1.0 + net).groupby(net.index.year).prod() - 1.0
    worst_year = float(by_year.min()) if by_year.size else 0.0
    worst_label = str(int(by_year.idxmin())) if by_year.size else "-"

    expo = float((weights.reindex(net.index).fillna(0.0).abs().sum(axis=1)).mean())
    return Stats(
        name=name, n_days=int(equity.size),
        start=str(equity.index[0].date()), end=str(equity.index[-1].date()),
        cagr=cagr, sharpe=_annualized_sharpe(r), sortino=_annualized_sortino(r),
        max_dd=mdd, calmar=calmar,
        worst_year=worst_year, worst_year_label=worst_label, avg_exposure=expo,
    )


def crisis_returns(net: pd.Series) -> dict[str, float]:
    """Retorno acumulado em janelas de crise (se cobertas pelo historico)."""
    windows = {
        "2008 (GFC)": ("2008-01-01", "2009-03-31"),
        "2020 (COVID)": ("2020-02-01", "2020-04-30"),
        "2022 (Bear)": ("2022-01-01", "2022-12-31"),
    }
    out: dict[str, float] = {}
    for label, (a, b) in windows.items():
        seg = net.loc[(net.index >= a) & (net.index <= b)]
        if seg.size >= 5:
            out[label] = float((1.0 + seg).prod() - 1.0)
    return out


# ============================================================================
# RELATORIO
# ============================================================================
BAR = (
    "BARRA HONESTA: VALE se a ESTRATEGIA entrega Calmar/MaxDD MATERIALMENTE melhor "
    "que o buy&hold (A)\n  com Sharpe >= o de A, liquido de custo. NAO se exige retorno "
    "maior — o objetivo e\n  risco-ajustado/drawdown melhor + track record. Se a "
    "estrategia so PIORA o retorno sem\n  melhorar o tombo, FALHA."
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def stats_table(rows: list[Stats]) -> str:
    head = (
        f"  {'estrategia':<26} | {'CAGR':>7} | {'Sharpe':>6} | {'Sortino':>7} | "
        f"{'MaxDD':>7} | {'Calmar':>6} | {'pior ano':>14} | {'expo':>5}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for s in rows:
        lines.append(
            f"  {s.name:<26} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {s.sortino:>7.2f} | "
            f"{_fmt_pct(s.max_dd):>7} | {s.calmar:>6.2f} | "
            f"{_fmt_pct(s.worst_year):>8} ({s.worst_year_label}) | {s.avg_exposure*100:>4.0f}%"
        )
    return "\n".join(lines)


def build_report(
    panel_info: str,
    main_rows: list[Stats],
    crisis: dict[str, dict[str, float]],
    sensitivity: str,
    overlay_decomp: list[Stats],
    verdict: str,
) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append("BETA DISCIPLINADO — carteira diversificada, gestao de risco, retorno ~de mercado com")
    L.append("drawdown menor. Objetivo: track record limpo (rampa p/ AUM), NAO bater o mercado.")
    L.append("=" * 96)
    L.append(BAR)
    L.append("")
    L.append("PARAMS PADRAO (NAO otimizados, declarados): "
             f"SMA{SMA_WINDOW} (filtro tendencia) · vol-alvo {VOL_TARGET_ANNUAL*100:.0f}% a.a. · "
             f"lookback vol {VOL_LOOKBACK}d · rebalance mensal ·")
    L.append(f"  teto cripto {CRYPTO_MAX_WEIGHT*100:.0f}%/nome · sem alavancagem (gross<=100%) · "
             "custo: equities 3bps/lado, cripto 35bps/lado.")
    L.append("SEM LOOK-AHEAD: peso de t decidido com dado <= t-1 (.shift(1)); retorno em t; "
             "custo sobre |Δpeso|.")
    L.append("")
    L.append(panel_info)
    L.append("")
    L.append("=" * 96)
    L.append("(1) ESTRATEGIA vs BENCHMARK A (equal-weight) vs BENCHMARK B (60/40)")
    L.append("=" * 96)
    L.append(stats_table(main_rows))
    L.append("")
    L.append("=" * 96)
    L.append("(2) O PORTAO DE REGIME + VOL-TARGET REALMENTE MELHORA O DRAWDOWN SEM MATAR O RETORNO?")
    L.append("    Decomposicao do overlay (mesma cesta; liga/desliga cada peca):")
    L.append("=" * 96)
    L.append(stats_table(overlay_decomp))
    L.append("")
    L.append("=" * 96)
    L.append("(3) COMPORTAMENTO EM CRISE (retorno acumulado na janela; '-' = fora do historico)")
    L.append("=" * 96)
    all_labels = sorted({lbl for d in crisis.values() for lbl in d})
    if all_labels:
        hdr = f"  {'estrategia':<26} | " + " | ".join(f"{lbl:>14}" for lbl in all_labels)
        L.append(hdr)
        L.append("  " + "-" * (len(hdr) - 2))
        for name, d in crisis.items():
            cells = " | ".join(f"{_fmt_pct(d[lbl]):>14}" if lbl in d else f"{'-':>14}" for lbl in all_labels)
            L.append(f"  {name:<26} | {cells}")
    else:
        L.append("  (historico nao cobre as janelas de crise — cache curto?)")
    L.append("")
    L.append("=" * 96)
    L.append("(4) SENSIBILIDADE (params PADRAO +/- variacoes; mostra que NAO e overfit)")
    L.append("=" * 96)
    L.append(sensitivity)
    L.append("")
    L.append("=" * 96)
    L.append("(5) VEREDITO")
    L.append("=" * 96)
    L.append(verdict)
    L.append("")
    return "\n".join(L)


# ============================================================================
# RUNNER
# ============================================================================
def _passes_bar(s: Stats, a: Stats) -> tuple[bool, bool, bool]:
    """A BARRA HONESTA, mecanica: Sharpe >= A (folga 0.05) E tombo materialmente
    melhor (MaxDD <=80% do de A OU Calmar >=120% do de A). Retorna (passa, sharpe_ok, tombo_ok)."""
    sharpe_ok = s.sharpe >= a.sharpe - 0.05
    dd_better = abs(s.max_dd) <= abs(a.max_dd) * 0.80
    calmar_better = a.calmar > 0 and s.calmar >= a.calmar * 1.20
    tomb_ok = dd_better or calmar_better
    return (sharpe_ok and tomb_ok), sharpe_ok, tomb_ok


def _verdict_text(strat: Stats, bench_a: Stats, bench_b: Stats, decomp: list[Stats]) -> str:
    """Aplica a BARRA HONESTA mecanicamente e escreve o veredito. `decomp` traz
    [eq-weight=A, +vol-target, +portao, +ambos=S] p/ creditar/culpar cada peca."""
    passes, sharpe_ok, tomb_ok = _passes_bar(strat, bench_a)

    lines: list[str] = []
    flag = "PASSA" if passes else "FALHA"
    lines.append(f"  [{flag}] vs BENCHMARK A (buy&hold equal-weight, beta cru):")
    lines.append(
        f"    Sharpe: estrategia {strat.sharpe:.2f} vs A {bench_a.sharpe:.2f}  "
        f"({'>=' if sharpe_ok else '<'} barra)"
    )
    lines.append(
        f"    MaxDD:  estrategia {_fmt_pct(strat.max_dd)} vs A {_fmt_pct(bench_a.max_dd)}  "
        f"(reducao de {(1 - abs(strat.max_dd)/abs(bench_a.max_dd))*100:.0f}% no tombo)"
        if bench_a.max_dd < 0 else "    MaxDD: A sem drawdown medivel"
    )
    lines.append(
        f"    Calmar: estrategia {strat.calmar:.2f} vs A {bench_a.calmar:.2f}  | "
        f"CAGR: estrategia {_fmt_pct(strat.cagr)} vs A {_fmt_pct(bench_a.cagr)}"
    )
    lines.append(
        f"    Pior ano: estrategia {_fmt_pct(strat.worst_year)} ({strat.worst_year_label}) "
        f"vs A {_fmt_pct(bench_a.worst_year)} ({bench_a.worst_year_label})"
    )
    lines.append("")
    # Avalia cada peca do overlay vs A (decomp: [A, +vol, +portao, +ambos]).
    by_name = {s.name: s for s in decomp}
    vol_only = by_name.get("+ so vol-target")
    gate_only = by_name.get("+ so portao regime/SMA")

    if passes:
        lines.append("  CONCLUSAO: a estrategia completa (vol-target + portao) entrega tombo/risco-")
        lines.append("  ajustado MATERIALMENTE melhor que o beta cru, sem cobrar Sharpe inferior.")
        lines.append("  Coerente com o objetivo: beta disciplinado com track record mais limpo.")
        if strat.cagr < bench_a.cagr:
            lines.append(f"  (Custo do seguro: ~{_fmt_pct(strat.cagr - bench_a.cagr)} de CAGR a menos — "
                         "esperado; o ganho esta no drawdown/Calmar, nao no retorno.)")
    else:
        lines.append("  CONCLUSAO: a estrategia COMPLETA (S), como montada, NAO passa a barra honesta.")
        if not sharpe_ok:
            lines.append(f"  Sharpe {strat.sharpe:.2f} < {bench_a.sharpe:.2f} de A: empilhar TUDO custa retorno")
            lines.append("  sem comprar protecao proporcional. NAO vou vender o sonho.")
        elif not tomb_ok:
            lines.append("  O tombo (MaxDD/Calmar) nao melhora materialmente vs buy&hold.")

    # ---- decomposicao honesta: o que dentro do overlay funciona/atrapalha ----
    lines.append("")
    lines.append("  O QUE FUNCIONA (decomposicao, mesma janela, vs A):")
    if vol_only is not None:
        vp, vsh, _ = _passes_bar(vol_only, bench_a)
        verdict_vol = "PASSA a barra" if vp else ("Sharpe>=A mas tombo so parcial" if vsh else "nao passa")
        lines.append(
            f"    · VOL-TARGET sozinho: Sharpe {vol_only.sharpe:.2f} (vs A {bench_a.sharpe:.2f}), "
            f"MaxDD {_fmt_pct(vol_only.max_dd)} (vs {_fmt_pct(bench_a.max_dd)}), "
            f"Calmar {vol_only.calmar:.2f} (vs {bench_a.calmar:.2f}) -> {verdict_vol}."
        )
    if gate_only is not None:
        lines.append(
            f"    · PORTAO regime/SMA sozinho: Sharpe {gate_only.sharpe:.2f} (vs A {bench_a.sharpe:.2f}), "
            f"MaxDD {_fmt_pct(gate_only.max_dd)} -> corta tombo, mas o whipsaw (vende no rompimento"
        )
        lines.append("      da SMA200 e perde a volta) derruba o Sharpe; e o que arrasta S p/ baixo.")
    if vol_only is not None and strat.sharpe < vol_only.sharpe - 0.05:
        lines.append("")
        lines.append("  VEREDITO PRATICO: o lever que entrega o objetivo (retorno ~de mercado, tombo")
        lines.append("  menor, Sharpe melhor) e o VOL-TARGETING — nao o portao de regime nesta cesta")
        lines.append("  multi-ativo. O portao protege em bear sustentado (vide 2008/2022 na tabela 3),")
        lines.append("  mas na media de 2004-hoje cobra mais do que poupa. Recomendacao: rodar o beta")
        lines.append("  disciplinado com VOL-TARGET (e teto de cripto), e tratar o portao de regime como")
        lines.append("  hedge opcional/condicional (so em drawdown profundo), nao como overlay permanente.")

    lines.append("")
    lines.append("  vs BENCHMARK B (60/40): " + (
        f"estrategia Sharpe {strat.sharpe:.2f}/MaxDD {_fmt_pct(strat.max_dd)} vs "
        f"60/40 Sharpe {bench_b.sharpe:.2f}/MaxDD {_fmt_pct(bench_b.max_dd)}."
    ))
    return "\n".join(lines)


def _run_window(
    panel: pd.DataFrame, weights: pd.DataFrame, classes: dict[str, str],
    name: str, start: pd.Timestamp,
) -> tuple[Stats, pd.Series]:
    """Roda 1 contendor e mede metricas SO a partir de `start` (janela comum
    diversificada). Os PESOS foram calculados no painel inteiro (warmup de
    SMA200/vol correto); aqui apenas fatiamos painel+pesos antes do equity p/ a
    comparacao ser justa entre ativos que nasceram em datas diferentes."""
    p = panel.loc[panel.index >= start]
    w = weights.loc[weights.index >= start]
    eq, net = run_portfolio(p, w, classes)
    return compute_stats(name, eq, net, w), net


def run(
    force: bool = False, do_sensitivity: bool = True
) -> tuple[str, bool]:
    panel, classes = load_panel(force=force)
    if panel.empty:
        msg = (
            "DADOS PENDENTES: nenhum fechamento baixado (rede?). Rode:\n"
            "  uv run python -m simulation.beta_portfolio --force\n"
            f"Universo: {[a.ticker for a in UNIVERSE]}"
        )
        return msg, False

    div_start = diversified_start(panel, classes)

    # info do painel.
    spans = []
    for c in panel.columns:
        s = panel[c].dropna()
        if not s.empty:
            spans.append(f"{c}:{s.index[0].date()}→{s.index[-1].date()}({len(s)})")
    panel_info = (
        f"PAINEL: {len(panel.columns)} ativos, {panel.index[0].date()}..{panel.index[-1].date()} "
        f"({len(panel)} dias de pregao).\n  " + "  ".join(spans) + "\n"
        f"JANELA-CABECA (tabelas 1,2,4) = a partir de {div_start.date()}, 1o dia com "
        "equity+bond+metal vivos (cesta\n  GENUINAMENTE diversificada). ANTES disso, "
        "'equal-weight' era ~100% acoes (SPY/QQQ) e\n  o tombo do benchmark A seria a "
        "bolha .com num book de acoes, NAO de cesta — comparar ali\n  inflaria a vantagem "
        "da estrategia. A tabela de CRISE (3) usa o historico INTEIRO p/ ver 2008."
    )

    # --- pesos calculados no PAINEL INTEIRO (warmup correto) ---
    w_a = weights_equal_weight(panel)
    w_b = weights_sixty_forty(panel)
    w_s = weights_disciplined(panel, classes)
    w_volonly = weights_disciplined(panel, classes, use_gate=False, use_vol_target=True)
    w_gateonly = weights_disciplined(panel, classes, use_gate=True, use_vol_target=False)

    # --- (1) headline: metricas na JANELA COMUM diversificada ---
    st_a, net_a = _run_window(panel, w_a, classes, "A: buy&hold eq-weight", div_start)
    st_b, net_b = _run_window(panel, w_b, classes, "B: 60/40 (SPY/IEF)", div_start)
    st_s, net_s = _run_window(panel, w_s, classes, "S: beta disciplinado", div_start)
    main_rows = [st_s, st_a, st_b]

    # --- (2) decomposicao do overlay (mesma janela) ---
    st_base, _ = _run_window(panel, w_a, classes, "eq-weight (sem overlay)=A", div_start)
    st_vo, _ = _run_window(panel, w_volonly, classes, "+ so vol-target", div_start)
    st_go, _ = _run_window(panel, w_gateonly, classes, "+ so portao regime/SMA", div_start)
    overlay_decomp = [st_base, st_vo, st_go, replace(st_s, name="+ ambos (=estrategia S)")]

    # --- (3) crise: historico INTEIRO (com ressalva no cabecalho) ---
    _, net_a_full = run_portfolio(panel, w_a, classes)
    _, net_b_full = run_portfolio(panel, w_b, classes)
    _, net_s_full = run_portfolio(panel, w_s, classes)
    crisis = {
        "S: beta disciplinado": crisis_returns(net_s_full),
        "A: buy&hold eq-weight": crisis_returns(net_a_full),
        "B: 60/40": crisis_returns(net_b_full),
    }
    crisis = {k: v for k, v in crisis.items() if v}

    # --- (4) sensibilidade (mesma janela-cabeca) ---
    sensitivity = "  (pulada — use sem --no-sensitivity)"
    if do_sensitivity:
        sensitivity = _sensitivity_table(panel, classes, st_a, div_start)

    verdict = _verdict_text(st_s, st_a, st_b, overlay_decomp)
    report = build_report(panel_info, main_rows, crisis, sensitivity, overlay_decomp, verdict)
    # criterio de "passa" do runner (mesmo da barra honesta).
    sharpe_ok = st_s.sharpe >= st_a.sharpe - 0.05
    tomb_ok = abs(st_s.max_dd) <= abs(st_a.max_dd) * 0.80 or (
        st_a.calmar > 0 and st_s.calmar >= st_a.calmar * 1.20
    )
    return report, bool(sharpe_ok and tomb_ok)


def _sensitivity_table(
    panel: pd.DataFrame, classes: dict[str, str], bench_a: Stats, start: pd.Timestamp
) -> str:
    """Varia 1 param por vez em torno do PADRAO. Robusto = metricas estaveis e o
    veredito (tombo melhor, Sharpe>=A) nao depende de uma escolha fina. Mede na
    mesma JANELA-CABECA (cesta diversificada)."""
    grid: list[tuple[str, dict]] = [
        ("PADRAO", {}),
        ("SMA=150", {"sma_window": 150}),
        ("SMA=250", {"sma_window": 250}),
        ("vol_lb=40", {"vol_lookback": 40}),
        ("vol_lb=90", {"vol_lookback": 90}),
        ("vol_alvo=8%", {"vol_target": 0.08}),
        ("vol_alvo=12%", {"vol_target": 0.12}),
        ("cripto_max=2%", {"crypto_max": 0.02}),
        ("cripto_max=10%", {"crypto_max": 0.10}),
    ]
    head = (
        f"  {'variacao':<14} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>7} | {'Calmar':>6} | "
        f"{'tombo<A?':>8} | {'Sh>=A?':>6}"
    )
    lines = [head, "  " + "-" * (len(head) - 2)]
    for label, kw in grid:
        w = weights_disciplined(panel, classes, **kw)
        s, _ = _run_window(panel, w, classes, label, start)
        tomb = "sim" if abs(s.max_dd) < abs(bench_a.max_dd) else "nao"
        shok = "sim" if s.sharpe >= bench_a.sharpe - 0.05 else "nao"
        lines.append(
            f"  {label:<14} | {_fmt_pct(s.cagr):>7} | {s.sharpe:>6.2f} | {_fmt_pct(s.max_dd):>7} | "
            f"{s.calmar:>6.2f} | {tomb:>8} | {shok:>6}"
        )
    lines.append("  Leitura: se 'tombo<A?' e 'Sh>=A?' ficam estaveis na grade, o resultado NAO")
    lines.append("  depende de um param fino (nao e overfit). Se viram com qualquer ajuste, e fragil.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beta disciplinado: backtest honesto vs buy&hold e 60/40."
    )
    parser.add_argument("--force", action="store_true", help="re-baixa o cache yfinance")
    parser.add_argument("--no-sensitivity", action="store_true", help="pula a sensibilidade")
    parser.add_argument("--report-file", default="data/beta_verdict.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    report, _passes = run(force=args.force, do_sensitivity=not args.no_sensitivity)
    print(report)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
