"""Backtest POINT-IN-TIME de alpha de sentimento de noticias (cross-sectional).

Tese: o sentimento das manchetes (informacao ORTOGONAL ao preco) preve o
retorno relativo entre acoes no dia seguinte. Se sim, e alpha de verdade —
nao beta de bull market (que ja derrubou breakout/momentum/regime-timing).

Honestidade por construcao:
  - PIT: a manchete so entra no sinal do dia em que foi publicada (created_at);
    a posicao e tomada NO DIA SEGUINTE (sem look-ahead).
  - Sinal = EWMA do sentimento diario por simbolo (a "visao" decai quando nao
    ha noticia — sentimento velho perde peso).
  - Long/short DEMEANADO cross-section (dollar-neutral) => retorno e ~puro alpha,
    nao beta. Reportamos a correlacao com a cesta para PROVAR ortogonalidade.
  - Custos por TURNOVER real; barra = bater buy&hold (alpha), nao zero.

Uso:
    uv run python -m simulation.news_alpha
    uv run python -m simulation.news_alpha --halflife 5 --report-file data/news_alpha_verdict.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lm_sentiment import score_headline
from simulation.news_data import _cache_path as _news_path
from simulation.statistics import EQUITY_PERIODS, evaluate_edge, observed_sharpe
from simulation.metrics import max_drawdown

EQUITIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "JPM",
    "V", "MA", "XOM", "KO", "PG", "JNJ", "UNH", "HD", "WMT", "DIS",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "GLD",
]
PRICE_CACHE = Path("data/cache")
EQUITY_RT_COST = 6.0 / 10000.0  # 6 bps round-trip (EQUITY_BASE) por unidade de turnover


def _load_prices(sym: str) -> pd.Series | None:
    p = PRICE_CACHE / f"{sym.replace('/', '_')}_1d.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    s = pd.Series(df["close"].astype(float).to_numpy(), index=idx.normalize())
    return s[~s.index.isna()]


def _daily_sentiment(sym: str, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Sentimento medio das manchetes por DIA de pregao (PIT: por data de publicacao)."""
    p = _news_path(sym)
    if not p.exists():
        return pd.Series(0.0, index=trading_days)
    df = pd.read_csv(p)
    if df.empty:
        return pd.Series(0.0, index=trading_days)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["score"] = df["headline"].map(score_headline)
    # data de pregao = data calendario da publicacao (acao so no dia seguinte)
    df["day"] = df["created_at"].dt.normalize()
    daily = df.groupby("day")["score"].mean()
    return daily.reindex(trading_days).fillna(0.0)


def build_panel(symbols: list[str], halflife: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (retornos_diarios, estado_de_sentimento) alinhados por data."""
    prices = {s: _load_prices(s) for s in symbols}
    prices = {s: v for s, v in prices.items() if v is not None and len(v) > 50}
    rets = pd.DataFrame({s: v.pct_change() for s, v in prices.items()}).sort_index()
    days = rets.index
    sent_raw = pd.DataFrame({s: _daily_sentiment(s, days) for s in prices})
    # EWMA: sentimento velho decai; dias sem noticia (0) puxam a visao p/ neutro
    state = sent_raw.ewm(halflife=halflife, min_periods=1).mean()
    return rets, state


def long_short_returns(rets: pd.DataFrame, state: pd.DataFrame, *, frac: float = 0.33) -> tuple[pd.Series, pd.Series, float]:
    """Carteira long/short dollar-neutral por tercil de sentimento (sinal de t -> retorno t+1).

    Retorna (serie_liq_de_custo, serie_bruta, correlacao_com_a_cesta)."""
    cols = rets.columns
    n = len(cols)
    k = max(1, int(n * frac))
    W = pd.DataFrame(0.0, index=rets.index, columns=cols)
    z = state.sub(state.mean(axis=1), axis=0)  # demeana cross-section (neutro)
    for t in range(len(z)):
        row = z.iloc[t].dropna()
        if row.size < 2 * k:
            continue
        order = row.sort_values()
        shorts = order.index[:k]; longs = order.index[-k:]
        W.iloc[t, W.columns.get_indexer(longs)] = 1.0 / k
        W.iloc[t, W.columns.get_indexer(shorts)] = -1.0 / k
    # posicao de t aplicada no retorno de t+1 (sem look-ahead)
    pos = W.shift(1).fillna(0.0)
    gross = (pos * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    net = gross - turnover * EQUITY_RT_COST
    basket = rets[cols].mean(axis=1)
    corr = float(net.corr(basket))
    return net.dropna(), gross.dropna(), corr


def _row(name: str, r: np.ndarray, periods: int, n_trials: int, corr: float | None = None) -> str:
    r = np.nan_to_num(np.asarray(r, float))
    if r.size < 30 or r.std() == 0:
        return f"{name:<34} amostra insuficiente"
    eq = np.cumprod(1 + r); yrs = len(r) / periods
    sh = observed_sharpe(r, periods_per_year=periods)
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = max_drawdown(eq)
    v = evaluate_edge(r, n_trials=n_trials, periods_per_year=periods, min_sharpe_annual=0.5)
    cc = f"{corr:+.2f}" if corr is not None else "  -"
    return (f"{name:<34}{sh:>7.2f}{cagr*100:>8.1f}%{mdd*100:>8.1f}%{cc:>7}"
            f"{v.psr:>8.3f}{v.dsr:>8.3f}  {'PASSA' if v.dsr>=0.95 else 'FALHA'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest PIT de alpha de sentimento de noticias")
    parser.add_argument("--report-file", default="data/news_alpha_verdict.txt")
    parser.add_argument("--n-trials", type=int, default=6, help="n_trials honesto (halflifes testados)")
    args = parser.parse_args(argv)

    header = [
        "ALPHA DE SENTIMENTO DE NOTICIAS — cross-sectional, point-in-time, market-neutral",
        "Sinal = EWMA do sentimento (Loughran-McDonald estendido) das manchetes (Alpaca/Benzinga).",
        "Posicao de t aplicada em t+1 (sem look-ahead). Custo por turnover (6bps RT).",
        "corr = correlacao com a cesta EW (proximo de 0 => ortogonal ao beta = alpha de verdade).",
    ]
    lines = list(header)
    lines.append("")
    lines.append(f"{'config':<34}{'Sharpe':>7}{'CAGR':>8}{'MaxDD':>8}{'corr':>7}{'PSR':>8}{'DSR':>8}  veredito")
    lines.append("-" * 96)

    # benchmark: cesta EW long-only (o alvo a bater = beta)
    rets0, _ = build_panel(EQUITIES, halflife=5)
    basket = rets0.mean(axis=1).dropna().to_numpy()
    lines.append(_row("Cesta EW long-only (BETA alvo)", basket, EQUITY_PERIODS, 1))
    lines.append("-" * 96)

    best = None
    for hl in (2, 3, 5, 8, 13):
        rets, state = build_panel(EQUITIES, halflife=hl)
        net, gross, corr = long_short_returns(rets, state)
        lines.append(_row(f"L/S sentimento (halflife={hl})", net.to_numpy(), EQUITY_PERIODS, args.n_trials, corr))
        sh = observed_sharpe(np.nan_to_num(net.to_numpy()), periods_per_year=EQUITY_PERIODS)
        if best is None or sh > best[0]:
            best = (sh, hl, gross, net, corr)

    # bruto (sem custo) da melhor config — quanto do edge o custo come
    if best:
        _, hl, gross, net, corr = best
        lines.append("-" * 96)
        lines.append(_row(f"melhor L/S BRUTO (hl={hl}, s/custo)", gross.to_numpy(), EQUITY_PERIODS, 1, corr))

    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
