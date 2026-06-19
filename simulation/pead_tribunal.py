"""Tribunal de PEAD — Post-Earnings-Announcement Drift (R&D ISOLADO).

PERGUNTA (imparcial, MEDIR): apos um resultado trimestral, a acao deriva na direcao
da surpresa por semanas (PEAD)? Long surpresa positiva / short surpresa negativa,
segurando ~20-60 pregoes. Tem edge risco-ajustado ROBUSTO e/ou diversifica o mercado?

Este modulo e R&D PURO. NAO toca em beta_*, main.py, nav_history nem config de producao.
Cria os SEUS proprios arquivos/cache. IMPORTA (read-only):
  - simulation.statistics : evaluate_edge / DSR / PSR / PBO / observed_sharpe
  - simulation.metrics    : max_drawdown
  - simulation.costs      : EQUITY_BASE (custo real de equity) + cenario estressado

DADOS (GRATIS, yfinance):
  - Datas de earnings + EPS estimado/realizado + Surprise(%) via Ticker.get_earnings_dates.
  - Precos: close ajustado diario (auto_adjust=True), period=max.
  - Cacheado em data/pead_cache/*.csv -> roda offline apos um fetch.
  - SPY como benchmark/mercado.

SURPRESA (sinal):
  - PRIMARIO: Surprise(%) REAL reportado pelo yfinance (EPS realizado vs estimado).
  - FALLBACK declarado: quando Surprise(%) ausente, usa a REACAO DE PRECO no dia do
    anuncio (gap/retorno do dia event) como proxy de surpresa. Isto e DECLARADO e
    testado como variante separada (gap-proxy) — nao se mistura com o sinal real.

PIT / SEM LOOK-AHEAD (rigoroso):
  - O timestamp do anuncio inclui hora (a maioria 16:00 = APOS o fechamento). Tratamos
    o anuncio como conhecido SO a partir do PROXIMO pregao. Entrada no close do 1o pregao
    >= dia seguinte ao anuncio (age APOS o anuncio). Saida apos `hold` pregoes.
  - O retorno do dia do evento (usado no gap-proxy) e do proprio dia do anuncio, mas a
    DECISAO de entrar so ocorre no pregao seguinte -> sem look-ahead (sinal de t com
    dado<=t, posicao a partir de t+1).
  - Surprise(%) real e um numero reportado JUNTO ao anuncio -> disponivel em t+1. OK.

CUSTO: EQUITY_BASE (3 bps/lado) round-trip por trade + cenario ESTRESSADO 2x.
  Veredito usa o cenario ESTRESSADO. Short tem o mesmo custo de execucao (ignora
  custo de borrow -> otimista p/ o short; declarado).

N_TRIALS HONESTO (anti data-snooping): conta TODAS as variantes varridas:
  sinal {surprise_real, gap_proxy} x hold {20,30,40,60} x corte de |surpresa|
  {p20, p33} = 2*4*2 = 16 configs. DSR usa n_trials=16; PBO sobre a matriz das 16.

BARRA (regua do programa): PASSA so se, no cenario ESTRESSADO:
  DSR >= 0.95  E  Sharpe liq robusto  E
  (bate buy&hold/SPY  OU  diversifica: corr baixa ao SPY melhorando o conjunto),
  robusto OOS (split temporal). Senao FALHA -> cemiterio. Imparcial.

Uso:
    uv run python -m simulation.pead_tribunal            # usa cache
    uv run python -m simulation.pead_tribunal --force    # re-baixa
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.costs import EQUITY_BASE
from simulation.metrics import max_drawdown
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("pead")

CACHE_DIR = Path("data/pead_cache")
REPORT = Path("data/pead_verdict.txt")
SPY = "SPY"

# Universo: nomes liquidos large/mega-cap US com historico longo de earnings.
# Diversidade setorial. ~40 nomes -> milhares de eventos.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CSCO",
    "ORCL", "IBM", "QCOM", "TXN", "ADBE", "CRM",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY",
    "WMT", "HD", "MCD", "NKE", "SBUX", "KO", "PEP", "PG", "COST",
    "XOM", "CVX", "CAT", "BA", "GE", "DIS", "VZ", "T",
]


# ----------------------------------------------------------------------------- data
def _price_cache(tk: str) -> Path:
    return CACHE_DIR / f"px_{tk}.csv"


def _earn_cache(tk: str) -> Path:
    return CACHE_DIR / f"earn_{tk}.csv"


def _load_prices(tk: str, force: bool = False) -> pd.Series:
    """Close ajustado diario (tz-naive, ordenado). Cacheado."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _price_cache(tk)
    if p.exists() and not force:
        s = pd.read_csv(p, index_col=0)
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()]
        return s["close"].astype(float).sort_index()
    import yfinance as yf

    df = yf.download(tk, period="max", interval="1d", progress=False,
                     auto_adjust=True, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"sem precos para {tk}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"close": close.astype(float)})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[out["close"] > 0].sort_index()
    out.to_csv(p)
    return out["close"]


def _load_earnings(tk: str, force: bool = False) -> pd.DataFrame:
    """Eventos de earnings: index=data(naive), cols EPS Estimate/Reported EPS/Surprise(%).

    Cacheado. Levanta se nao conseguir (runner trata como dado pendente).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _earn_cache(tk)
    if p.exists() and not force:
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df[~df.index.isna()].sort_index()
    import yfinance as yf

    ed = yf.Ticker(tk).get_earnings_dates(limit=80)
    if ed is None or ed.empty:
        raise RuntimeError(f"sem earnings para {tk}")
    df = ed.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_csv(p)
    return df


# ----------------------------------------------------------------------------- events
@dataclass
class Event:
    ticker: str
    ann_date: pd.Timestamp        # dia (naive) do anuncio
    surprise_real: float | None   # Surprise(%) reportado
    entry_idx: int                # posicao no array de precos p/ entrada (close, t+1+)
    px: pd.Series                 # serie de precos do ticker (referencia)


def _build_events(tk: str, force: bool) -> list[Event]:
    px = _load_prices(tk, force=force)
    earn = _load_earnings(tk, force=force)
    dates = px.index
    events: list[Event] = []
    for ann_ts, row in earn.iterrows():
        ann_day = pd.Timestamp(ann_ts).normalize()
        # Anuncio (maioria after-close) conhecido a partir do PROXIMO pregao.
        # Entrada = primeiro pregao ESTRITAMENTE depois do dia do anuncio.
        pos = dates.searchsorted(ann_day, side="right")
        if pos >= len(dates):
            continue  # anuncio no futuro / sem pregao posterior em cache
        sr = row.get("Surprise(%)")
        sr = float(sr) if pd.notna(sr) else None
        events.append(Event(tk, ann_day, sr, int(pos), px))
    return events


def _event_gap(ev: Event) -> float | None:
    """Reacao de preco no DIA do anuncio (proxy de surpresa). Retorno do dia event.

    Usa close[event_day] vs close[event_day-1]. event_day = entry_idx-1 (o pregao do
    anuncio, se after-close) — mas como muitos anunciam after-close, a reacao real
    aparece no pregao seguinte (= entry day). Para nao usar o retorno que vamos
    capturar como sinal (look-ahead), o gap-proxy usa a reacao ATE entry_idx-1.
    """
    i = ev.entry_idx
    if i < 2:
        return None
    vals = ev.px.values
    prev = vals[i - 2]
    react = vals[i - 1]
    if prev <= 0:
        return None
    return float(react / prev - 1.0)


# ----------------------------------------------------------------------------- sim
def _trade_return(ev: Event, side: int, hold: int) -> float | None:
    """Retorno BRUTO de um trade: entra no close de entry_idx, sai hold pregoes depois."""
    i = ev.entry_idx
    j = i + hold
    vals = ev.px.values
    if j >= len(vals):
        return None
    entry = vals[i]
    exit_ = vals[j]
    if entry <= 0:
        return None
    raw = exit_ / entry - 1.0
    return side * raw


@dataclass
class ConfigResult:
    label: str
    signal: str
    hold: int
    cut: float
    n_trades: int
    # serie de retorno por trade (liquido, estressado), indexada por data de entrada
    net_by_date: pd.Series
    gross_sharpe: float


def _signal_value(ev: Event, signal: str) -> float | None:
    if signal == "surprise_real":
        return ev.surprise_real
    if signal == "gap_proxy":
        return _event_gap(ev)
    raise ValueError(signal)


def run_config(events: list[Event], signal: str, hold: int, cut_pct: float,
               cost_round_trip_bps: float, entry_dates_index: dict) -> ConfigResult:
    """Roda uma config sobre TODOS os eventos. Long/short conforme sinal vs corte.

    cut_pct: quantil de |sinal| abaixo do qual o evento e IGNORADO (so trada surpresas
    materiais). Long se sinal>0, short se sinal<0.
    """
    sigs = []
    valid = []
    for ev in events:
        v = _signal_value(ev, signal)
        if v is None or not np.isfinite(v):
            continue
        sigs.append(abs(v))
        valid.append((ev, v))
    if not valid:
        return ConfigResult(f"{signal}|h{hold}|c{cut_pct:.2f}", signal, hold, cut_pct,
                            0, pd.Series(dtype=float), 0.0)
    thr = float(np.quantile(np.asarray(sigs), cut_pct))
    rows_date = []
    rows_ret = []
    cost = cost_round_trip_bps / 1e4
    for ev, v in valid:
        if abs(v) < thr:
            continue
        side = 1 if v > 0 else -1
        gross = _trade_return(ev, side, hold)
        if gross is None:
            continue
        net = gross - cost  # round-trip ja
        entry_day = ev.px.index[ev.entry_idx]
        rows_date.append(entry_day)
        rows_ret.append(net)
    if not rows_ret:
        return ConfigResult(f"{signal}|h{hold}|c{cut_pct:.2f}", signal, hold, cut_pct,
                            0, pd.Series(dtype=float), 0.0)
    s = pd.Series(rows_ret, index=pd.DatetimeIndex(rows_date)).sort_index()
    gs = observed_sharpe(s.values)
    return ConfigResult(f"{signal}|h{hold}|c{cut_pct:.2f}", signal, hold, cut_pct,
                        len(s), s, gs)


# ---------------------------------------------------------- portfolio time series
def _to_monthly_portfolio(net_by_date: pd.Series, hold: int) -> pd.Series:
    """Agrega trades sobrepostos numa serie de retorno de PORTFOLIO mensal.

    Trades duram `hold` pregoes e se sobrepoem; para uma serie de retorno comparavel
    ao mercado, aloca cada trade igualmente e distribui seu retorno ao longo da janela
    de holding. Aqui usamos uma agregacao mensal simples: media dos retornos por trade
    cujo ENTRY cai no mes, escalada por participacao -> proxy de uma carteira que abre
    posicoes equal-weight conforme os eventos chegam. Conservador e comparavel ao SPY mensal.
    """
    if net_by_date.empty:
        return pd.Series(dtype=float)
    m = net_by_date.groupby(net_by_date.index.to_period("M")).mean()
    m.index = m.index.to_timestamp("M")
    return m


def _ann_from_pertrade(net_by_date: pd.Series, hold: int) -> float:
    """Sharpe ANUALIZADO a partir de retornos POR TRADE.

    Cada trade dura `hold` pregoes. periods/ano efetivos ~= 252/hold * (n_concorrentes).
    Para nao inflar com sobreposicao, anualizamos conservadoramente assumindo trades
    sequenciais nao-sobrepostos de tamanho hold: periods_per_year = 252/hold.
    """
    if net_by_date.size < 2:
        return 0.0
    ppy = max(EQUITY_PERIODS / hold, 1.0)
    return observed_sharpe(net_by_date.values) * float(np.sqrt(ppy))


# ----------------------------------------------------------------------------- run
def _spy_monthly(force: bool) -> pd.Series:
    spy = _load_prices(SPY, force=force)
    m = spy.resample("ME").last()
    return m.pct_change().dropna()


def _cagr_from_monthly(monthly: pd.Series) -> float:
    if monthly.empty:
        return 0.0
    growth = float((1.0 + monthly).prod())
    years = len(monthly) / 12.0
    if years <= 0 or growth <= 0:
        return 0.0
    return growth ** (1.0 / years) - 1.0


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def run(force: bool = False) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("TRIBUNAL PEAD — Post-Earnings-Announcement Drift (R&D isolado)")
    lines.append("=" * 78)

    # 1) coletar eventos de todo o universo
    all_events: list[Event] = []
    n_real = 0
    failed = []
    for tk in UNIVERSE:
        try:
            evs = _build_events(tk, force=force)
        except Exception as e:  # rede / dado faltando
            failed.append((tk, repr(e)))
            continue
        all_events.extend(evs)
        n_real += sum(1 for e in evs if e.surprise_real is not None)

    n_total = len(all_events)
    lines.append(f"universo: {len(UNIVERSE)} nomes | eventos totais: {n_total} | "
                 f"com Surprise(%) real: {n_real}")
    if failed:
        lines.append(f"falharam no fetch ({len(failed)}): "
                     + ", ".join(t for t, _ in failed[:10]))

    if n_total < 200:
        lines.append("DADOS INSUFICIENTES — abortando (eventos < 200).")
        REPORT.write_text("\n".join(lines))
        return "\n".join(lines)

    # 2) grade de configs (n_trials honesto)
    signals = ["surprise_real", "gap_proxy"]
    holds = [20, 30, 40, 60]
    cuts = [0.20, 0.33]
    n_trials = len(signals) * len(holds) * len(cuts)
    lines.append(f"n_trials (honesto, anti-snooping) = {n_trials} "
                 f"= sinais{signals} x hold{holds} x cut{cuts}")

    cost_stress = EQUITY_BASE.stressed(2.0).round_trip_bps  # bps round-trip
    cost_base = EQUITY_BASE.round_trip_bps
    lines.append(f"custo round-trip: base={cost_base:.0f}bps | "
                 f"ESTRESSADO(veredito)={cost_stress:.0f}bps (short ignora borrow)")

    results: list[ConfigResult] = []
    for sig in signals:
        for h in holds:
            for c in cuts:
                results.append(run_config(all_events, sig, h, c, cost_stress, {}))

    # 3) matriz p/ PBO (alinhar series por data de entrada, mensal)
    monthly_map: dict[str, pd.Series] = {}
    for r in results:
        monthly_map[r.label] = _to_monthly_portfolio(r.net_by_date, r.hold)
    mat_df = pd.DataFrame(monthly_map).dropna(how="all")
    mat_df = mat_df.dropna()  # so meses com todas as configs (para CSCV honesto)
    pbo = (probability_of_backtest_overfitting(mat_df.values)
           if mat_df.shape[0] >= 8 and mat_df.shape[1] >= 2 else float("nan"))

    # 4) SPY benchmark mensal alinhado
    try:
        spy_m = _spy_monthly(force=force)
    except Exception as e:
        spy_m = pd.Series(dtype=float)
        lines.append(f"SPY indisponivel: {e!r}")

    # 5) escolher campeao por sinal REAL (o sinal primario), reportar tudo
    lines.append("")
    lines.append(f"{'config':<26}{'nTr':>6}{'Shrp_an':>9}{'CAGR':>8}"
                 f"{'MaxDD':>8}{'corrSPY':>9}")
    lines.append("-" * 78)

    def corr_spy(monthly: pd.Series) -> float:
        if spy_m.empty or monthly.empty:
            return float("nan")
        j = pd.concat([monthly, spy_m], axis=1, join="inner").dropna()
        if j.shape[0] < 6:
            return float("nan")
        return float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])

    rep_rows = []
    for r in sorted(results, key=lambda x: -_ann_from_pertrade(x.net_by_date, x.hold)):
        m = monthly_map[r.label]
        sh = _ann_from_pertrade(r.net_by_date, r.hold)
        cagr = _cagr_from_monthly(m)
        mdd = max_drawdown((1.0 + m).cumprod().values) if not m.empty else 0.0
        cs = corr_spy(m)
        rep_rows.append((r, sh, cagr, mdd, cs))
        lines.append(f"{r.label:<26}{r.n_trades:>6}{sh:>9.2f}{_pct(cagr):>8}"
                     f"{_pct(mdd):>8}{cs:>9.2f}")

    # 6) VEREDITO: campeao do sinal REAL (primario), DSR honesto com n_trials total
    real_results = [r for r in results if r.signal == "surprise_real" and r.n_trades >= 30]
    if not real_results:
        lines.append("\nSinal REAL sem trades suficientes — usando gap_proxy declarado.")
        real_results = [r for r in results if r.n_trades >= 30]

    champ = max(real_results, key=lambda x: _ann_from_pertrade(x.net_by_date, x.hold))
    champ_m = monthly_map[champ.label]
    ppy = max(EQUITY_PERIODS / champ.hold, 1.0)

    verdict = evaluate_edge(
        champ.net_by_date.values,
        n_trials=n_trials,
        periods_per_year=int(round(ppy)),
        min_sharpe_annual=0.8,
        dsr_threshold=0.95,
    )

    # 7) OOS robustez: split temporal por data de entrada (1a metade vs 2a metade)
    s_sorted = champ.net_by_date.sort_index()
    mid = len(s_sorted) // 2
    is_sh = _ann_from_pertrade(s_sorted.iloc[:mid], champ.hold)
    oos_sh = _ann_from_pertrade(s_sorted.iloc[mid:], champ.hold)

    champ_cagr = _cagr_from_monthly(champ_m)
    champ_mdd = max_drawdown((1.0 + champ_m).cumprod().values) if not champ_m.empty else 0.0
    champ_corr = corr_spy(champ_m)
    spy_sharpe = (observed_sharpe(spy_m.values) * np.sqrt(12)) if not spy_m.empty else float("nan")
    spy_cagr = _cagr_from_monthly(spy_m) if not spy_m.empty else float("nan")

    lines.append("")
    lines.append("-" * 78)
    lines.append(f"CAMPEAO (sinal primario REAL): {champ.label}  n_trades={champ.n_trades}")
    lines.append(f"  Sharpe_an (estressado) = {verdict.sharpe_annual:.2f} "
                 f"(obstaculo snooping = {verdict.sr_benchmark_annual:.2f})")
    lines.append(f"  PSR = {verdict.psr:.3f} | DSR = {verdict.dsr:.3f} "
                 f"(barra 0.95) | n_obs={verdict.n_obs} n_trials={verdict.n_trials}")
    lines.append(f"  skew={verdict.skew:.2f} kurt={verdict.kurtosis:.2f} | "
                 f"PBO={pbo:.3f} (barra <0.5)")
    lines.append(f"  CAGR={_pct(champ_cagr)} MaxDD={_pct(champ_mdd)} corrSPY={champ_corr:.2f}")
    lines.append(f"  OOS split: Sharpe_an IS={is_sh:.2f} -> OOS={oos_sh:.2f}")
    lines.append(f"  Benchmark SPY: Sharpe_an={spy_sharpe:.2f} CAGR={_pct(spy_cagr)}")

    # criterio de aprovacao
    beats_bh = (not np.isnan(spy_sharpe)) and verdict.sharpe_annual > spy_sharpe
    diversifies = (not np.isnan(champ_corr)) and abs(champ_corr) < 0.3 and verdict.sharpe_annual >= 0.8
    robust_oos = (oos_sh > 0.3) and (is_sh > 0.3)
    robust_pbo = (not np.isnan(pbo)) and pbo < 0.5

    passed = bool(
        verdict.dsr >= 0.95
        and verdict.sharpe_annual >= 0.8
        and (beats_bh or diversifies)
        and robust_oos
        and robust_pbo
    )

    lines.append("")
    lines.append("CRITERIOS:")
    lines.append(f"  DSR>=0.95 ............ {verdict.dsr >= 0.95}  ({verdict.dsr:.3f})")
    lines.append(f"  Sharpe_an>=0.8 ...... {verdict.sharpe_annual >= 0.8}  ({verdict.sharpe_annual:.2f})")
    lines.append(f"  bate SPY ............ {beats_bh}  (estrat {verdict.sharpe_annual:.2f} vs SPY {spy_sharpe:.2f})")
    lines.append(f"  OU diversifica ...... {diversifies}  (|corr|={abs(champ_corr):.2f})")
    lines.append(f"  robusto OOS ......... {robust_oos}  (IS {is_sh:.2f} / OOS {oos_sh:.2f})")
    lines.append(f"  PBO<0.5 ............. {robust_pbo}  ({pbo:.3f})")
    lines.append("")
    lines.append(f"VEREDITO: {'PASSA' if passed else 'FALHA'}")
    lines.append("=" * 78)

    REPORT.write_text("\n".join(lines))

    # devolve dict-friendly via atributos no modulo (lido pelo runner se quiser)
    run.last = {  # type: ignore[attr-defined]
        "passed": passed,
        "dsr": float(verdict.dsr),
        "psr": float(verdict.psr),
        "pbo": float(pbo) if not np.isnan(pbo) else None,
        "sharpe": float(verdict.sharpe_annual),
        "cagr": float(champ_cagr),
        "maxdd": float(champ_mdd),
        "corr_spy": float(champ_corr) if not np.isnan(champ_corr) else None,
        "champ": champ.label,
        "n_trials": n_trials,
        "n_events": n_total,
        "n_real": n_real,
        "is_sharpe": float(is_sh),
        "oos_sharpe": float(oos_sh),
        "spy_sharpe": float(spy_sharpe) if not np.isnan(spy_sharpe) else None,
    }
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    print(run(force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
