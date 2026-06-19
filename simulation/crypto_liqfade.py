"""TRIBUNAL DE FADING DE DISLOCACAO EXTREMA (cripto perp) — H2: proxy de liquidacao.

Contexto (o que motiva isto): o tribunal de momentum (simulation.crypto_intraday, H1)
rodou sobre ~3,1M de candles 1min reais (data/binance_1m/, 6 moedas, 12 meses) e FALHOU
com edge bruto NEGATIVO e win rate 27-34%. Ou seja: cripto de curto prazo REVERTE, nao
segue. Mas inverter o sinal de forma GERAL nao paga — o bruto invertido (~+0.2 a +1.5 bps)
e menor que o custo round-trip (8-11 bps taker).

TESE H2: a reversao so paga a conta nos EVENTOS EXTREMOS (cascatas de liquidacao). Nesses
momentos o preco overshoota muito alem do fair value (forcado por liquidacoes em cadeia) e
reverte forte na sequencia. Se isso for verdade, o edge bruto/trade deve CRESCER conforme o
limiar fica mais extremo. Esse e o DIAGNOSTICO-CHAVE deste tribunal.

RESSALVA OBRIGATORIA — ISTO E UM PROXY:
  Nao usamos dado real de liquidacao da Binance (historico nao e gratis). Usamos a ASSINATURA
  documentada de uma cascata: pavio/retorno de 1-5 min muito extremo (z-score de vol alto E/OU
  cauda de percentil) acompanhado de PICO DE VOLUME (volume da barra >> media rolante = ordens
  forcadas). Pavio-grande+volume e assinatura conhecida, mas ENVIESADA: tambem captura noticias,
  fat-fingers e gaps de fim-de-semana. Trate os numeros como LIMITE SUPERIOR otimista da tese.

SINAL (sem look-ahead):
  - Janela curta w em {1, 3, 5} min: retorno acumulado ate o CLOSE da barra t.
  - Extremo definido por |z| >= limiar, com z = ret_w / vol_rolante (vol = std dos rets de 1min,
    janela VOL_WINDOW, escalada por sqrt(w)). Varre de MODERADO a MUITO EXTREMO: z em {3,4,6}.
  - Filtro de volume (com/sem): volume da barra t >= VOL_SPIKE_MULT * media rolante de volume.
    Assinatura de liquidacao forcada. Conta no n_trials (dobra a grade).

TRADE:
  - Entra CONTRA o movimento no OPEN da barra t+1 (compra apos queda extrema; short apos
    disparo extremo). Segura H em {5,10,15} min. Sai no OPEN de t+1+H. Sem sobreposicao.

CUSTO: taker 4bps/lado (8bps RT) e maker 2bps/lado (4bps RT) — reporta os dois. Funding
  ~1bp/8h se cruzar settlement (00/08/16 UTC). Reporta ret medio/trade BRUTO vs LIQUIDO,
  trades/dia (deve ser POUCO — so cauda), win rate.

n_trials honesto = w x limiar x H x [com/sem filtro de volume].

Uso:
    uv run python -m simulation.crypto_liqfade        # roda o tribunal (usa o cache de klines)
    (klines via simulation.crypto_intraday --download)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.metrics import max_drawdown
from simulation.statistics import (
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

# Reusa o loader de klines e as constantes/custos do tribunal H1 (NAO edita aquele arquivo).
from simulation.crypto_intraday import (
    FUNDING_BPS_PER_8H,
    FUNDING_HOURS,
    MAKER_BPS,
    MINUTES_PER_YEAR,
    MONTHS,
    TAKER_BPS,
    UNIVERSE,
    VOL_WINDOW,
    buy_and_hold_sharpe,
    load_symbol,
)

# --------------------------------------------------------------------------- #
# Grade de hipoteses H2 (conta TODA no n_trials)
# --------------------------------------------------------------------------- #
WINDOWS_W = (1, 3, 5)            # minutos da janela curta do sinal (o "pavio")
Z_THRESHOLDS = (3.0, 4.0, 6.0)   # limiar de extremo em z-score de vol: MODERADO -> MUITO EXTREMO
HOLDINGS_H = (5, 10, 15)         # minutos de holding apos o fade
VOL_FILTERS = (False, True)      # exigir ou nao pico de volume (assinatura de liquidacao)

VOL_SPIKE_MULT = 3.0             # volume da barra >= este multiplo da media rolante
VOLUME_WINDOW = 60               # janela (min) para a media de volume do filtro


@dataclass
class ConfigVerdict:
    w: int
    z_thr: float
    hold: int
    vol_filter: bool
    n_trades: int
    trades_per_day: float
    win_rate: float
    gross_mean_bps: float
    net_taker_mean_bps: float
    net_maker_mean_bps: float
    sharpe_taker_annual: float
    sharpe_maker_annual: float
    sharpe_gross_annual: float
    dsr_taker: float
    dsr_maker: float
    maxdd_taker: float
    net_taker_total: float
    rets_taker: np.ndarray


def _funding_crossings(entry_idx: int, exit_idx: int, hours: np.ndarray) -> int:
    """Settlements de funding (00/08/16 UTC) dentro do hold (entry, exit]."""
    if exit_idx <= entry_idx:
        return 0
    window = hours[entry_idx + 1: exit_idx + 1]
    if window.size == 0:
        return 0
    return int(np.isin(window, FUNDING_HOURS).sum())


def run_h2_config(
    df: pd.DataFrame, w: int, z_thr: float, hold: int, vol_filter: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Roda UMA config (w, z_thr, hold, vol_filter) de H2 sobre um simbolo.

    Sinal no CLOSE de t: ret acumulado dos ultimos w min, normalizado por vol -> z.
    EXTREMO: |z| >= z_thr (e, se vol_filter, volume[t] >= VOL_SPIKE_MULT*media).
    FADE: entra CONTRA o sinal no OPEN de t+1 (side = -sign(z)), sai no OPEN de t+1+hold.
    Sem look-ahead. Posicoes nao sobrepostas (entra na proxima barra livre apos a saida).

    Retorna (gross[], net_taker[], net_maker[], n_bars).
    """
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    hours = df.index.hour.to_numpy()
    n = len(c)
    if n < VOL_WINDOW + w + hold + 5:
        return np.array([]), np.array([]), np.array([]), n

    logc = np.log(c)
    r1 = np.diff(logc, prepend=logc[0])               # r1[t] = log(c[t]/c[t-1])
    ret_w = pd.Series(r1).rolling(w).sum().to_numpy()  # retorno acumulado dos ultimos w min
    vol_1m = pd.Series(r1).rolling(VOL_WINDOW).std(ddof=0).to_numpy()
    vol_w = vol_1m * np.sqrt(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(vol_w > 0, ret_w / vol_w, 0.0)

    # filtro de volume: pico vs media rolante (assinatura de liquidacao forcada)
    if vol_filter:
        vbar = pd.Series(vol).rolling(VOLUME_WINDOW).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            vspike = np.where(vbar > 0, vol / vbar, 0.0)
    else:
        vspike = None

    gross: list[float] = []
    net_t: list[float] = []
    net_m: list[float] = []

    taker_rt = 2.0 * TAKER_BPS / 1e4
    maker_rt = 2.0 * MAKER_BPS / 1e4
    fund_unit = FUNDING_BPS_PER_8H / 1e4

    t = max(VOL_WINDOW, VOLUME_WINDOW) + w  # 1a barra com sinal valido
    while t + 1 + hold < n:
        zt = z[t]
        if not np.isfinite(zt) or abs(zt) < z_thr:
            t += 1
            continue
        if vol_filter and (vspike is None or not np.isfinite(vspike[t]) or vspike[t] < VOL_SPIKE_MULT):
            t += 1
            continue
        # FADE: posicao CONTRA o movimento. zt > 0 (disparo) -> short; zt < 0 (queda) -> long.
        side = -1.0 if zt > 0 else 1.0
        entry = o[t + 1]
        exit_px = o[t + 1 + hold]
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_px) and exit_px > 0):
            t += 1
            continue
        g = side * (exit_px / entry - 1.0)
        n_fund = _funding_crossings(t + 1, t + 1 + hold, hours)
        fund_cost = n_fund * fund_unit
        gross.append(g)
        net_t.append(g - taker_rt - fund_cost)
        net_m.append(g - maker_rt - fund_cost)
        t = t + 1 + hold  # nao sobrepoe
    return np.asarray(gross), np.asarray(net_t), np.asarray(net_m), n


def evaluate_config(
    symbols_data: dict[str, pd.DataFrame],
    w: int,
    z_thr: float,
    hold: int,
    vol_filter: bool,
    n_trials: int,
) -> ConfigVerdict:
    """Roda a config em TODOS os simbolos, junta os trades e julga."""
    all_g, all_nt, all_nm = [], [], []
    total_bars = 0
    for df in symbols_data.values():
        g, nt, nm, nbars = run_h2_config(df, w, z_thr, hold, vol_filter)
        all_g.append(g)
        all_nt.append(nt)
        all_nm.append(nm)
        total_bars += nbars
    g = np.concatenate(all_g) if all_g else np.array([])
    nt = np.concatenate(all_nt) if all_nt else np.array([])
    nm = np.concatenate(all_nm) if all_nm else np.array([])

    n = int(g.size)
    days = max(total_bars / (24 * 60), 1e-9)
    periods_per_year_trade = MINUTES_PER_YEAR / hold

    if n < 2:
        return ConfigVerdict(w, z_thr, hold, vol_filter, n, 0.0, 0.0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nt)

    win_rate = float((nt > 0).mean())
    sh_t = observed_sharpe(nt, periods_per_year=periods_per_year_trade)
    sh_m = observed_sharpe(nm, periods_per_year=periods_per_year_trade)
    sh_g = observed_sharpe(g, periods_per_year=periods_per_year_trade)
    v_t = evaluate_edge(nt, n_trials=n_trials, periods_per_year=periods_per_year_trade)
    v_m = evaluate_edge(nm, n_trials=n_trials, periods_per_year=periods_per_year_trade)
    eq_t = np.cumprod(1.0 + nt)
    return ConfigVerdict(
        w=w, z_thr=z_thr, hold=hold, vol_filter=vol_filter,
        n_trades=n,
        trades_per_day=n / days,
        win_rate=win_rate,
        gross_mean_bps=float(g.mean() * 1e4),
        net_taker_mean_bps=float(nt.mean() * 1e4),
        net_maker_mean_bps=float(nm.mean() * 1e4),
        sharpe_taker_annual=sh_t,
        sharpe_maker_annual=sh_m,
        sharpe_gross_annual=sh_g,
        dsr_taker=v_t.dsr,
        dsr_maker=v_m.dsr,
        maxdd_taker=max_drawdown(eq_t),
        net_taker_total=float(eq_t[-1] - 1.0),
        rets_taker=nt,
    )


def n_trials_total() -> int:
    return len(WINDOWS_W) * len(Z_THRESHOLDS) * len(HOLDINGS_H) * len(VOL_FILTERS)


def run_tribunal(
    symbols_data: dict[str, pd.DataFrame],
) -> tuple[list[ConfigVerdict], float, float, dict]:
    """Roda toda a grade H2. Retorna (verdicts, bh_sharpe_medio, bh_ret_medio, pbo_info)."""
    n_trials = n_trials_total()
    verdicts: list[ConfigVerdict] = []
    for w in WINDOWS_W:
        for z_thr in Z_THRESHOLDS:
            for hold in HOLDINGS_H:
                for vf in VOL_FILTERS:
                    verdicts.append(evaluate_config(symbols_data, w, z_thr, hold, vf, n_trials))

    bh_sh, bh_ret = [], []
    for df in symbols_data.values():
        s, t = buy_and_hold_sharpe(df)
        bh_sh.append(s)
        bh_ret.append(t)
    bh_sharpe = float(np.mean(bh_sh)) if bh_sh else 0.0
    bh_total = float(np.mean(bh_ret)) if bh_ret else 0.0

    pbo_info = {"pbo": None, "note": "amostra insuficiente"}
    series = [v.rets_taker for v in verdicts if v.rets_taker.size >= 20]
    if len(series) >= 2:
        m = min(s.size for s in series)
        if m >= 20:
            mat = np.column_stack([s[:m] for s in series])
            pbo = probability_of_backtest_overfitting(mat, n_splits=10)
            pbo_info = {"pbo": pbo, "note": f"{len(series)} configs x {m} trades"}
    return verdicts, bh_sharpe, bh_total, pbo_info


# --------------------------------------------------------------------------- #
# Diagnostico-chave: edge BRUTO vs limiar
# --------------------------------------------------------------------------- #
def edge_vs_threshold_table(verdicts: list[ConfigVerdict]) -> list[str]:
    """A TABELA mais importante: o edge BRUTO/trade CRESCE conforme o limiar fica extremo?

    Para cada limiar z, agrega os trades de TODAS as configs (w x H x vol_filter) naquele z e
    reporta o edge bruto medio ponderado por n. Se cresce monotonicamente com z -> assinatura
    de reversao pos-liquidacao real. Se nao -> ruido.
    """
    lines = [
        "DIAGNOSTICO-CHAVE — edge BRUTO/trade vs LIMIAR (a reversao paga mais no extremo?):",
        "  (agregado sobre todas as combinacoes w x H; 'sem/com vol' = filtro de pico de volume)",
        "",
        f"  {'z_thr':>6} {'vol_filt':>9} | {'n_trades':>9} {'gross_bps':>10} {'netT_bps':>9} {'win%':>6}",
        "  " + "-" * 60,
    ]
    rows = []
    for z_thr in Z_THRESHOLDS:
        for vf in VOL_FILTERS:
            gross_w, nt_w, n_tot, wins = 0.0, 0.0, 0, 0.0
            for v in verdicts:
                if v.z_thr == z_thr and v.vol_filter == vf and v.n_trades >= 1:
                    gross_w += v.gross_mean_bps * v.n_trades
                    nt_w += v.net_taker_mean_bps * v.n_trades
                    wins += v.win_rate * v.n_trades
                    n_tot += v.n_trades
            if n_tot > 0:
                rows.append((z_thr, vf, n_tot, gross_w / n_tot, nt_w / n_tot, wins / n_tot))
    for z_thr, vf, n_tot, gbps, ntbps, win in rows:
        tag = "com vol" if vf else "sem vol"
        lines.append(f"  {z_thr:>6.1f} {tag:>9} | {n_tot:>9} {gbps:>10.2f} {ntbps:>9.2f} {win * 100:>6.1f}")
    lines.append("")

    # leitura automatica da monotonicidade (por trilha de filtro de volume)
    for vf in VOL_FILTERS:
        track = [(z, g) for (z, v, n, g, nt, w) in rows if v == vf]
        track.sort(key=lambda x: x[0])
        tag = "COM filtro de volume" if vf else "SEM filtro de volume"
        if len(track) >= 2:
            gs = [g for _, g in track]
            growing = all(gs[i + 1] >= gs[i] for i in range(len(gs) - 1))
            seq = " -> ".join(f"{g:+.2f}" for g in gs)
            verdict = ("CRESCE (assinatura de reversao pos-liquidacao)" if growing
                       else "NAO cresce monotonicamente (sinal fraco/ruido)")
            lines.append(f"  {tag}: edge bruto por z ({'/'.join(f'{z:.0f}' for z, _ in track)}) = "
                         f"{seq} bps  ->  {verdict}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #
def _kill_criterion_header() -> list[str]:
    nt = n_trials_total()
    return [
        "=" * 100,
        "TRIBUNAL DE FADING DE DISLOCACAO EXTREMA (cripto perp) — H2: PROXY DE LIQUIDACAO",
        "=" * 100,
        "",
        "KILL-CRITERION (pre-registrado, ANTES dos numeros):",
        "  PASSA so se (TAKER): DSR >= 0.95 E Sharpe_liq_anual >= 1.0 E ret_liq > 0 E",
        "  supera buy-and-hold do universo no periodo. Senao FALHA.",
        "  (So-maker => PENDENTE-DE-ORDER-BOOK, nao PASSA: maker assume fill passivo SEM",
        "   adverse selection, otimista — exige bookTicker p/ confirmar.)",
        "",
        "RESSALVA OBRIGATORIA: isto e um PROXY de liquidacao (pavio extremo + pico de volume),",
        "  NAO dado real de liquidacao da Binance (historico nao e gratis). Pavio-grande+volume",
        "  e assinatura documentada POREM ENVIESADA (pega tambem noticias/fat-finger/gaps).",
        "  Os numeros sao LIMITE SUPERIOR otimista da tese, nao a tese provada.",
        "",
        f"Custos: TAKER 4bps/lado (8bps RT) | MAKER 2bps/lado (4bps RT, otimista) |",
        f"  funding {FUNDING_BPS_PER_8H:.0f}bp/8h aprox em 00/08/16 UTC.",
        f"n_trials (anti-snooping) = {len(WINDOWS_W)}w x {len(Z_THRESHOLDS)}z x {len(HOLDINGS_H)}H "
        f"x {len(VOL_FILTERS)}[vol] = {nt} configs.",
        "Fade: entra CONTRA o movimento no OPEN de t+1, segura H, sai. Long+short, sem sobreposicao.",
        "",
    ]


def render_report(
    verdicts: list[ConfigVerdict],
    bh_sharpe: float,
    bh_total: float,
    pbo_info: dict,
    symbols_data: dict[str, pd.DataFrame],
) -> str:
    lines = _kill_criterion_header()
    lines.append(f"Dados: {len(symbols_data)} simbolos, "
                 f"{sum(len(d) for d in symbols_data.values()):,} barras 1min totais.")
    lines.append(f"Buy-and-hold medio do universo: Sharpe_anual={bh_sharpe:.2f}  "
                 f"ret_total_periodo={bh_total * 100:+.1f}%")
    lines.append("")

    # DIAGNOSTICO-CHAVE primeiro: edge bruto vs limiar
    lines.extend(edge_vs_threshold_table(verdicts))

    # tabela completa por config
    hdr = (f"{'w':>3} {'z':>4} {'H':>3} {'vol':>4} | {'trades':>7} {'t/dia':>6} {'win%':>5} | "
           f"{'gross_bps':>9} {'netT_bps':>8} {'netM_bps':>8} | "
           f"{'ShT':>6} {'ShM':>6} | {'DSR_T':>6} {'DSR_M':>6} | {'MddT%':>6} | veredito")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for v in sorted(verdicts, key=lambda x: x.sharpe_taker_annual, reverse=True):
        vtag = "sim" if v.vol_filter else "nao"
        if v.n_trades < 2:
            lines.append(f"{v.w:>3} {v.z_thr:>4.1f} {v.hold:>3} {vtag:>4} |  "
                         f"amostra insuficiente ({v.n_trades})")
            continue
        passes_taker = (
            v.dsr_taker >= 0.95
            and v.sharpe_taker_annual >= 1.0
            and v.net_taker_mean_bps > 0
            and v.sharpe_taker_annual > bh_sharpe
        )
        passes_maker_only = (
            not passes_taker
            and v.dsr_maker >= 0.95
            and v.sharpe_maker_annual >= 1.0
            and v.net_maker_mean_bps > 0
        )
        flag = "PASSA" if passes_taker else ("PEND-OB" if passes_maker_only else "FALHA")
        lines.append(
            f"{v.w:>3} {v.z_thr:>4.1f} {v.hold:>3} {vtag:>4} | {v.n_trades:>7} "
            f"{v.trades_per_day:>6.2f} {v.win_rate * 100:>5.1f} | {v.gross_mean_bps:>9.2f} "
            f"{v.net_taker_mean_bps:>8.2f} {v.net_maker_mean_bps:>8.2f} | "
            f"{v.sharpe_taker_annual:>6.2f} {v.sharpe_maker_annual:>6.2f} | "
            f"{v.dsr_taker:>6.3f} {v.dsr_maker:>6.3f} | {v.maxdd_taker * 100:>6.1f} | {flag}"
        )

    lines.append("")
    if pbo_info["pbo"] is not None:
        lines.append(f"PBO (overfitting na selecao de config, net taker): {pbo_info['pbo']:.2f}  "
                     f"({pbo_info['note']})  [<0.5 = aceitavel]")
    else:
        lines.append(f"PBO: {pbo_info['note']}")
    lines.append("")

    # --- VEREDITO ---
    valid = [v for v in verdicts if v.n_trades >= 2]
    any_pass_taker = any(
        v.dsr_taker >= 0.95 and v.sharpe_taker_annual >= 1.0
        and v.net_taker_mean_bps > 0 and v.sharpe_taker_annual > bh_sharpe
        for v in valid
    )
    any_pass_maker = any(
        v.dsr_maker >= 0.95 and v.sharpe_maker_annual >= 1.0 and v.net_maker_mean_bps > 0
        for v in valid
    )
    lines.append("=" * 100)
    lines.append("VEREDITO H2 — FADING DE DISLOCACAO EXTREMA:")
    lines.append("=" * 100)
    if any_pass_taker:
        best = max(
            (v for v in valid if v.dsr_taker >= 0.95 and v.sharpe_taker_annual >= 1.0
             and v.net_taker_mean_bps > 0 and v.sharpe_taker_annual > bh_sharpe),
            key=lambda x: x.sharpe_taker_annual,
        )
        lines.append("")
        lines.append("  >>> PASSA (TAKER) <<<  ao menos uma config supera o kill-criterion liquido de taker.")
        lines.append(f"  Melhor: w={best.w} z={best.z_thr} H={best.hold} vol_filter={best.vol_filter} -> "
                     f"Sharpe_T={best.sharpe_taker_annual:.2f} DSR_T={best.dsr_taker:.3f} "
                     f"netT={best.net_taker_mean_bps:.2f}bps t/dia={best.trades_per_day:.2f}")
        lines.append("  LEMBRE DA RESSALVA: e proxy de liquidacao (vies otimista). Confirmar com")
        lines.append("  walk-forward OOS, custo estressado e dado real de liquidacao/funding antes de capital.")
    elif any_pass_maker:
        lines.append("")
        lines.append("  >>> PENDENTE-DE-ORDER-BOOK <<<  passa SO no maker (execucao passiva idealizada).")
        lines.append("  NAO e PASSA: num evento de liquidacao o fill passivo sofre adverse selection severa.")
        lines.append("  Precisa de bookTicker/dado de liquidacao real. E ainda assim e PROXY.")
    else:
        lines.append("")
        lines.append("  ##############################################################################")
        lines.append("  #                                                                            #")
        lines.append("  #   FALHA                                                                    #")
        lines.append("  #                                                                            #")
        lines.append("  #   Fading de dislocacao extrema NAO sobrevive ao custo real de taker.       #")
        lines.append("  #   Nenhuma config bate DSR>=0.95 + Sharpe>=1.0 + ret>0 + > buy-and-hold.     #")
        lines.append("  #   ARQUIVADO. A reversao no extremo nao paga 8bps de round-trip.            #")
        lines.append("  #                                                                            #")
        lines.append("  ##############################################################################")
        best_g = max(valid, key=lambda x: x.gross_mean_bps) if valid else None
        if best_g is not None:
            lines.append("")
            lines.append(f"  Diagnostico (config de maior edge BRUTO: w={best_g.w} z={best_g.z_thr} "
                         f"H={best_g.hold} vol_filter={best_g.vol_filter}):")
            lines.append(f"    gross={best_g.gross_mean_bps:.2f}bps  taker={best_g.net_taker_mean_bps:.2f}bps  "
                         f"maker={best_g.net_maker_mean_bps:.2f}bps  (RT taker=8bps, maker=4bps)")
            lines.append(f"    Sharpe: bruto={best_g.sharpe_gross_annual:.2f} -> "
                         f"taker={best_g.sharpe_taker_annual:.2f} -> maker={best_g.sharpe_maker_annual:.2f}")
            lines.append("    -> ver a tabela edge_bruto vs limiar acima para julgar se a tese (cresce no")
            lines.append("       extremo) se sustenta ou se mesmo crescendo nao cobre o custo de cruzar.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _no_data_report() -> str:
    return (
        "\n".join(_kill_criterion_header())
        + "\n"
        + "REDE BLOQUEADA / SEM DADOS: harness pronto, faltam os klines.\n"
        + "Cache esperado: data/binance_1m/<SYM>/<SYM>-1m-<YYYY-MM>.csv.gz\n"
        + f"Universo: {', '.join(UNIVERSE)}\n"
        + f"Janela: {MONTHS[0]} .. {MONTHS[-1]} (12 meses)\n\n"
        + "Baixe os klines com o tribunal H1 (mesmo cache):\n"
        + "    uv run python -m simulation.crypto_intraday --download\n"
        + "depois:\n"
        + "    uv run python -m simulation.crypto_liqfade\n"
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tribunal de fading de dislocacao extrema (cripto perp) — H2 proxy de liquidacao"
    )
    parser.add_argument("--report-file", default="data/crypto_liqfade_verdict.txt")
    parser.add_argument("--symbols", nargs="*", default=UNIVERSE)
    parser.add_argument("--months", nargs="*", default=MONTHS)
    args = parser.parse_args(argv)

    symbols_data: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        df = load_symbol(sym, args.months)
        if df is not None and len(df) > VOL_WINDOW + 100:
            symbols_data[sym] = df

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not symbols_data:
        text = _no_data_report()
        print(text)
        out.write_text(text, encoding="utf-8")
        print(f"Relatorio (SEM DADOS) salvo em {out}")
        return 0

    print(f"Carregados {len(symbols_data)} simbolos. Rodando tribunal H2 (fading de extremo)...")
    verdicts, bh_sharpe, bh_total, pbo_info = run_tribunal(symbols_data)
    text = render_report(verdicts, bh_sharpe, bh_total, pbo_info, symbols_data)
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"Relatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
