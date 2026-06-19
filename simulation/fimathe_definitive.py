"""TESTE DEFINITIVO DA FIMATHE INTRADAY — o Sharpe 1.50 do ouro e real ou artefato?

CONTEXTO. O tribunal anterior (simulation/fimathe_intraday.py, 3 anos 2023-2026,
XAUUSD) achou que a MELHOR config (take2/base) tinha Sharpe_liq ANUAL 1.50 — ACIMA
do buy&hold do ouro (1.25). No RISCO-AJUSTADO a FIMATHE empatava/ganhava: a critica
do usuario de que "retorno bruto e regua injusta" esta CORRETA. MAS reprovou por dois
motivos que ESTE teste ataca de frente:
  (a) DSR=0.21 << 0.95  -> num universo de 12 trials, esse Sharpe NAO se distingue
      do melhor que a SORTE produziria;
  (b) FRAGILIDADE de ancora: ancorar o Canal de Abertura na ABERTURA REAL do ouro
      (rollover ~22h UTC), em vez da meia-noite UTC arbitraria, derrubava o Sharpe.

OS 3 PILARES (cada um aperta um parafuso diferente do julgamento):
  1. MAIS DADOS. Baixa XAUUSD M15 da Dukascopy o MAXIMO de historico (~2004 -> 2026,
     ~22 anos vs os 3 de antes). Mais anos => mais observacoes => o obstaculo do DSR
     (E[max] de N trials) fica MAIS facil de bater SE o edge for real e persistente;
     se o edge some num historico longo, era sorte de um regime.
  2. ANCORA DE SESSAO REAL. O CA (4 primeiras velas M15) deve nascer da ABERTURA REAL
     do ouro/FX, NAO de 00:00 UTC. Metais/FX rolam ~17:00 ET (~21-22h UTC); a semana
     abre domingo ~22h UTC. Varre a ancora como PARAMETRO (21/22/23 UTC + 00 UTC). Se
     o edge so vive numa ancora arbitraria, e ruido; se sobrevive em todas, e sinal.
  3. WALK-FORWARD OOS. Divide o historico longo em sub-periodos contiguos e mostra se
     o Sharpe/DSR se SUSTENTA fora da janela onde o setup "parecia" funcionar.

JULGAMENTO (regua = RISCO-AJUSTADO, nao retorno bruto — incorpora a critica valida):
  RESSUSCITA (FIMATHE VIVE) so se TODOS:
    * DSR >= 0.95 NO HISTORICO LONGO (a melhor ancora/take), E
    * Sharpe_liq robusto NA ANCORA REAL (~22h), nao so na 00:00 UTC, E
    * estavel OUT-OF-SAMPLE (Sharpe nao desaba / nao troca de sinal entre sub-periodos).
  Se sobreviver e Sharpe_liq > buy&hold do ouro, registra-se que DOMINA via alavancagem
  (mesma logica de beta: risco-ajustado superior pode ser alavancado ao mesmo risco).
  FALHA DEFINITIVA se DSR continua baixo OU o Sharpe depende da ancora OU morre OOS.

REUTILIZACAO (nao reescreve o motor nem o harness):
  - fimathe.canal_abertura.CanalAbertura / CanalAberturaParams (motor mecanico, com
    session_break_hour ja parametrizavel -> a ANCORA e nativa).
  - simulation.fimathe_intraday._simulate_asset / _buy_hold_daily_returns (fills
    gap-aware, custo so-spread, serie por sessao) — IMPORTADOS, nao copiados.
  - simulation.statistics (observed_sharpe / evaluate_edge / PBO).
  - data.fimathe_intraday_data.spread_price / pip_size (custo do ouro).
  Adiciona APENAS: loader de historico profundo (estende _fetch_blocks) + os 3 pilares.

Uso:
    uv run python -m simulation.fimathe_definitive            # usa/baixa cache profundo
    uv run python -m simulation.fimathe_definitive --start-year 2004
    uv run python -m simulation.fimathe_definitive --report-file data/fimathe_definitive_verdict.txt
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- REUTILIZACAO (importa, nao reescreve) -----------------------------------
from data.fimathe_intraday_data import (
    INSTRUMENTS,
    _normalize,            # noqa: PLC2701 — normalizacao do mesmo loader (reuso honesto)
    pip_size,
    spread_price,
)
from fimathe.canal_abertura import CanalAbertura, CanalAberturaParams
from simulation.fimathe_intraday import (
    _buy_hold_daily_returns,
    _simulate_asset,
)
from simulation.statistics import (
    EQUITY_PERIODS,
    evaluate_edge,
    observed_sharpe,
)

logger = logging.getLogger("simulation.fimathe_definitive")

PERIODS_PER_YEAR = EQUITY_PERIODS  # 252 dias de pregao (forex/XAU)
INSTRUMENT = "XAUUSD"              # a tese central do usuario: 99% das operacoes dele

# Cache do historico PROFUNDO (separado do cache de 3 anos do tribunal antigo, p/ nao
# sobrescreve-lo: este e o dataset longo do teste definitivo).
DEEP_CACHE = Path("data/fimathe_intraday_cache/XAUUSD_M15_deep.csv")

# As ancoras de sessao a varrer (hora UTC que INICIA o "dia de pregao" do CA):
#   22 = rollover de metais/FX (~17:00 ET, EST) — a ABERTURA REAL primaria
#   21 = rollover no horario de verao (~17:00 EDT)
#   23 = folga de fuso / ancora alternativa
#   00 = a meia-noite UTC ARBITRARIA do tribunal antigo (o "control")
# Se o edge so existe na 00 UTC e some em 21/22/23, era artefato da ancora.
SESSION_ANCHORS: tuple[int | None, ...] = (22, 21, 23, 0)

# Grade de gestao (take 1 vs 2 niveis): e a unica escolha de "trial" do motor.
TAKE_LEVELS = (1, 2)


# =============================================================================
# PILAR 1 — historico profundo (estende o loader; nao reescreve a logica)
# =============================================================================
def _fetch_deep(start: datetime, end: datetime) -> pd.DataFrame:
    """Baixa XAUUSD M15 [start, end) em blocos mensais, resiliente a falha de bloco.

    Mesma estrategia de data.fimathe_intraday_data._fetch_blocks (1 mes/bloco, retry-
    soft, dedup), mas sobre um intervalo de MUITOS anos. Loga progresso para um
    download longo. Dukascopy entrega o que existe; blocos vazios/falhos sao logados
    e pulados (a profundidade REAL e reportada no veredito).
    """
    import dukascopy_python as d
    from dukascopy_python import instruments as ins

    instrument = getattr(ins, INSTRUMENTS[INSTRUMENT]["duka"])
    frames: list[pd.DataFrame] = []
    cur = start
    n_ok = n_fail = 0
    total_months = max(1, (end.year - start.year) * 12 + (end.month - start.month))
    done = 0
    while cur < end:
        # avanca ~1 mes (dia 1 -> dia 1) p/ blocos alinhados; trunca no fim
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        nxt = min(nxt, end)
        try:
            df = d.fetch(instrument, d.INTERVAL_MIN_15, d.OFFER_SIDE_BID, cur, nxt)
            if df is not None and not df.empty:
                frames.append(df)
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:  # noqa: BLE001 — rede/duka; loga e segue
            logger.warning("Bloco [%s..%s] falhou: %s", cur.date(), nxt.date(), exc)
            n_fail += 1
        done += 1
        if done % 12 == 0:
            logger.info("  ... %d/%d meses (%s) | blocos OK=%d vazio/falha=%d",
                        done, total_months, cur.date(), n_ok, n_fail)
        cur = nxt
    logger.info("Fetch profundo XAUUSD: %d blocos OK, %d vazios/falha", n_ok, n_fail)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def load_deep_xauusd(
    *, start_year: int = 2004, end: datetime | None = None, force: bool = False
) -> pd.DataFrame:
    """OHLC M15 PROFUNDO de XAUUSD (cache em DEEP_CACHE). Baixa via Dukascopy se ausente.

    Devolve DataFrame possivelmente vazio se a rede falhar por completo (o teste trata
    como DADOS PENDENTES — nada inventado).
    """
    DEEP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DEEP_CACHE.exists() and not force:
        df = pd.read_csv(DEEP_CACHE, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "timestamp"
        return df[~df.index.isna()]

    end = end or datetime.now(timezone.utc)
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    logger.info("Baixando XAUUSD M15 profundo %s..%s (pode demorar)...",
                start.date(), end.date())
    raw = _fetch_deep(start, end)
    df = _normalize(raw)
    if not df.empty:
        df.to_csv(DEEP_CACHE)
        logger.info("XAUUSD M15 profundo: %d barras (%s..%s) -> %s",
                    len(df), df.index[0].date(), df.index[-1].date(), DEEP_CACHE)
    return df


# =============================================================================
# Avaliacao de UMA serie por sessao (anti-overfitting), reutilizando o tribunal
# =============================================================================
@dataclass
class TrialResult:
    anchor: int | None
    take: int
    n_sessions: int
    n_trades: int
    win_rate: float
    avg_net: float
    sharpe_net_annual: float
    total_net: float
    dsr: float            # preenchido depois (precisa do universo de trials)
    ret_net: np.ndarray   # serie por sessao (para PBO/OOS)


def _run_trial(
    df: pd.DataFrame, *, anchor: int | None, take: int, full_spread: float
) -> TrialResult:
    """Roda o motor+harness para (ancora, take) e devolve metricas + serie por sessao.

    A ANCORA entra via CanalAberturaParams(session_break_hour) — nativo do motor.
    Sem skip_friday aqui (o foco e ancora x take x sub-periodo; manter a grade enxuta
    mantem o n_trials honesto e o DSR comparavel ao tribunal de 3 anos).
    """
    params = CanalAberturaParams(session_break_hour=anchor)
    eng = CanalAbertura(params)
    trades, ret_net, _ret_gross = _simulate_asset(
        df, eng, take_level=take, full_spread=full_spread, skip_friday=False
    )
    n_sessions = int(ret_net.size)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.ret_net > 0)
    win_rate = wins / n_trades if n_trades else 0.0
    avg_net = float(np.mean([t.ret_net for t in trades])) if trades else 0.0
    equity = np.cumprod(1.0 + ret_net)
    total_net = float(equity[-1] - 1.0) if equity.size else 0.0
    return TrialResult(
        anchor=anchor,
        take=take,
        n_sessions=n_sessions,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_net=avg_net,
        sharpe_net_annual=observed_sharpe(ret_net, periods_per_year=PERIODS_PER_YEAR),
        total_net=total_net,
        dsr=float("nan"),
        ret_net=ret_net,
    )


def _label(anchor: int | None) -> str:
    return "00 UTC (control)" if anchor in (None, 0) else f"{anchor:02d} UTC"


# =============================================================================
# Construcao do relatorio (os 5 entregaveis + veredito)
# =============================================================================
def _build_report(df: pd.DataFrame, *, start_year: int) -> str:
    full_spread = spread_price(INSTRUMENT)
    pip = pip_size(INSTRUMENT)
    span_days = (df.index[-1] - df.index[0]).days
    span_years = span_days / 365.25

    L: list[str] = []
    L.append("=" * 100)
    L.append("TESTE DEFINITIVO DA FIMATHE INTRADAY — o Sharpe 1.50 do ouro e REAL ou ARTEFATO?")
    L.append("=" * 100)
    L.append("")
    L.append("REGUA = RISCO-AJUSTADO (Sharpe), nao retorno bruto. RESSUSCITA so se:")
    L.append("  DSR>=0.95 no historico LONGO  E  Sharpe_liq robusto na ANCORA REAL (~22h, nao so 00 UTC)")
    L.append("  E  estavel OUT-OF-SAMPLE por sub-periodo. Senao -> FALHA DEFINITIVA.")
    L.append("Custo = SO spread (intraday, sem swap); fills gap-aware; serie = retorno por sessao.")
    L.append("Motor: fimathe/canal_abertura.py (reuso). Harness: simulation/fimathe_intraday.py (reuso).")
    L.append("")

    # ---- PILAR 1: profundidade de dados --------------------------------------
    L.append("-" * 100)
    L.append("PILAR 1 — PROFUNDIDADE DE DADOS (mais anos apertam o DSR)")
    L.append("-" * 100)
    L.append(
        f"  XAUUSD M15 (Dukascopy, BID): {len(df):,} barras  "
        f"{df.index[0].date()}..{df.index[-1].date()}  (~{span_years:.1f} anos)"
    )
    L.append(
        f"  vs tribunal anterior: ~3.0 anos (2023-2026).  Ganho: ~{span_years/3.0:.1f}x mais historico."
    )
    L.append(f"  spread modelado: {full_spread:.5f} (~{full_spread/pip:.1f} pip)")
    L.append("")

    # ---- buy&hold do ouro no historico longo (benchmark de alpha) ------------
    eng_bh = CanalAbertura(CanalAberturaParams(session_break_hour=22))
    bh = _buy_hold_daily_returns(df, eng_bh)
    bh_sharpe = observed_sharpe(bh, periods_per_year=PERIODS_PER_YEAR)
    bh_total = float(np.prod(1.0 + bh) - 1.0) if bh.size else 0.0
    L.append(
        f"  BUY&HOLD do ouro (historico longo): Sharpe_anual={bh_sharpe:.2f}  "
        f"ret_total={bh_total*100:+.0f}%"
    )
    L.append("")

    # ---- roda TODAS as trials (ancora x take) sobre o historico longo --------
    trials: dict[tuple[int | None, int], TrialResult] = {}
    for anchor in SESSION_ANCHORS:
        for take in TAKE_LEVELS:
            trials[(anchor, take)] = _run_trial(
                df, anchor=anchor, take=take, full_spread=full_spread
            )

    # n_trials HONESTO = ancoras x takes (a busca REAL feita aqui)
    n_trials = len(SESSION_ANCHORS) * len(TAKE_LEVELS)
    trial_sharpes_period = [observed_sharpe(t.ret_net) for t in trials.values()]
    for key, t in trials.items():
        v = evaluate_edge(
            t.ret_net, n_trials=n_trials, trial_sharpes=trial_sharpes_period,
            periods_per_year=PERIODS_PER_YEAR, min_sharpe_annual=1.0,
        )
        t.dsr = v.dsr

    # ---- PILAR 2: Sharpe vs ANCORA (o teste-chave da fragilidade) ------------
    L.append("-" * 100)
    L.append("PILAR 2 — SHARPE vs ANCORA DE SESSAO (o teste-chave da fragilidade)")
    L.append("-" * 100)
    L.append("  Se o Sharpe so existe na ancora 00 UTC (arbitraria) e some na ABERTURA REAL")
    L.append("  (~22h), o '1.50' era artefato da ancora. Se sobrevive em todas, e sinal.")
    L.append("")
    L.append(
        f"  {'ancora':>16} | {'take':>4} | {'#tr':>5} {'tr/dia':>6} | {'win%':>5} | "
        f"{'liq/tr':>8} | {'Sh_liq':>7} | {'DSR':>6} | {'liq_tot':>9} {'alpha':>8}"
    )
    L.append("  " + "-" * 94)
    for anchor in SESSION_ANCHORS:
        for take in TAKE_LEVELS:
            t = trials[(anchor, take)]
            tr_day = t.n_trades / t.n_sessions if t.n_sessions else 0.0
            alpha = t.total_net - bh_total
            L.append(
                f"  {_label(anchor):>16} | {('t'+str(take)):>4} | "
                f"{t.n_trades:>5d} {tr_day:>6.2f} | {t.win_rate*100:>4.1f}% | "
                f"{t.avg_net*100:>+7.3f}% | {t.sharpe_net_annual:>7.2f} | {t.dsr:>6.3f} | "
                f"{t.total_net*100:>+8.0f}% {alpha*100:>+7.0f}%"
            )
    L.append("")

    # melhor trial (por Sharpe liquido) e a real (ancora) vs o control (00 UTC)
    best_key = max(trials, key=lambda k: trials[k].sharpe_net_annual)
    best = trials[best_key]
    real_anchor = 22
    ctrl_anchor = 0
    real_t2 = trials[(real_anchor, 2)]
    ctrl_t2 = trials[(ctrl_anchor, 2)]
    L.append(
        f"  MELHOR trial: ancora={_label(best_key[0])} take{best_key[1]}  ->  "
        f"Sharpe_liq={best.sharpe_net_annual:.2f}  DSR={best.dsr:.3f}"
    )
    L.append(
        f"  ANCORA REAL (~22h) take2 -> Sharpe_liq={real_t2.sharpe_net_annual:.2f}  "
        f"DSR={real_t2.dsr:.3f}   |   CONTROL (00 UTC) take2 -> Sharpe_liq="
        f"{ctrl_t2.sharpe_net_annual:.2f}  DSR={ctrl_t2.dsr:.3f}"
    )
    # dispersao do Sharpe entre ancoras (take2): grande dispersao = fragilidade
    anchor_sharpes_t2 = [trials[(a, 2)].sharpe_net_annual for a in SESSION_ANCHORS]
    sh_min, sh_max = min(anchor_sharpes_t2), max(anchor_sharpes_t2)
    L.append(
        f"  Dispersao do Sharpe entre ancoras (take2): [{sh_min:.2f} .. {sh_max:.2f}]  "
        f"(amplitude {sh_max - sh_min:.2f}).  Amplitude grande => o numero depende da ancora."
    )
    L.append("")

    # ---- PILAR 3: walk-forward OOS por sub-periodo ---------------------------
    L.append("-" * 100)
    L.append("PILAR 3 — WALK-FORWARD OOS (o Sharpe/DSR se sustenta fora da amostra?)")
    L.append("-" * 100)
    L.append("  Divide o historico em sub-periodos contiguos e mede a MELHOR config")
    L.append(f"  (ancora={_label(best_key[0])} take{best_key[1]}) em cada um. Sinal estavel?")
    L.append("")

    sub = _walk_forward(df, anchor=best_key[0], take=best_key[1],
                        full_spread=full_spread, n_periods=5)
    L.append(
        f"  {'sub-periodo':>23} | {'#tr':>5} | {'win%':>5} | {'liq/tr':>8} | "
        f"{'Sh_liq':>7} | {'DSR':>6} | {'liq_tot':>9} | {'BH_Sh':>6} {'BH_tot':>8}"
    )
    L.append("  " + "-" * 92)
    oos_sharpes = []
    oos_signs = []
    for s in sub:
        oos_sharpes.append(s["sharpe"])
        oos_signs.append(np.sign(s["sharpe"]))
        L.append(
            f"  {s['label']:>23} | {s['n_trades']:>5d} | {s['win_rate']*100:>4.1f}% | "
            f"{s['avg_net']*100:>+7.3f}% | {s['sharpe']:>7.2f} | {s['dsr']:>6.3f} | "
            f"{s['total_net']*100:>+8.0f}% | {s['bh_sharpe']:>6.2f} {s['bh_total']*100:>+7.0f}%"
        )
    L.append("")
    n_pos = sum(1 for x in oos_signs if x > 0)
    n_beats_bh = sum(1 for s in sub if s["sharpe"] > s["bh_sharpe"])
    sh_arr = np.array(oos_sharpes, dtype=float)
    L.append(
        f"  OOS: Sharpe positivo em {n_pos}/{len(sub)} sub-periodos;  "
        f"bate B&H em {n_beats_bh}/{len(sub)};  "
        f"Sharpe medio={sh_arr.mean():.2f}  desvio={sh_arr.std(ddof=1) if sh_arr.size>1 else 0.0:.2f}  "
        f"min={sh_arr.min():.2f}"
    )
    L.append("")

    # ---- VEREDITO (sem suavizar) ---------------------------------------------
    # Criterios de RESSURREICAO (todos obrigatorios):
    dsr_ok = best.dsr >= 0.95
    # robustez de ancora: a ANCORA REAL (~22h) tem que segurar perto da melhor (>=80%)
    # e ser >= 1.0 anual; e nao pode haver troca de SINAL entre ancoras.
    anchor_ok = (
        real_t2.sharpe_net_annual >= 1.0
        and real_t2.sharpe_net_annual >= 0.8 * sh_max
        and sh_min > 0.0
    )
    # estabilidade OOS: Sharpe positivo em >=80% dos sub-periodos E nunca troca de sinal
    oos_ok = (n_pos >= int(np.ceil(0.8 * len(sub)))) and all(x >= 0 for x in oos_signs)
    beats_bh = best.sharpe_net_annual > bh_sharpe

    ressuscita = dsr_ok and anchor_ok and oos_ok

    L.append("=" * 100)
    L.append("VEREDITO DEFINITIVO")
    L.append("=" * 100)
    L.append(f"  n_trials honesto = {len(SESSION_ANCHORS)} ancoras x {len(TAKE_LEVELS)} takes "
             f"= {n_trials}  (sobre ~{span_years:.0f} anos de XAUUSD M15)")
    L.append("")
    L.append("  Criterios de RESSURREICAO (regua = risco-ajustado):")
    L.append(f"    [{'OK' if dsr_ok else 'X '}] DSR>=0.95 no historico longo (melhor trial): "
             f"DSR={best.dsr:.3f}")
    L.append(f"    [{'OK' if anchor_ok else 'X '}] Sharpe robusto na ANCORA REAL (~22h, nao so 00 UTC): "
             f"22h take2 Sh={real_t2.sharpe_net_annual:.2f} vs melhor {sh_max:.2f}; "
             f"min entre ancoras={sh_min:.2f}")
    L.append(f"    [{'OK' if oos_ok else 'X '}] Estavel OUT-OF-SAMPLE: Sharpe>0 em "
             f"{n_pos}/{len(sub)} sub-periodos, sem troca de sinal")
    L.append(f"    [info] Sharpe_liq > buy&hold do ouro ({bh_sharpe:.2f}): "
             f"{'SIM' if beats_bh else 'NAO'} (melhor Sh={best.sharpe_net_annual:.2f})")
    L.append("")

    if ressuscita:
        L += [
            "  " + "#" * 90,
            "  ##" + " " * 86 + "##",
            "  ##" + "   A   F I M A T H E   V I V E .".center(86) + "##",
            "  ##" + " " * 86 + "##",
            "  ##" + ("O Sharpe sobreviveu a ~%.0f anos, a ANCORA REAL e ao OOS." % span_years).center(86) + "##",
            "  ##" + "Risco-ajustado robusto -> DOMINA o buy&hold via alavancagem.".center(86) + "##",
            "  ##" + " " * 86 + "##",
            "  ##" + "PRECISA DE AUDITORIA DO CODER antes de QUALQUER dinheiro real:".center(86) + "##",
            "  ##" + "revisao de fills/look-ahead/custo/ancora + confirmacao em M1.".center(86) + "##",
            "  ##" + " " * 86 + "##",
            "  " + "#" * 90,
            "",
            "  RESPOSTA: o Sharpe ~1.50 era REAL — sobrevive ao historico longo, a ancora de",
            "  sessao real e ao walk-forward. A FIMATHE intraday tem edge risco-ajustado.",
        ]
    else:
        # explica POR QUE morreu (qual pilar falhou) e responde a pergunta-chave
        reasons = []
        if not dsr_ok:
            reasons.append(f"DSR={best.dsr:.3f}<0.95 (indistinguivel de sorte mesmo em ~{span_years:.0f} anos)")
        if not anchor_ok:
            reasons.append(
                f"FRAGIL a ancora (22h Sh={real_t2.sharpe_net_annual:.2f} vs best {sh_max:.2f}; "
                f"min entre ancoras={sh_min:.2f})"
            )
        if not oos_ok:
            reasons.append(f"instavel OOS (Sharpe>0 so em {n_pos}/{len(sub)} sub-periodos)")
        L += [
            "  " + "#" * 90,
            "  ##" + " " * 86 + "##",
            "  ##" + "F A L H A   D E F I N I T I V A".center(86) + "##",
            "  ##" + " " * 86 + "##",
            "  ##" + "A FIMATHE intraday NAO ressuscita.".center(86) + "##",
            "  ##" + " " * 86 + "##",
            "  " + "#" * 90,
            "",
            "  Pilar(es) que reprovaram: " + "; ".join(reasons) + ".",
        ]
        # responde a pergunta central com base em qual pilar quebrou
        if not dsr_ok and (not anchor_ok or real_t2.sharpe_net_annual < ctrl_t2.sharpe_net_annual):
            L.append("")
            L.append("  RESPOSTA: o Sharpe ~1.50 era ARTEFATO. Com mais dados o DSR nao sobe a 0.95")
            L.append("  e na ANCORA REAL (~22h) o Sharpe cai vs o control 00 UTC — confirma a")
            L.append("  fragilidade que o Coder apontou: o '1.50' dependia da ancora arbitraria.")
        elif not dsr_ok:
            L.append("")
            L.append("  RESPOSTA: o Sharpe pode ser estavel na ancora, mas continua INDISTINGUIVEL")
            L.append("  de sorte (DSR<0.95) mesmo no historico longo: nao e edge estatisticamente real.")
        else:
            L.append("")
            L.append("  RESPOSTA: o edge nao sobrevive fora da amostra — era especifico de um regime.")
        L.append("")
        L.append("  NOTA: 'risco-ajustado > B&H' so importa se for ROBUSTO (DSR + ancora + OOS).")
        L.append("  Um Sharpe alto que depende da ancora ou do periodo nao e alavancavel com")
        L.append("  seguranca — alavancar ruido amplifica a ruina, nao o retorno.")

    L.append("")
    L.append("  MARCA: este veredito NAO autoriza capital real. Requer AUDITORIA INDEPENDENTE")
    L.append("  do Coder (fills, look-ahead, custo, ancora) e confirmacao em M1 antes de tudo.")
    L.append("")
    L.append("  Arquivos: simulation/fimathe_definitive.py  |  data/fimathe_definitive_verdict.txt")
    L.append("            cache profundo: " + str(DEEP_CACHE))
    L.append("=" * 100)
    return "\n".join(L) + "\n"


def _walk_forward(
    df: pd.DataFrame, *, anchor: int | None, take: int, full_spread: float, n_periods: int
) -> list[dict]:
    """Divide o DF em n_periods sub-periodos CONTIGUOS (por tempo) e avalia a config
    em cada um (Sharpe/DSR/total + buy&hold do mesmo sub-periodo).

    DSR de cada sub-periodo usa n_trials=1 (dentro do sub-periodo nao ha selecao de
    multiplas configs) — e o PSR contra zero deflacionado por 1 trial, i.e. so a
    significancia da amostra. O n_trials de selecao (ancora x take) ja foi cobrado
    no PILAR 2 sobre o historico inteiro.
    """
    idx = pd.DatetimeIndex(df.index)
    edges = pd.date_range(idx[0], idx[-1], periods=n_periods + 1)
    out: list[dict] = []
    eng = CanalAbertura(CanalAberturaParams(session_break_hour=anchor))
    for k in range(n_periods):
        lo, hi = edges[k], edges[k + 1]
        # ultimo sub-periodo inclui o fim; os demais sao [lo, hi)
        mask = (idx >= lo) & (idx <= hi if k == n_periods - 1 else idx < hi)
        seg = df.loc[mask]
        if len(seg) < 96 * 30:  # < ~30 dias de M15 -> sub-periodo degenerado
            continue
        trades, ret_net, _g = _simulate_asset(
            seg, eng, take_level=take, full_spread=full_spread, skip_friday=False
        )
        n_trades = len(trades)
        wins = sum(1 for t in trades if t.ret_net > 0)
        win_rate = wins / n_trades if n_trades else 0.0
        avg_net = float(np.mean([t.ret_net for t in trades])) if trades else 0.0
        equity = np.cumprod(1.0 + ret_net)
        total_net = float(equity[-1] - 1.0) if equity.size else 0.0
        sh = observed_sharpe(ret_net, periods_per_year=PERIODS_PER_YEAR)
        v = evaluate_edge(
            ret_net, n_trials=1, periods_per_year=PERIODS_PER_YEAR, min_sharpe_annual=1.0
        )
        bh = _buy_hold_daily_returns(seg, eng)
        bh_sh = observed_sharpe(bh, periods_per_year=PERIODS_PER_YEAR)
        bh_total = float(np.prod(1.0 + bh) - 1.0) if bh.size else 0.0
        out.append({
            "label": f"{seg.index[0].date()}..{seg.index[-1].date()}",
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_net": avg_net,
            "sharpe": sh,
            "dsr": v.dsr,
            "total_net": total_net,
            "bh_sharpe": bh_sh,
            "bh_total": bh_total,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teste DEFINITIVO da FIMATHE intraday (XAUUSD, historico longo)"
    )
    parser.add_argument("--report-file", default="data/fimathe_definitive_verdict.txt")
    parser.add_argument("--start-year", type=int, default=2004,
                        help="ano inicial do historico profundo (default 2004)")
    parser.add_argument("--force", action="store_true",
                        help="ignora o cache profundo e rebaixa")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    df = load_deep_xauusd(start_year=args.start_year, force=args.force)

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or len(df) < 96 * 252:  # < ~1 ano de M15 -> sem teste serio
        text = (
            "=" * 100 + "\n"
            "TESTE DEFINITIVO DA FIMATHE INTRADAY\n"
            + "=" * 100 + "\n\n"
            "!!! DADOS PENDENTES / INSUFICIENTES !!!\n"
            "Nao foi possivel obter historico profundo de XAUUSD M15. Rode:\n"
            "  uv run python -m simulation.fimathe_definitive --force\n"
            "Sem historico longo NAO ha veredito definitivo (nada inventado).\n"
            f"(barras em cache: {len(df)})\n"
        )
        print(text)
        out.write_text(text, encoding="utf-8")
        return 2

    text = _build_report(df, start_year=args.start_year)
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"\nRelatorio salvo em {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
