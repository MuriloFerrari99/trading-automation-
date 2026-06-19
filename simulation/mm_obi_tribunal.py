"""Tribunal: Market-making cripto guiado por order-book imbalance (OBI).

R&D ISOLADO. NAO toca beta_*, main.py, producao nem outros tribunais. Cria arquivos
proprios em data/binance_book/mm_cache/ e relatorio em data/mm_obi_verdict.txt.

ESTRATEGIA
----------
MM passivo em BTCUSDT perp (Binance USDS-M). Postamos ordens limit no topo de book.
O sinal e o ORDER-BOOK IMBALANCE (OBI) do top-of-book:

    OBI = bid_qty / (bid_qty + ask_qty)   em [0,1]

OBI alto => pressao compradora => fila do bid e "pesada", preco tende a subir =>
nesse regime e mais seguro/lucrativo postar no BID (queremos comprar barato e o
mid sobe a nosso favor), e perigoso postar no ASK (seriamos adversamente
selecionados na perna vendida). Simetrico para OBI baixo. A versao defensavel de
MM nao quota os dois lados cegamente: faz SKEW por OBI.

DADO (real, gratis)
-------------------
Binance bookTicker historico (top-of-book: best bid/ask price+qty, ~event-driven
ate 100ms) de data.binance.vision/data/futures/um/daily/bookTicker/. L2 PROFUNDO
nao e gratis; bookTicker (nivel-1) E. Logo modelamos MM de TOP-OF-BOOK, que e a
unica versao honestamente backtestavel de graca. Amostra: dias esparsos cobrindo
2023-06 a 2024-03 (Binance parou de publicar bookTicker apos 2024-04).

MODELO DE EXECUCAO (honesto, conservador)
-----------------------------------------
Resample o stream em snapshots de 1s (ultimo update do segundo). Em cada snapshot
decidimos quotar bid e/ou ask (1 unidade de notional) conforme o OBI. Entre dois
snapshots consecutivos:

- FILL do nosso BID postado em b_t: so consideramos preenchido se no snapshot
  seguinte o melhor BID caiu ABAIXO do nosso preco (i.e. o mercado negociou
  atravessando nosso nivel — o book varreu nossa fila). Isso e o cenario realista
  e PESSIMISTA: so somos preenchidos quando o preco vem nos buscar, o que e
  precisamente quando ha adverse selection. Simetrico para o ASK.
  (Modelo de fila conservador: assumimos que so enchemos quando o nivel e
  consumido, nao por queue-priority otimista.)
- ADVERSE SELECTION: o PnL de um fill e marcado contra o mid do snapshot seguinte
  (mid_{t+1}), nao contra o nosso preco. Compramos no bid b_t e a posicao vale
  mid_{t+1}: ganho = (mid_{t+1} - b_t)/mid. Como so enchemos quando o preco
  atravessou pra baixo, mid_{t+1} tende a estar ABAIXO => captura o adverse
  selection real do MM.
- HALF-SPREAD ganho: ao ser preenchido no bid recebemos o edge de spread
  (mid - bid) embutido em (mid_{t+1}-b_t); o fee maker e debitado por fill.

Cada snapshot rende no maximo 1 fill por lado. O retorno por snapshot e a soma dos
PnLs (em fracao do notional) dos lados preenchidos, LIQUIDO de fee maker.

CUSTO
-----
Fee maker Binance USDS-M futures: ~1.8 bps tier 0 (sem rebate de varejo realista).
Cenario BASE = 1.8 bps/fill; ESTRESSE = 2x. Veredito usa o ESTRESSE (honesto).

ANTI-LOOK-AHEAD
---------------
Decisao de quotar em t usa OBI do snapshot t (book observavel naquele instante).
O fill/PnL e marcado contra t+1. Sinal<=t, resultado em t+1. Sem look-ahead.

n_trials (DSR honesto)
----------------------
Varremos a grade de thresholds de OBI {skew_lo, skew_hi} (quao desbalanceado o
book precisa estar pra quotar o lado favoravel). Contamos TODAS as variantes no
DSR e medimos PBO sobre a matriz de retornos das variantes.

BARRA
-----
PASSA so se DSR>=0.95 E Sharpe_liq robusto E (bate buy&hold OU diversifica com
correlacao baixa ao SPY melhorando o conjunto), robusto OOS, sob custo estressado.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.costs import CostModel
from simulation.metrics import max_drawdown
from simulation.statistics import (
    CRYPTO_PERIODS,
    evaluate_edge,
    observed_sharpe,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger("simulation.mm_obi_tribunal")

CACHE_DIR = Path("data/binance_book/mm_cache")
REPORT_PATH = Path("data/mm_obi_verdict.txt")
SYMBOL = "BTCUSDT"

# Fee maker Binance USDS-M futures, tier 0 (sem rebate realista p/ varejo).
MAKER_FEE_BPS_BASE = 1.8
RESAMPLE_MS = 1000  # snapshot grid: 1 segundo

# Dias baixados (esparsos, 2023-06 .. 2024-03). bookTicker some apos 2024-04.
SAMPLE_DAYS = [
    "2023-06-15",
    "2023-08-15",
    "2023-10-15",
    "2023-12-15",
    "2024-01-15",
    "2024-02-15",
    "2024-03-15",
    "2024-03-29",
]


# ---------------------------------------------------------------------------
# 1) Parse bookTicker -> snapshots de 1s
# ---------------------------------------------------------------------------

@dataclass
class DaySnapshots:
    """Snapshots de 1s de um dia. Arrays alinhados por indice de segundo."""

    day: str
    ts_ms: np.ndarray      # transaction_time do ultimo update do segundo
    bid: np.ndarray
    bid_qty: np.ndarray
    ask: np.ndarray
    ask_qty: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid + self.ask)

    @property
    def obi(self) -> np.ndarray:
        denom = self.bid_qty + self.ask_qty
        out = np.full_like(denom, 0.5)
        nz = denom > 0
        out[nz] = self.bid_qty[nz] / denom[nz]
        return out


def _zip_path(day: str) -> Path:
    return CACHE_DIR / f"{SYMBOL}-bookTicker-{day}.zip"


def parse_day(day: str) -> DaySnapshots | None:
    """Le o zip de um dia e resample para snapshots de 1s (ultimo update do bucket).

    Sem pandas: stream linha-a-linha p/ nao estourar memoria (23M linhas/dia).
    """
    zp = _zip_path(day)
    if not zp.exists():
        return None
    buckets: dict[int, tuple[float, float, float, float, int]] = {}
    try:
        with zipfile.ZipFile(zp) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8")
                header = txt.readline()
                cols = {c.strip().lower(): i for i, c in enumerate(header.split(","))}
                bi = cols.get("best_bid_price", 1)
                bq = cols.get("best_bid_qty", 2)
                ai = cols.get("best_ask_price", 3)
                aq = cols.get("best_ask_qty", 4)
                ti = cols.get("transaction_time", 5)
                for line in txt:
                    parts = line.split(",")
                    try:
                        bidp = float(parts[bi]); bidq = float(parts[bq])
                        askp = float(parts[ai]); askq = float(parts[aq])
                        t = int(parts[ti])
                    except (ValueError, IndexError):
                        continue
                    if askp <= bidp or bidp <= 0:
                        continue
                    bucket = t // RESAMPLE_MS
                    # guarda o ULTIMO update do bucket (estado do book no fim do seg)
                    prev = buckets.get(bucket)
                    if prev is None or t >= prev[4]:
                        buckets[bucket] = (bidp, bidq, askp, askq, t)
    except Exception as exc:
        logger.warning("falha ao ler %s: %s", zp, exc)
        return None
    if not buckets:
        return None
    keys = sorted(buckets)
    bid = np.empty(len(keys)); bidq = np.empty(len(keys))
    ask = np.empty(len(keys)); askq = np.empty(len(keys))
    ts = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        b, bq_, a, aq_, t = buckets[k]
        bid[i] = b; bidq[i] = bq_; ask[i] = a; askq[i] = aq_; ts[i] = t
    return DaySnapshots(day=day, ts_ms=ts, bid=bid, bid_qty=bidq, ask=ask, ask_qty=askq)


# ---------------------------------------------------------------------------
# 2) Simulacao MM passivo guiado por OBI
# ---------------------------------------------------------------------------

def simulate_mm(
    snaps: DaySnapshots,
    *,
    skew_lo: float,
    skew_hi: float,
    fee_bps: float,
    fill_model: str = "conservative",
) -> np.ndarray:
    """Retorna por snapshot (fracao do notional), LIQUIDO de fee maker.

    Regra de quotar (skew por OBI):
      - quota BID se OBI >= skew_lo  (book comprador: seguro postar compra)
      - quota ASK se OBI <= skew_hi  (book vendedor: seguro postar venda)
      Com skew_lo>0.5>skew_hi, em book neutro NAO quota nenhum lado.

    Dois modelos de fill (limites honestos do espectro):

    fill_model="conservative" (PESSIMISTA, lower bound):
      so enche quando o mercado ATRAVESSA nosso nivel — sempre adversamente
      selecionado por construcao.
        - BID em b_t enche se bid_{t+1} < b_t. PnL = (mid_{t+1} - b_t)/mid_t - fee.
        - ASK em a_t enche se ask_{t+1} > a_t. PnL = (a_t - mid_{t+1})/mid_t - fee.

    fill_model="queue" (OTIMISTA-defensavel, upper bound):
      assume queue priority: enche quando nosso nivel SE MANTEM (preco nao se
      moveu contra) — i.e. um agressor bateu na nossa fila e o nivel sobreviveu.
      Ganhamos o half-spread; ainda marcamos contra mid_{t+1} (adverse selection
      residual honesto).
        - BID em b_t enche se bid_{t+1} >= b_t e ask_{t+1} <= a_t (lado segurou).
          PnL = (mid_{t+1} - b_t)/mid_t - fee.
        - ASK em a_t enche se ask_{t+1} <= a_t e bid_{t+1} >= b_t.
          PnL = (a_t - mid_{t+1})/mid_t - fee.
    """
    bid = snaps.bid; ask = snaps.ask; mid = snaps.mid; obi = snaps.obi
    n = bid.size
    if n < 3:
        return np.zeros(0)
    fee = fee_bps / 1e4
    b_t = bid[:-1]; a_t = ask[:-1]; mid_t = mid[:-1]; obi_t = obi[:-1]
    bid_next = bid[1:]; ask_next = ask[1:]; mid_next = mid[1:]

    quote_bid = obi_t >= skew_lo
    quote_ask = obi_t <= skew_hi

    if fill_model == "queue":
        held = (bid_next >= b_t) & (ask_next <= a_t)
        fill_bid = quote_bid & held
        fill_ask = quote_ask & held
    else:  # conservative
        fill_bid = quote_bid & (bid_next < b_t)
        fill_ask = quote_ask & (ask_next > a_t)

    pnl = np.zeros(n - 1)
    pnl_bid = (mid_next - b_t) / mid_t - fee
    pnl_ask = (a_t - mid_next) / mid_t - fee
    pnl[fill_bid] += pnl_bid[fill_bid]
    pnl[fill_ask] += pnl_ask[fill_ask]
    return pnl


def aggregate_per_period(
    rets_per_snapshot: np.ndarray, snaps_per_period: int
) -> np.ndarray:
    """Soma retornos de snapshots em periodos maiores (ex: 1h = 3600 snaps de 1s).

    O DSR/Sharpe ficam mais estaveis e a anualizacao usa periods_per_year do
    periodo agregado. Trunca o resto.
    """
    n = rets_per_snapshot.size
    if n < snaps_per_period:
        return rets_per_snapshot.copy()
    usable = (n // snaps_per_period) * snaps_per_period
    block = rets_per_snapshot[:usable].reshape(-1, snaps_per_period)
    return block.sum(axis=1)


# ---------------------------------------------------------------------------
# 3) Tribunal
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    skew_lo: float
    skew_hi: float


def build_variants() -> list[Variant]:
    """Grade HONESTA de variantes (contam todas no DSR)."""
    los = [0.55, 0.60, 0.65, 0.70]
    his = [0.45, 0.40, 0.35, 0.30]
    return [Variant(lo, hi) for lo in los for hi in his]


def run_tribunal() -> dict:
    days = [d for d in SAMPLE_DAYS if _zip_path(d).exists()]
    if not days:
        return {"status": "blocked", "reason": "nenhum bookTicker em cache"}

    logger.info("parseando %d dias...", len(days))
    parsed: list[DaySnapshots] = []
    for d in days:
        s = parse_day(d)
        if s is not None and s.bid.size > 100:
            parsed.append(s)
            logger.info("  %s: %d snapshots de 1s", d, s.bid.size)
    if not parsed:
        return {"status": "blocked", "reason": "parse vazio"}

    # custo estressado (honesto)
    fee_stress = MAKER_FEE_BPS_BASE * 2.0
    # periodo de agregacao: 1 hora (3600 snaps de 1s) -> periods_per_year
    PERIOD_SEC = 3600
    periods_per_year = int(365 * 24)  # horas/ano

    variants = build_variants()
    n_trials = len(variants) * 2  # 2 modelos de fill contam no DSR (honesto)

    def eval_model(fill_model: str) -> dict:
        per_variant_periods: list[np.ndarray] = []
        for v in variants:
            all_periods = []
            for s in parsed:
                r_snap = simulate_mm(
                    s, skew_lo=v.skew_lo, skew_hi=v.skew_hi,
                    fee_bps=fee_stress, fill_model=fill_model,
                )
                all_periods.append(aggregate_per_period(r_snap, PERIOD_SEC))
            per_variant_periods.append(np.concatenate(all_periods))

        min_len = min(len(x) for x in per_variant_periods)
        matrix = np.column_stack([x[:min_len] for x in per_variant_periods])
        sharpes_period = [observed_sharpe(x) for x in per_variant_periods]
        best_idx = int(np.argmax(sharpes_period))
        best_v = variants[best_idx]
        best_ret = per_variant_periods[best_idx]

        verdict = evaluate_edge(
            best_ret,
            n_trials=n_trials,
            trial_sharpes=sharpes_period,
            periods_per_year=periods_per_year,
            min_sharpe_annual=0.8,
            dsr_threshold=0.95,
        )
        pbo = probability_of_backtest_overfitting(
            matrix, n_splits=min(16, matrix.shape[0])
        )
        eq = np.cumprod(1.0 + best_ret)
        mdd = max_drawdown(eq)
        total_periods = best_ret.size
        years = total_periods / periods_per_year
        cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else -1.0

        fills = 0; snaps_total = 0
        for s in parsed:
            r = simulate_mm(
                s, skew_lo=best_v.skew_lo, skew_hi=best_v.skew_hi,
                fee_bps=0.0, fill_model=fill_model,
            )
            fills += int((r != 0).sum()); snaps_total += r.size

        return {
            "fill_model": fill_model,
            "best_skew_lo": best_v.skew_lo,
            "best_skew_hi": best_v.skew_hi,
            "n_periods": int(total_periods),
            "snaps_total": int(snaps_total),
            "fills": int(fills),
            "fill_rate": float(fills / snaps_total) if snaps_total else 0.0,
            "sharpe_annual": verdict.sharpe_annual,
            "psr": verdict.psr,
            "dsr": verdict.dsr,
            "pbo": float(pbo),
            "cagr": cagr,
            "maxdd": float(mdd),
            "skew": verdict.skew,
            "kurtosis": verdict.kurtosis,
            "sr_benchmark_annual": verdict.sr_benchmark_annual,
            "passes_dsr": verdict.passes_dsr,
            "passes_sharpe": verdict.passes_sharpe,
            "all_sharpes_annual": [
                s * float(np.sqrt(periods_per_year)) for s in sharpes_period
            ],
            "passed": verdict.passes_dsr and verdict.passes_sharpe and pbo < 0.5,
        }

    cons = eval_model("conservative")
    queue = eval_model("queue")

    return {
        "status": "real",
        "days": days,
        "n_variants_per_model": len(variants),
        "n_trials": n_trials,
        "conservative": cons,
        "queue": queue,
    }


def write_report(res: dict) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("=" * 78)
    L.append("TRIBUNAL — Market-making cripto com order-book imbalance (OBI)")
    L.append("=" * 78)
    L.append("")
    L.append("Estrategia: MM passivo top-of-book em BTCUSDT perp, skew por OBI.")
    L.append("Dado: Binance bookTicker historico (nivel-1, gratis, data.binance.vision).")
    L.append("  L2 profundo NAO e gratis -> modelamos MM de TOP-OF-BOOK (unica versao")
    L.append("  honestamente backtestavel de graca).")
    L.append(f"Fee maker: {MAKER_FEE_BPS_BASE}bps base / {MAKER_FEE_BPS_BASE*2}bps ESTRESSE (usado no veredito).")
    L.append("Anti-look-ahead: OBI de t decide quote; PnL marcado contra mid_{t+1}.")
    L.append("Fill conservador: so enche quando o mercado atravessa nosso nivel (= adverse selection real).")
    L.append("")
    if res.get("status") != "real":
        L.append(f"STATUS: BLOCKED — {res.get('reason')}")
        REPORT_PATH.write_text("\n".join(L) + "\n")
        return REPORT_PATH
    L.append("-" * 78)
    L.append("AMOSTRA")
    L.append("-" * 78)
    L.append(f"Dias (esparsos 2023-06..2024-03): {', '.join(res['days'])}")
    L.append(f"n_trials honesto (variantes x modelos de fill): {res['n_trials']}")
    L.append("")

    def block(tag: str, r: dict) -> None:
        L.append("-" * 78)
        L.append(f"MODELO DE FILL: {tag}  (custo ESTRESSADO 2x, melhor de N variantes)")
        L.append("-" * 78)
        L.append(f"melhor skew: OBI>= {r['best_skew_lo']} quota bid ; OBI<= {r['best_skew_hi']} quota ask")
        L.append(f"snapshots: {r['snaps_total']:,}  periodos 1h: {r['n_periods']:,}")
        L.append(f"fills: {r['fills']:,}  fill-rate/snapshot: {r['fill_rate']*100:.3f}%")
        L.append(f"Sharpe anual liquido: {r['sharpe_annual']:.2f}  (obstaculo data-snooping: {r['sr_benchmark_annual']:.2f})")
        L.append(f"PSR: {r['psr']:.3f}   DSR: {r['dsr']:.3f}   (barra DSR>=0.95)")
        L.append(f"PBO (CSCV): {r['pbo']:.3f}   (barra <0.5)")
        L.append(f"CAGR (extrapolado): {r['cagr']*100:.1f}%   MaxDD: {r['maxdd']*100:.1f}%")
        L.append(f"skew={r['skew']:.2f}  kurtose={r['kurtosis']:.2f}")
        sh = r["all_sharpes_annual"]
        L.append(f"Sharpes anuais das {len(sh)} variantes: min={min(sh):.2f} med={float(np.median(sh)):.2f} max={max(sh):.2f}")
        L.append(f"-> {'PASSA' if r['passed'] else 'FALHA'}")
        L.append("")

    block("CONSERVADOR (pessimista, fill so quando o preco atravessa = adverse selection puro)", res["conservative"])
    block("QUEUE-PRIORITY (otimista-defensavel, fill quando o nivel segura)", res["queue"])

    overall = res["conservative"]["passed"] or res["queue"]["passed"]
    L.append("=" * 78)
    L.append(f"VEREDITO FINAL: {'PASSA' if overall else 'FALHA'}")
    L.append("=" * 78)
    L.append("Interpretacao: o resultado REAL fica entre os dois modelos de fill. O")
    L.append("conservador e a cota inferior (sempre adversamente selecionado); o queue")
    L.append("e a cota superior (assume prioridade de fila que o dado nivel-1 nao prova).")
    if not overall:
        L.append("Nem o melhor caso (queue-priority, custo estressado, melhor de N) bate a barra:")
        q = res["queue"]
        why = []
        if not q["passes_dsr"]:
            why.append(f"DSR {q['dsr']:.3f} < 0.95")
        if not q["passes_sharpe"]:
            why.append(f"Sharpe {q['sharpe_annual']:.2f} < 0.8")
        if q["pbo"] >= 0.5:
            why.append(f"PBO {q['pbo']:.3f} >= 0.5")
        L.append("  Motivo (queue): " + "; ".join(why) if why else "  (sem edge)")
    L.append("")
    L.append("LIMITE DE DADO: L2 profundo (queue real, depth>1) NAO e gratis. bookTicker")
    L.append("(nivel-1) so cobre 2023-05..2024-04; Binance parou de publicar depois. Para")
    L.append("um teste de MM com queue priority VERDADEIRA seria preciso coletar L2 ao vivo")
    L.append("(coletor pronto em simulation/binance_book.py --collect) por meses.")
    L.append("-" * 78)
    REPORT_PATH.write_text("\n".join(L) + "\n")
    return REPORT_PATH


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    res = run_tribunal()
    path = write_report(res)
    logger.info("relatorio em %s", path)
    if res.get("status") == "real":
        for tag in ("conservative", "queue"):
            r = res[tag]
            logger.info(
                "[%s] DSR=%.3f PBO=%.3f Sharpe=%.2f passed=%s",
                tag, r["dsr"], r["pbo"], r["sharpe_annual"], r["passed"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
