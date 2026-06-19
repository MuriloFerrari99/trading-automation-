"""Fase 1 end-to-end: baixa aggTrades, mede IC, escreve relatorio + veredito.

  uv run python -m microstructure.report \
      --symbols BTCUSDT,ETHUSDT --days 2026-06-10,2026-06-11,2026-06-12,2026-06-13,2026-06-14

Sem --days, usa o que ja estiver em data/microstructure_cache/ (modo --analyze).

O relatorio (data/microstructure_signal_report.txt) traz:
  1) Status dos dados (real / pendente), com a nota de que bookTicker historico
     esta indisponivel no arquivo publico (404) -> sinais de livro PENDENTES.
  2) Tabela IC por sinal x horizonte, com dispersao e estabilidade entre dias.
  3) Veredito: existe sinal de order-flow preditivo e persistente? horizonte/magnitude?
  4) Vale ir pra Fase 2 (modelo de fill + tribunal)?

Vereditos sao baseados em LIMIARES EXPLICITOS (abaixo), nao em narrativa. Se o IC
for ruido/decai instantaneo, o veredito diz que NAO ha sinal exploravel na nossa
latencia — sem vender.
"""

from __future__ import annotations

import logging
from pathlib import Path

from microstructure.data import (
    DEFAULT_SYMBOLS,
    download_universe_aggtrades,
    probe_bookticker,
)
from microstructure.ic import ICStudy, run_ic_study
from microstructure.signals import DEFAULT_HORIZONS_S, DEFAULT_WINDOWS_S

logger = logging.getLogger("microstructure.report")

REPORT_PATH = Path("data/microstructure_signal_report.txt")

# --- Limiares de veredito (explicitos). IC e Spearman, por dia, agregado entre dias.
# Em microestrutura HFT institucional, IC de 1-5% ja e material; mas a barra util
# para VAREJO (que come spread+fee+latencia) tem que ser maior para sobreviver.
IC_MEANINGFUL = 0.02      # |IC medio| acima disso e potencialmente material
IC_STRONG = 0.05          # |IC medio| acima disso e forte para microestrutura
TSTAT_SIGNIF = 3.0        # |t-stat| entre dias para chamar de persistente (conservador)
HIT_RATE_STABLE = 0.80    # fracao de dias com IC do mesmo sinal


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return "  n/a"
    return f"{x:+.{nd}f}"


def _signal_order(table) -> list[str]:
    """Ordena nomes de sinal de forma estavel (familia, janela)."""
    names = sorted({k[0] for k in table})
    def keyf(n: str):
        fam = n.rsplit("_", 1)[0]
        win = n.rsplit("_", 1)[1]
        wnum = int("".join(c for c in win if c.isdigit()) or 0)
        return (fam, wnum)
    return sorted(names, key=keyf)


def _render_ic_table(study: ICStudy) -> list[str]:
    lines: list[str] = []
    horizons = study.horizons_s
    head = f"{'signal':<16} " + " ".join(f"{f'{h}s':>9}" for h in horizons)
    lines.append("IC SPEARMAN MEDIO (entre dias) por sinal x horizonte")
    lines.append(head)
    lines.append("-" * len(head))
    for name in _signal_order(study.table):
        row = f"{name:<16} "
        cells = []
        for h in horizons:
            shic = study.table.get((name, h))
            cells.append(f"{_fmt(shic.ic_spearman_mean) if shic else '  n/a':>9}")
        lines.append(row + " ".join(cells))
    lines.append("")
    # dispersao (std) e t-stat para o melhor horizonte de cada sinal
    lines.append("ESTABILIDADE entre dias (no horizonte de |IC| maximo por sinal)")
    lines.append(
        f"{'signal':<16} {'h*':>4} {'IC':>9} {'std':>8} {'tstat':>8} "
        f"{'hit':>6} {'days':>5} {'obs':>10}"
    )
    lines.append("-" * 74)
    for name in _signal_order(study.table):
        best = None
        for h in horizons:
            shic = study.table.get((name, h))
            if shic is None or shic.ic_spearman_mean != shic.ic_spearman_mean:
                continue
            if best is None or abs(shic.ic_spearman_mean) > abs(best.ic_spearman_mean):
                best = shic
        if best is None:
            lines.append(f"{name:<16} {'-':>4}  (sem IC valido)")
            continue
        lines.append(
            f"{name:<16} {best.horizon_s:>3}s {_fmt(best.ic_spearman_mean):>9} "
            f"{best.ic_spearman_std:>8.4f} {_fmt(best.ic_spearman_tstat, 2):>8} "
            f"{best.hit_rate:>6.2f} {best.n_days:>5d} {best.n_obs_total:>10,d}"
        )
    return lines


def _decay_note(study: ICStudy) -> list[str]:
    """Comenta o decaimento: compara |IC| no menor vs maior horizonte para o melhor sinal."""
    # acha o sinal com maior |IC| em qualquer horizonte
    best_key = None
    best_abs = 0.0
    for k, shic in study.table.items():
        m = shic.ic_spearman_mean
        if m == m and abs(m) > best_abs:
            best_abs, best_key = abs(m), k
    if best_key is None:
        return ["  decaimento: sem IC valido para avaliar."]
    name = best_key[0]
    hs = study.horizons_s
    series = []
    for h in hs:
        shic = study.table.get((name, h))
        series.append((h, shic.ic_spearman_mean if shic else float("nan")))
    txt = ", ".join(f"{h}s={_fmt(v,4)}" for h, v in series)
    h0v = next((v for h, v in series if v == v), float("nan"))
    hlv = next((v for h, v in reversed(series) if v == v), float("nan"))
    trend = "decai" if abs(hlv) < abs(h0v) else "cresce/estavel"
    return [
        f"  decaimento (sinal de maior |IC| = {name}): {txt}",
        f"    -> de {hs[0]}s a {hs[-1]}s o |IC| {trend} "
        f"({_fmt(h0v,4)} -> {_fmt(hlv,4)}).",
    ]


def _verdict(studies: dict[str, ICStudy | None]) -> list[str]:
    have = {s: st for s, st in studies.items() if st is not None}
    lines: list[str] = ["VEREDITO — existe sinal de order-flow preditivo e persistente?"]
    if not have:
        lines.append(
            "  PENDENTE: sem aggTrades em cache. Rode o downloader (comando no topo) "
            "quando houver rede. Nada foi inventado."
        )
        return lines

    # Procura, entre todos os simbolos/sinais/horizontes, os que cruzam as barras.
    material: list[tuple[str, str, int, float, float, float, float]] = []
    strong: list[tuple] = []
    for sym, st in have.items():
        for (name, h), shic in st.table.items():
            m = shic.ic_spearman_mean
            if m != m:
                continue
            persistent = (
                (shic.ic_spearman_tstat == shic.ic_spearman_tstat
                 and abs(shic.ic_spearman_tstat) >= TSTAT_SIGNIF)
                or shic.hit_rate >= HIT_RATE_STABLE
            )
            if abs(m) >= IC_MEANINGFUL and persistent:
                rec = (sym, name, h, m, shic.ic_spearman_std, shic.ic_spearman_tstat, shic.hit_rate)
                material.append(rec)
                if abs(m) >= IC_STRONG:
                    strong.append(rec)

    if not material:
        # ha sinal mensuravel? reporta o maior |IC| visto mesmo que nao passe
        best = None
        for sym, st in have.items():
            for (name, h), shic in st.table.items():
                m = shic.ic_spearman_mean
                if m == m and (best is None or abs(m) > abs(best[3])):
                    best = (sym, name, h, m, shic.ic_spearman_tstat, shic.hit_rate)
        lines.append(
            "  NAO. Nenhum sinal de trade-flow atinge IC material E persistente "
            f"(barra: |IC|>= {IC_MEANINGFUL} com t-stat>= {TSTAT_SIGNIF} entre dias "
            f"ou hit-rate>= {HIT_RATE_STABLE})."
        )
        if best is not None:
            lines.append(
                f"  Maior |IC| observado: {best[1]} @ {best[2]}s em {best[0]} = "
                f"{_fmt(best[3])} (t-stat={_fmt(best[4],2)}, hit={best[5]:.2f}). "
                "Compativel com RUIDO na nossa latencia."
            )
        lines.append(
            "  Conclusao: nao ha sinal de order-flow EXPLORAVEL com aggTrades nesta "
            "latencia de varejo. Os sinais que poderiam ter mais conteudo (queue "
            "imbalance, microprice, OFI de livro) dependem de top-of-book/L2, hoje "
            "INDISPONIVEL no arquivo publico (bookTicker 404)."
        )
        return lines

    # ha material
    material.sort(key=lambda r: -abs(r[3]))
    # familias DISTINTAS (tfi_log e ofi_trade sao rank-identicos => mesma familia)
    def _family(name: str) -> str:
        fam = name.rsplit("_", 1)[0]
        return "net_signed_flow" if fam in ("tfi_log", "ofi_trade") else fam
    fams = sorted({_family(r[1]) for r in material})
    lines.append(
        f"  SIM (parcial): {len(material)} combinacao(oes) sinal x horizonte cruzam "
        f"a barra de IC material+persistente."
    )
    lines.append(
        f"  (Nota: ha redundancia — tfi_log_* e ofi_trade_* sao rank-identicos sob "
        f"Spearman. Familias de sinal DISTINTAS com edge: {', '.join(fams)}.) Top:"
    )
    for sym, name, h, m, sd, ts, hit in material[:6]:
        tag = "FORTE" if abs(m) >= IC_STRONG else "material"
        lines.append(
            f"    [{tag}] {sym} {name} @ {h}s: IC={_fmt(m)} std={sd:.4f} "
            f"t-stat={_fmt(ts,2)} hit={hit:.2f}"
        )
    lines.append("")
    lines.append(
        "  HORIZONTE/MAGNITUDE: ver acima. Em microestrutura, IC dessa ordem e o "
        "esperado e DECAI com o horizonte (ver secao de decaimento)."
    )
    if not strong:
        lines.append(
            "  Porem nenhum chega a IC FORTE (>= 0.05): o conteudo preditivo e pequeno. "
            "Margem fina para sobreviver a custo+fill."
        )
    return lines


def _phase2_recommendation(studies: dict[str, ICStudy | None]) -> list[str]:
    have = {s: st for s, st in studies.items() if st is not None}
    lines = ["VALE IR PARA A FASE 2 (modelo de fill + tribunal)?"]
    if not have:
        lines.append("  PENDENTE — sem dados. Sem decisao.")
        return lines
    # ha algum material+persistente?
    any_material = False
    for st in have.values():
        for shic in st.table.values():
            m = shic.ic_spearman_mean
            if m != m:
                continue
            persistent = (
                (shic.ic_spearman_tstat == shic.ic_spearman_tstat
                 and abs(shic.ic_spearman_tstat) >= TSTAT_SIGNIF)
                or shic.hit_rate >= HIT_RATE_STABLE
            )
            if abs(m) >= IC_MEANINGFUL and persistent:
                any_material = True
                break
    if any_material:
        lines += [
            "  CONDICIONAL-SIM. Ha sinal preditivo persistente em horizontes curtos, mas",
            "  IC NAO E LUCRO. Antes de qualquer claim de tradabilidade, a Fase 2 PRECISA:",
            "   (a) Modelo de FILL realista: adverse selection na latencia de varejo —",
            "       quando o sinal e forte, o preco JA andou; voce entra por ultimo.",
            "       Modelar fill maker (fila/cancel) e taker (cruza o spread).",
            "   (b) CUSTO: fee (taker ~ tier 0) + spread real (do bookTicker, hoje pendente).",
            "       Reaproveitar simulation.binance_book p/ o spread quando o dado voltar.",
            "   (c) Submeter o PnL liquido ao tribunal (simulation.statistics: PSR/DSR/PBO).",
            "  Se o edge bruto (IC pequeno) nao sobreviver a (a)+(b), morre na Fase 2 — e",
            "  esse e exatamente o ponto do gate.",
        ]
    else:
        lines += [
            "  NAO (ainda). O trade-flow de aggTrades nao mostra IC material+persistente",
            "  nesta latencia. Construir modelo de fill+tribunal sobre um sinal sem edge",
            "  bruto so consumiria tempo. Caminho com mais chance de conteudo:",
            "   - Coletar bookTicker/L2 AO VIVO (simulation.binance_book --collect) por",
            "     dias, e re-medir IC de queue imbalance / microprice / OFI (sinais de",
            "     livro), que costumam ter mais conteudo que trade-flow puro.",
            "   - So passar a Fase 2 se ALGUM sinal cruzar a barra de IC material+persistente.",
        ]
    return lines


def write_report(
    studies: dict[str, ICStudy | None],
    book_probe: dict[tuple[str, str], int | str] | None,
    days_requested: list[str],
) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    L.append("=" * 78)
    L.append("SYSTEM 3 — FASE 1: EXISTE SINAL DE ORDER-FLOW PREDITIVO? (cripto, Binance)")
    L.append("=" * 78)
    L.append("")
    L.append("Pergunta: o sinal de order-flow em t PREVE o retorno t->t+h em horizontes")
    L.append("curtos? Metrica = Information Coefficient (correlacao de Spearman/Pearson),")
    L.append("sem look-ahead (sinal termina em t, retorno comeca em t).")
    L.append("")
    L.append("Comando p/ baixar e reanalisar (quando houver rede):")
    L.append("  uv run python -m microstructure.report --symbols BTCUSDT,ETHUSDT \\")
    L.append("      --days 2026-06-10,2026-06-11,2026-06-12,2026-06-13,2026-06-14")
    L.append("")

    # (1) STATUS DOS DADOS
    L.append("-" * 78)
    L.append("(1) STATUS DOS DADOS")
    L.append("-" * 78)
    any_real = False
    for sym, st in studies.items():
        if st is None:
            L.append(f"  {sym:<10} aggTrades: DADOS PENDENTES (nenhum CSV em cache)")
        else:
            any_real = True
            L.append(
                f"  {sym:<10} aggTrades ({st.market}): REAL — {len(st.days)} dia(s) "
                f"[{', '.join(st.days)}]"
            )
    L.append("")
    L.append("  bookTicker (topo de livro p/ queue imbalance / microprice / OFI):")
    if book_probe:
        n404 = sum(1 for v in book_probe.values() if v == 404)
        n200 = sum(1 for v in book_probe.values() if v == 200)
        L.append(
            f"    HEAD em {len(book_probe)} arquivo(s): {n200} OK, {n404} 404. "
        )
        if n200 == 0:
            L.append(
                "    -> bookTicker historico INDISPONIVEL no arquivo publico (data.binance.vision)."
            )
            L.append(
                "       Sinais de LIVRO (queue imbalance, microprice, OFI Cont-Kukanov) ficam"
            )
            L.append(
                "       DEFINIDOS no codigo (microstructure.signals) porem PENDENTES de dado."
            )
            L.append(
                "       Captura viavel: coletor L2 AO VIVO (forward capture), ja pronto em"
            )
            L.append(
                "       simulation.binance_book --collect (reaproveitado, nao reescrito)."
            )
    else:
        L.append("    (probe nao executado nesta rodada)")
    L.append("")
    L.append("  Sinais EFETIVAMENTE testados nesta rodada (so aggTrades / trade-flow):")
    L.append("    tfi_Ws      = (Vbuy-Vsell)/(Vbuy+Vsell) na janela W")
    L.append("    tfi_cnt_Ws  = mesmo por CONTAGEM de trades")
    L.append("    tfi_log_Ws  = sign-log do fluxo assinado liquido")
    L.append("    ofi_trade_Ws= fluxo assinado normalizado por preco (proxy de OFI so-trade)")
    L.append(f"    janelas W = {list(DEFAULT_WINDOWS_S)}s | horizontes = {list(DEFAULT_HORIZONS_S)}s")
    L.append("    Preco de referencia = last-trade price (bookTicker indisponivel); proxy")
    L.append("    sem look-ahead, porem com bid-ask bounce que infla ruido nos horizontes curtos.")
    L.append("")

    # (2) TABELA IC
    L.append("-" * 78)
    L.append("(2) IC POR SINAL x HORIZONTE (dispersao + estabilidade entre dias)")
    L.append("-" * 78)
    if not any_real:
        L.append("  DADOS PENDENTES — sem aggTrades para medir IC.")
        L.append("")
    else:
        for sym, st in studies.items():
            if st is None:
                continue
            L.append("")
            L.append(f"### {sym} ({st.market}, step={st.step_s}s, dias={len(st.days)})")
            L.extend("  " + ln for ln in _render_ic_table(st))
            L.append("")
            L.append("  DECAIMENTO com o horizonte:")
            L.extend("  " + ln for ln in _decay_note(st))
        L.append("")
        L.append("  Leitura: IC>0 => sinal alto antecede retorno positivo. |t-stat| alto e")
        L.append("  hit-rate alto entre dias => persistente (nao ruido de 1 dia). std alto e")
        L.append("  hit~0.5 => instavel/ruido. |IC| caindo com h => decaimento (esperado).")
        L.append("")

    # (3) VEREDITO
    L.append("-" * 78)
    L.append("(3) " + _verdict(studies)[0])
    L.append("-" * 78)
    L.extend("  " + ln for ln in _verdict(studies)[1:])
    L.append("")

    # (4) FASE 2
    L.append("-" * 78)
    L.append("(4) " + _phase2_recommendation(studies)[0])
    L.append("-" * 78)
    L.extend("  " + ln for ln in _phase2_recommendation(studies)[1:])
    L.append("")

    # HONESTIDADE
    L.append("-" * 78)
    L.append("NOTA DE HONESTIDADE (IC != lucro)")
    L.append("-" * 78)
    L.append("  - IC mede CORRELACAO sinal->retorno futuro, condicao NECESSARIA, nao suficiente.")
    L.append("  - Adverse selection: na latencia de varejo, quando o sinal fica forte o preco")
    L.append("    JA se moveu; o fill que voce consegue e pior que o preco do sinal.")
    L.append("  - Custo: cruzar o spread (taker) ou perder fila (maker) come o edge. Em cripto")
    L.append("    o spread mediano e fino (ver simulation/binance_book/spread_report.txt).")
    L.append("  - So a Fase 2 (fill + custo + tribunal PSR/DSR/PBO) decide tradabilidade.")
    L.append("")

    text = "\n".join(L) + "\n"
    REPORT_PATH.write_text(text)
    logger.info("relatorio escrito em %s", REPORT_PATH)
    return REPORT_PATH


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="System 3 Fase 1: aggTrades -> IC de order-flow -> relatorio + veredito"
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="lista por virgula")
    parser.add_argument("--days", default="", help="YYYY-MM-DD por virgula; vazio = so analisa cache")
    parser.add_argument("--market", default="um", choices=["um", "spot"])
    parser.add_argument("--step", type=int, default=1, help="passo da grade em s (default 1)")
    parser.add_argument("--no-probe", action="store_true", help="nao faz HEAD no bookTicker")
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo com cache")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    days = [d.strip() for d in args.days.split(",") if d.strip()]

    if days:
        logger.info("baixando aggTrades %s para %d dia(s)...", symbols, len(days))
        download_universe_aggtrades(symbols, days, market=args.market, force=args.force)

    book_probe = None
    if not args.no_probe and days:
        logger.info("HEAD no bookTicker p/ documentar disponibilidade...")
        book_probe = probe_bookticker(symbols, days)

    studies: dict[str, ICStudy | None] = {}
    for sym in symbols:
        logger.info("estudo de IC: %s", sym)
        studies[sym] = run_ic_study(sym, market=args.market, step_s=args.step)
        st = studies[sym]
        if st is None:
            logger.info("  %s: DADOS PENDENTES", sym)
        else:
            logger.info("  %s: %d dias, %d sinais x horizontes", sym, len(st.days), len(st.table))

    write_report(studies, book_probe, days)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
