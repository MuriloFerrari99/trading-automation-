"""Avaliacao do historico de decisoes — metricas por estrategia e por regime.

Le as linhas da tabela `decisions` e produz um Report com estatisticas de
performance AJUSTADAS AO RISCO (nao so retorno bruto), agrupadas por estrategia
e por regime de mercado. E essa visao que responde "onde o bot ganha? onde ele
sangra?" — o insumo para recalibrar parametros (Camada 0 -> melhoria continua).

Sem dependencias externas: stdlib `statistics` (sem numpy/pandas), para manter
o pacote leve e testavel.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from feedback.decision_log import DecisionLog
from feedback.models import OutcomeStatus

# Desfechos que representam um trade efetivamente realizado (fechado).
_CLOSED = {OutcomeStatus.WIN.value, OutcomeStatus.LOSS.value, OutcomeStatus.BREAKEVEN.value}


@dataclass
class GroupStats:
    """Estatisticas de um grupo (uma estrategia, um regime, ou o total)."""

    label: str
    n_decisions: int = 0
    n_skips: int = 0  # decisoes de nao operar (hold/skip)
    n_open: int = 0  # trades ainda abertos
    n_trades: int = 0  # trades fechados (win/loss/breakeven)
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0  # soma dos P&L positivos
    gross_loss: float = 0.0  # soma dos P&L negativos (valor negativo)
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    _returns: list[float] = field(default_factory=list, repr=False)
    _pnls: list[float] = field(default_factory=list, repr=False)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_win(self) -> float:
        return self.gross_profit / self.wins if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return self.gross_loss / self.losses if self.losses else 0.0

    @property
    def profit_factor(self) -> float:
        """gross_profit / |gross_loss|. inf se nunca perdeu (com lucros)."""
        if self.gross_loss == 0:
            return math.inf if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)

    @property
    def expectancy(self) -> float:
        """P&L medio esperado por trade fechado."""
        return self.total_pnl / self.n_trades if self.n_trades else 0.0

    @property
    def sharpe_per_trade(self) -> float:
        """Sharpe NAO anualizado, sobre os retornos % por trade. So faz sentido
        com varios trades; e um proxy de consistencia, nao um Sharpe anual."""
        rets = [r for r in self._returns if r is not None]
        if len(rets) < 2:
            return 0.0
        sd = statistics.pstdev(rets)
        return statistics.fmean(rets) / sd if sd else 0.0

    @property
    def avg_return_pct(self) -> float:
        rets = [r for r in self._returns if r is not None]
        return statistics.fmean(rets) if rets else 0.0


@dataclass
class Report:
    overall: GroupStats
    by_strategy: dict[str, GroupStats]
    by_regime: dict[str, GroupStats]
    # Cruzamento estrategia x regime: a granularidade que o gate de decisao usa
    # ("esta estrategia, NESTE regime, tem edge?"). Chave: (strategy, regime).
    by_strategy_regime: dict[tuple[str, str], GroupStats] = field(default_factory=dict)

    def combo(self, strategy: str, regime: str) -> GroupStats | None:
        return self.by_strategy_regime.get((strategy, regime))


def _accumulate(stats: GroupStats, row: dict) -> None:
    stats.n_decisions += 1
    status = row.get("outcome_status")
    if status in (OutcomeStatus.SKIPPED.value,):
        stats.n_skips += 1
        return
    if status == OutcomeStatus.OPEN.value:
        stats.n_open += 1
        return
    if status == OutcomeStatus.CANCELED.value:
        return
    if status not in _CLOSED:
        return

    pnl = float(row["realized_pnl"]) if row.get("realized_pnl") is not None else 0.0
    stats.n_trades += 1
    stats.total_pnl += pnl
    stats._pnls.append(pnl)
    if row.get("return_pct") is not None:
        stats._returns.append(float(row["return_pct"]))
    if pnl > 0:
        stats.wins += 1
        stats.gross_profit += pnl
    elif pnl < 0:
        stats.losses += 1
        stats.gross_loss += pnl  # negativo
    # breakeven (pnl == 0) conta como trade, mas nao como win nem loss.


def _finalize_drawdown(stats: GroupStats) -> None:
    """Max drawdown sobre a curva de P&L acumulado (ordem cronologica)."""
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for pnl in stats._pnls:
        cum += pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    stats.max_drawdown = max_dd  # <= 0


def evaluate(records: list[dict]) -> Report:
    """Constroi o Report a partir das linhas da tabela `decisions`
    (ja em ordem cronologica — DecisionLog.all_records() garante isso)."""
    overall = GroupStats(label="TOTAL")
    by_strategy: dict[str, GroupStats] = defaultdict(lambda: GroupStats(label=""))
    by_regime: dict[str, GroupStats] = defaultdict(lambda: GroupStats(label=""))
    by_combo: dict[tuple[str, str], GroupStats] = defaultdict(lambda: GroupStats(label=""))

    for row in records:
        _accumulate(overall, row)
        s = by_strategy[row["strategy"]]
        s.label = row["strategy"]
        _accumulate(s, row)
        r = by_regime[row["regime"]]
        r.label = row["regime"]
        _accumulate(r, row)
        key = (row["strategy"], row["regime"])
        c = by_combo[key]
        c.label = f"{row['strategy']}@{row['regime']}"
        _accumulate(c, row)

    for grp in [
        overall, *by_strategy.values(), *by_regime.values(), *by_combo.values()
    ]:
        _finalize_drawdown(grp)

    return Report(
        overall=overall,
        by_strategy=dict(by_strategy),
        by_regime=dict(by_regime),
        by_strategy_regime=dict(by_combo),
    )


def _fmt_row(s: GroupStats) -> str:
    pf = "inf" if s.profit_factor == math.inf else f"{s.profit_factor:.2f}"
    return (
        f"{s.label:<16} dec={s.n_decisions:<4} trades={s.n_trades:<4} "
        f"skip={s.n_skips:<4} win%={s.win_rate * 100:5.1f} PF={pf:<5} "
        f"exp={s.expectancy:8.2f} pnl={s.total_pnl:10.2f} "
        f"maxDD={s.max_drawdown:9.2f} shpe={s.sharpe_per_trade:5.2f}"
    )


def format_report(report: Report) -> str:
    """Renderiza o Report como texto (para CLI/log)."""
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("LOOP DE FEEDBACK — avaliacao de decisoes")
    lines.append("=" * 96)
    lines.append(_fmt_row(report.overall))
    lines.append("-" * 96)
    lines.append("Por estrategia:")
    for s in sorted(report.by_strategy.values(), key=lambda g: g.total_pnl, reverse=True):
        lines.append("  " + _fmt_row(s))
    lines.append("-" * 96)
    lines.append("Por regime de mercado:")
    for r in sorted(report.by_regime.values(), key=lambda g: g.total_pnl, reverse=True):
        lines.append("  " + _fmt_row(r))
    losers = [
        c for c in report.by_strategy_regime.values()
        if c.n_trades > 0 and c.expectancy < 0
    ]
    if losers:
        lines.append("-" * 96)
        lines.append("Combos estrategia@regime com expectancy NEGATIVA (candidatos a veto):")
        for c in sorted(losers, key=lambda g: g.expectancy):
            lines.append("  " + _fmt_row(c))
    lines.append("=" * 96)
    lines.append(
        "Legenda: dec=decisoes trades=fechados skip=nao-operou win%=acerto "
        "PF=profit factor exp=expectancy(P&L medio) pnl=P&L total "
        "maxDD=max drawdown shpe=Sharpe por trade"
    )
    return "\n".join(lines)


def generate_report(db_path: Path | str = "data/trading.sqlite") -> str:
    """Atalho: abre o log, avalia e devolve o relatorio em texto."""
    log = DecisionLog(db_path=db_path)
    try:
        return format_report(evaluate(log.all_records()))
    finally:
        log.close()


if __name__ == "__main__":  # pragma: no cover
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/trading.sqlite"
    print(generate_report(path))
