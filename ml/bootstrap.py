"""Bootstrap do dataset de ML a partir de backtest historico (dados reais).

O `DecisionLog` nasce vazio: o ML (SetupClassifier) so aprende DEPOIS que o
sistema operou e fechou trades. Para enriquecer a inteligencia ANTES de
acumular meses de paper trading, este harness gera um dataset rotulado a
partir de dados historicos reais (cache Alpaca em data/cache/*.csv):

  - usa a FimatheEngine (a MESMA engine da producao) como gerador de setup:
    a cada barra, signal/stop/take_profit + features (pcm_score, rsi, adx,
    dist_to_zn, breakout_strength). Os indicadores sao CAUSAIS (rolling/atr/
    rsi olham so o passado) -> a feature da barra i nao usa o futuro;
  - cada sinal LONG (signal==1) vira uma DECISAO gravada no instante da barra
    i, com o MESMO contexto que o `DecisionEnricher` gravaria em producao
    (mesmas chaves _FIM_KEYS + signal_strength) e o regime de `classify_regime`
    sobre as barras ate i;
  - a SAIDA e simulada para frente (i+1..): atinge o stop -> loss, atinge o
    alvo (R:R 1:2 = take_profit_2) -> win, senao expira por tempo (rotulado
    pelo sinal do retorno). Sem look-ahead: features de [:i], desfecho de
    [i+1:]; entrada no open da barra i+1 (preenchimento realista).

O resultado e um `DecisionLog` PERSISTENTE e SEPARADO (default
data/ml_bootstrap.sqlite, para nao poluir o decision log de producao) que
`ml.retraining` consome (walk-forward + calibracao + gate de promocao).
Honesto por construcao: o mesmo vetorizador (`ml.dataset.row_features`) serve
bootstrap, treino e producao -> sem train/serve skew.

Uso:
    python -m ml.bootstrap                       # universo padrao, cache
    python -m ml.bootstrap --years 7 --max-hold 20
    python -m ml.bootstrap --db data/ml_bootstrap.sqlite --retrain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from feedback.decision_log import DecisionLog
from feedback.models import Decision, DecisionAction, MarketRegime, Outcome, OutcomeStatus
from feedback.regime import classify_regime
from fimathe.engine import FimatheEngine

logger = logging.getLogger("ml.bootstrap")

# Features da FimatheEngine gravadas no contexto (espelha integration.enricher._FIM_KEYS).
_FIM_KEYS = ("pcm_score", "dist_to_zn", "breakout_strength", "rsi", "adx")

# Nocional fixo por trade: normaliza o P&L para que acoes caras nao dominem a
# expectancy (o build_training_set usa realized_pnl). Retorno% * NOTIONAL.
NOTIONAL = 1000.0
STRATEGY_TAG = "fimathe_bootstrap"
REGIME_LOOKBACK = 60  # barras usadas para classificar o regime (causal)


@dataclass
class BootstrapStats:
    symbols: int = 0
    decisions: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    skipped_short: int = 0
    by_regime: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return self.wins / closed if closed else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "BOOTSTRAP DO DATASET DE ML — setups FimatheEngine, dados reais",
            "=" * 70,
            f"simbolos processados : {self.symbols}",
            f"decisoes rotuladas   : {self.decisions}",
            f"  wins               : {self.wins}",
            f"  losses             : {self.losses}",
            f"  (saidas por tempo) : {self.timeouts}",
            f"win-rate base        : {self.win_rate * 100:.1f}%",
        ]
        if self.by_regime:
            lines.append("por regime (n decisoes):")
            for reg, n in sorted(self.by_regime.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {reg:<12} {n:>6}")
        lines.append("=" * 70)
        return "\n".join(lines)


def _label_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry: float,
    stop: float,
    target: float,
    start: int,
    max_hold: int,
) -> tuple[float, bool] | None:
    """Simula a saida do trade LONG a partir da barra `start`.

    Retorna (return_pct, timeout) ou None se nao houver barras suficientes.
    Conservador: dentro de uma barra checa o STOP antes do alvo (pior caso).
    """
    n = len(close)
    end = min(start + max_hold, n - 1)
    if start > end or entry <= 0:
        return None
    for j in range(start, end + 1):
        if stop > 0 and low[j] <= stop:
            return (stop - entry) / entry, False
        if target > 0 and high[j] >= target:
            return (target - entry) / entry, False
    # expirou por tempo -> fecha no close da ultima barra do horizonte.
    return (close[end] - entry) / entry, True


def bootstrap_symbol(
    log: DecisionLog,
    symbol: str,
    df: pd.DataFrame,
    *,
    engine: FimatheEngine,
    max_hold: int,
    require_valid: bool,
    stats: BootstrapStats,
) -> None:
    """Gera decisoes rotuladas de UM simbolo no `DecisionLog`."""
    if len(df) < engine.swing_period + 40:
        return
    out = engine.process(df)
    idx = out.index
    ts = pd.to_datetime(idx, utc=True, errors="coerce")

    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float)
    signal = out["signal"].to_numpy(dtype=float)
    valid = out["setup_valid"].to_numpy(dtype=bool)
    stop_loss = out["stop_loss"].to_numpy(dtype=float)
    take_profit = out["take_profit_2"].to_numpy(dtype=float)
    n = len(close)

    for i in range(engine.swing_period, n - 1):
        if signal[i] != 1:  # so LONG (sistema e long-biased; ver FeedbackAgent)
            if signal[i] == -1:
                stats.skipped_short += 1
            continue
        if require_valid and not valid[i]:
            continue
        entry = float(open_[i + 1])  # fill realista: open da barra seguinte
        stop = float(stop_loss[i])
        target = float(take_profit[i])
        if not (np.isfinite(entry) and entry > 0):
            continue
        labeled = _label_trade(high, low, close, entry, stop, target, i + 1, max_hold)
        if labeled is None:
            continue
        return_pct, timeout = labeled
        if return_pct == 0.0:
            continue  # breakeven nao e exemplo de qualidade de setup

        created = ts[i]
        if created is pd.NaT:
            continue  # sem timestamp valido nao da p/ ordenar cronologicamente

        regime = classify_regime(close[max(0, i - REGIME_LOOKBACK) : i + 1].tolist())
        context = {k: float(out[k].iloc[i]) for k in _FIM_KEYS if k in out.columns and pd.notna(out[k].iloc[i])}
        strength = float(out["signal_strength"].iloc[i]) if pd.notna(out["signal_strength"].iloc[i]) else 0.0
        context["signal_strength"] = round(strength, 4)

        dec = Decision(
            strategy=STRATEGY_TAG,
            symbol=symbol,
            action=DecisionAction.BUY,
            regime=regime,
            reference_price=Decimal(str(round(entry, 6))),
            signal_strength=max(0.0, min(1.0, strength)),
            context=context,
            created_at=created.to_pydatetime(),
        )
        dec_id = log.record(dec)

        exit_price = entry * (1.0 + return_pct)
        status = OutcomeStatus.WIN if return_pct > 0 else OutcomeStatus.LOSS
        log.attach_outcome(
            dec_id,
            Outcome(
                status=status,
                entry_price=Decimal(str(round(entry, 6))),
                exit_price=Decimal(str(round(max(exit_price, 1e-6), 6))),
                realized_pnl=Decimal(str(round(return_pct * NOTIONAL, 4))),
                return_pct=float(return_pct),
                closed_at=created.to_pydatetime(),
                note=f"bootstrap {'timeout' if timeout else 'stop/alvo'} (R:R 1:2)",
            ),
        )

        stats.decisions += 1
        if status == OutcomeStatus.WIN:
            stats.wins += 1
        else:
            stats.losses += 1
        if timeout:
            stats.timeouts += 1
        stats.by_regime[regime.value] = stats.by_regime.get(regime.value, 0) + 1


def run_bootstrap(
    data: dict[str, pd.DataFrame],
    db_path: str | Path,
    *,
    max_hold: int = 20,
    require_valid: bool = True,
    engine: FimatheEngine | None = None,
) -> tuple[DecisionLog, BootstrapStats]:
    """Gera o dataset rotulado de TODO o universo num DecisionLog persistente.

    Recria o DB do zero (idempotente: rodar de novo da o mesmo dataset)."""
    path = Path(db_path)
    if path.exists():
        path.unlink()  # dataset reproduzivel: reconstroi do zero
    for suffix in ("-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            side.unlink()

    log = DecisionLog(db_path=path)
    engine = engine or FimatheEngine()
    stats = BootstrapStats()

    for symbol, df in data.items():
        try:
            bootstrap_symbol(
                log, symbol, df,
                engine=engine, max_hold=max_hold, require_valid=require_valid, stats=stats,
            )
            stats.symbols += 1
        except Exception as exc:  # um simbolo ruim nao derruba o universo
            logger.warning("Falha no bootstrap de %s: %s", symbol, exc)
    return log, stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap do dataset de ML (setups FimatheEngine, dados reais)")
    parser.add_argument("--years", type=int, default=7, help="anos de historico (default: todo o cache ~7a)")
    parser.add_argument("--no-crypto", action="store_true")
    parser.add_argument("--max-hold", type=int, default=20, help="horizonte maximo do trade em barras")
    parser.add_argument("--all-signals", action="store_true", help="inclui sinais sem setup_valid (R:R/breakout)")
    parser.add_argument("--db", default="data/ml_bootstrap.sqlite")
    parser.add_argument("--report-file", default="data/ml_bootstrap_report.txt")
    parser.add_argument("--retrain", action="store_true", help="roda ml.retraining no fim (walk-forward + promocao)")
    parser.add_argument("--model", default="data/models/setup_classifier.json", help="caminho do model store (com --retrain)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    from simulation.data import fetch_daily
    from simulation.run import DEFAULT_CRYPTO, DEFAULT_STOCKS

    crypto = [] if args.no_crypto else DEFAULT_CRYPTO
    logger.info("Carregando dados (cache): %d acoes, %d cripto...", len(DEFAULT_STOCKS), len(crypto))
    data = fetch_daily(DEFAULT_STOCKS, crypto, years=args.years)
    logger.info("Dados: %d simbolos.", len(data))

    log, stats = run_bootstrap(
        data, args.db, max_hold=args.max_hold, require_valid=not args.all_signals,
    )
    text = stats.summary()
    print(text)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("DecisionLog persistido em %s (%d decisoes).", args.db, stats.decisions)

    if args.retrain:
        from ml.retraining import run_retraining_from_log

        print("\n" + "=" * 70 + "\nRETREINO (walk-forward + calibracao + gate de promocao)\n" + "=" * 70)
        report = run_retraining_from_log(args.db, args.model)
        print(report.summary())
    log.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
