"""Spike alphalens-reloaded — IC cross-sectional de um sinal sobre dados REAIS.

Radar de stack (Fase 2): o QuantStats avalia RETORNOS de uma estrategia ja pronta;
ele nao diz se um SINAL tem poder preditivo *antes* de virar estrategia. Esse e o
buraco que o alphalens preenche (IC, decis/quantis, forward returns, turnover) —
a peca que faltava no "tribunal de alpha".

Aqui medimos um fator classico de cripto intraday — short-term reversal (o inverso
do retorno recente) — cross-section sobre as 6 perp da Binance ja cacheadas
(reusa simulation.crypto_intraday.load_symbol; klines 1m reamostrados p/ 1h). O
numero que importa e o IC (correlacao de Spearman do fator com o retorno futuro):
  |IC| < 0.02  -> sinal fraco/ruido (nao vale standalone)
  |IC| ~ 0.05  -> fraco mas explorravel como tilt
  |IC| > 0.10  -> forte para o padrao cross-sectional

So LE klines cacheados — nada de rede, broker ou ordem. Determinista.

CLI:
    uv run --with alphalens-reloaded python -m simulation.alphalens_spike
    # ou:  uv sync --extra research && uv run python -m simulation.alphalens_spike
"""

from __future__ import annotations

import pandas as pd

from simulation.crypto_intraday import MONTHS, UNIVERSE, load_symbol


def build_panel(
    symbols: list[str] = UNIVERSE,
    months: list[str] = MONTHS,
    *,
    rule: str = "1h",
) -> pd.DataFrame:
    """Painel wide de precos de fechamento (linhas=tempo, colunas=simbolo), 1h."""
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        df = load_symbol(sym, months)
        if df is None or df.empty:
            continue
        closes[sym] = df["close"].resample(rule).last()
    if not closes:
        raise FileNotFoundError("nenhum simbolo cacheado em data/binance_1m (rode o download)")
    return pd.DataFrame(closes).dropna(how="all").ffill().dropna()


def reversal_factor(prices: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """Short-term reversal: -(retorno das ultimas `lookback` barras), em formato long.

    Retorna Series MultiIndex (date, asset) — o contrato que o alphalens espera.
    """
    ret = prices.pct_change(lookback)
    factor = (-ret).stack()
    factor.index = factor.index.set_names(["date", "asset"])
    return factor.dropna()


def run_spike(
    symbols: list[str] = UNIVERSE,
    months: list[str] = MONTHS,
    *,
    lookback: int = 6,
    periods: tuple[int, ...] = (1, 6, 24),
    quantiles: int = 2,
) -> dict:
    """Roda o alphalens e devolve o IC medio por horizonte + diagnostico."""
    from alphalens.performance import factor_information_coefficient
    from alphalens.utils import get_clean_factor_and_forward_returns

    prices = build_panel(symbols, months)
    factor = reversal_factor(prices, lookback)

    data = get_clean_factor_and_forward_returns(
        factor, prices, periods=periods, quantiles=quantiles, max_loss=0.5
    )
    ic = factor_information_coefficient(data)
    ic_mean = {str(k): float(v) for k, v in ic.mean().to_dict().items()}

    best = max(ic_mean.values(), key=abs) if ic_mean else 0.0
    if abs(best) >= 0.10:
        verdict = f"FORTE: |IC| {abs(best):.3f} >= 0.10 cross-sectional"
    elif abs(best) >= 0.03:
        verdict = f"FRACO/TILT: |IC| {abs(best):.3f} (so vale como tilt ou stack)"
    else:
        verdict = f"RUIDO: |IC| {abs(best):.3f} < 0.03 (sem poder preditivo)"

    return {
        "n_assets": int(prices.shape[1]),
        "n_periods_time": int(prices.shape[0]),
        "n_obs_factor": int(len(data)),
        "ic_mean": ic_mean,
        "verdict": verdict,
    }


def main() -> None:
    r = run_spike()
    print("=== alphalens spike (reversal cross-sectional, Binance perp 1h reais) ===")
    print(f"universo           : {r['n_assets']} ativos x {r['n_periods_time']} barras 1h")
    print(f"observacoes (fator): {r['n_obs_factor']}")
    print(f"IC medio por horizonte: {{ {', '.join(f'{k}: {v:+.4f}' for k, v in r['ic_mean'].items())} }}")
    print(f"VEREDITO: {r['verdict']}")


if __name__ == "__main__":
    main()
