"""Benchmark vivo: SPY (buy&hold) e 60/40 sintetico (SPY/IEF), capturados em
PARALELO ao NAV da estrategia, no MESMO dia.

Por que capturar em paralelo (anti-cherry-picking): se o benchmark fosse
reconstruido depois, daria para "escolher" retroativamente o benchmark que
favorece a estrategia. Gravando SPY e o NAV 60/40 no mesmo snapshot diario, a
comparacao fica honesta e travada no tempo.

REUSO DA CONVENCAO DO BACKTEST: os pesos 60/40 vem de
`simulation.beta_portfolio.SIXTY_FORTY` (= {"SPY":0.60, "IEF":0.40}), o mesmo
"benchmark B" do backtest. Aqui mantemos o NAV 60/40 de forma INCREMENTAL e SEM
LOOK-AHEAD: cada dia marca-se a mercado com o preco do PROPRIO dia, e o
rebalance para 60/40 ocorre no 1o pregao de cada mes (mesma cadencia do backtest).

Determinismo: dado o ultimo estado (NAV, pesos efetivos, precos) + os precos de
hoje, o proximo NAV 60/40 e funcao pura — reproduzivel por um terceiro.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Reusa a convencao de pesos do backtest (benchmark B).
from simulation.beta_portfolio import SIXTY_FORTY

# Tickers que compoem o 60/40. Mantido aqui para o caso de o backtest evoluir;
# a fonte da VERDADE dos pesos continua sendo SIXTY_FORTY.
SPY = "SPY"
IEF = "IEF"

# NAV inicial do benchmark sintetico (base 1.0 — uma "cota" do 60/40).
INITIAL_NAV = Decimal("1")


@dataclass(frozen=True)
class SixtyFortyState:
    """Estado incremental do NAV 60/40. Serializavel para persistir em `state`.

    `nav`: NAV atual (cota do 60/40).
    `units`: quantidade de "cotas" de cada ticker (NAV = sum(units*price)).
    `last_month`: 'YYYY-MM' do ultimo rebalance (rebalanceia quando o mes vira).
    `last_prices`: ultimos precos vistos (para marcar dias sem rebalance).
    """

    nav: Decimal
    units: dict[str, Decimal]
    last_month: str | None
    last_prices: dict[str, Decimal]


def _month_of(day: str) -> str:
    return day[:7]  # 'YYYY-MM'


def _rebalanced_units(nav: Decimal, prices: dict[str, Decimal]) -> dict[str, Decimal]:
    """Distribui o NAV nos tickers conforme os pesos 60/40, dados os precos do dia."""
    units: dict[str, Decimal] = {}
    for ticker, weight in SIXTY_FORTY.items():
        price = prices.get(ticker)
        if price is None or price <= 0:
            units[ticker] = Decimal("0")
            continue
        units[ticker] = (nav * Decimal(str(weight))) / Decimal(str(price))
    return units


def initial_state(day: str, prices: dict[str, Decimal]) -> SixtyFortyState:
    """Estado do 1o dia: compra 60/40 a base 1.0 com os precos do dia."""
    nav = INITIAL_NAV
    units = _rebalanced_units(nav, prices)
    return SixtyFortyState(
        nav=nav, units=units, last_month=_month_of(day), last_prices=dict(prices)
    )


def step(state: SixtyFortyState | None, day: str, prices: dict[str, Decimal]) -> SixtyFortyState:
    """Avanca o NAV 60/40 um dia.

    1. Marca a mercado: NAV_hoje = sum(units * preco_hoje) com os precos do dia
       (sem look-ahead — usa apenas o fechamento do proprio dia).
    2. Se virou o mes desde o ultimo rebalance, re-divide o NAV em 60/40 aos
       precos de hoje (mesma cadencia do backtest: 1o pregao do mes).

    Precos ausentes para um ticker => mantem as `units` antigas e usa o ultimo
    preco conhecido para a marcacao (sem inventar dado).
    """
    if state is None:
        return initial_state(day, prices)

    # Precos efetivos: usa o de hoje quando presente; senao o ultimo conhecido.
    eff_prices: dict[str, Decimal] = dict(state.last_prices)
    for ticker in SIXTY_FORTY:
        p = prices.get(ticker)
        if p is not None and p > 0:
            eff_prices[ticker] = p

    # (1) marca a mercado com as cotas atuais.
    nav = Decimal("0")
    for ticker, units in state.units.items():
        price = eff_prices.get(ticker)
        if price is not None and price > 0:
            nav += units * price
    if nav <= 0:
        nav = state.nav  # degenerado (sem precos) -> mantem

    month = _month_of(day)
    units_out = state.units
    last_month = state.last_month
    # (2) rebalance mensal.
    if last_month is None or month != last_month:
        units_out = _rebalanced_units(nav, eff_prices)
        last_month = month

    return SixtyFortyState(
        nav=nav, units=units_out, last_month=last_month, last_prices=eff_prices
    )
