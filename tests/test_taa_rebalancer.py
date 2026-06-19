"""Testes do REBALANCEADOR VIVO do sleeve de TAA (strategies/taa_rebalancer.py).

O QUE GARANTE (o gate do Planner antes do track record):
  1. PARIDADE pesquisa<->producao: os pesos VIVOS para uma data batem EXATAMENTE
     com a linha dessa data no vetor de pesos do BACKTEST (sobre o painel inteiro).
     E o que prova que o que vai pro paper e o que foi auditado.
  2. SEM LOOK-AHEAD: o alvo de uma data passada NAO muda quando se adicionam (ou
     se perturbam) precos FUTUROS no painel.
  3. BANDA DE NAO-TRADE: desvio pequeno alvo-vs-atual NAO gera ordem (turnover baixo).
  4. ORDENS DE DIFERENCA corretas: lado/quantidade derivados de (alvo - atual) em
     notional; long-only, ETFs em lote inteiro.
  5. GATILHO MENSAL: so rebalanceia 1x por mes (estado persistido), com o gate de
     timing (>= 2o pregao do mes), sem look-ahead.
  6. ESPEC TRAVADA: config de producao = a campea do veredito (broad8_L6_N4_AGG).

A paridade real usa o cache yfinance de simulation.dual_momentum (data/taa_cache
ou data/beta_cache); se ausente, e pulada (skip), nunca falha por ambiente. Os
demais usam um painel SINTETICO deterministico — sem rede.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from broker.fake_broker import FakeBroker
from data.db import Database
from data.state_repo import StateRepository
from simulation.dual_momentum import ALL_TICKERS, load_panel
from strategies.taa_rebalancer import (
    DEFAULT_NO_TRADE_BAND,
    PROD_DEFENSIVE,
    PROD_LOOKBACK_M,
    PROD_TOP_N,
    STRATEGY_NAME,
    TRADABLE,
    TAARebalancer,
    is_rebalance_due,
    mark_rebalanced,
    production_weight_frame,
    target_weights_for_date,
)
from core.models import OrderSide


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _synthetic_panel(n_days: int = 1200, seed: int = 11) -> pd.DataFrame:
    """Painel SINTETICO deterministico no calendario de pregao (sem rede).

    Geometric random walk por ticker. Vols/drifts distintos para que o filtro de
    momentum (absoluto vs BIL + relativo top-N) produza uma cesta NAO-trivial e
    estavel (alguns ativos de risco entram, outros caem p/ o defensivo AGG)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n_days, tz="UTC")
    # BIL (cash proxy): drift minimo, vol baixissima. Defensivos: vol baixa.
    spec = {
        "SPY": (0.0005, 0.011), "QQQ": (0.0007, 0.014), "EFA": (0.0003, 0.011),
        "EEM": (0.0002, 0.015), "VNQ": (0.0003, 0.013), "DBC": (0.0000, 0.012),
        "GLD": (0.0003, 0.009), "TLT": (0.0001, 0.008), "IEF": (0.0001, 0.004),
        "AGG": (0.0001, 0.003), "BIL": (0.00005, 0.0005),
    }
    cols = {}
    for t in ALL_TICKERS:
        drift, vol = spec.get(t, (0.0003, 0.012))
        rets = rng.normal(drift, vol, size=n_days)
        cols[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(cols, index=idx)


@pytest.fixture()
def synthetic() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.fixture()
def real_panel() -> pd.DataFrame:
    panel = load_panel()
    if panel.empty:
        pytest.skip("cache de precos (data/taa_cache|beta_cache) ausente — pula paridade real.")
    return panel


@pytest.fixture()
def state() -> StateRepository:
    db = Database(":memory:")
    try:
        yield StateRepository(db)
    finally:
        db.close()


def _broker_at_weights(
    panel: pd.DataFrame, asof, weights_frac, equity=Decimal("100000")
) -> tuple[FakeBroker, Decimal]:
    """FakeBroker posicionado EXATAMENTE nos pesos `weights_frac` (por ticker), a
    precos = ultimo fechamento do painel; caixa ajustado p/ o equity total bater."""
    last = panel.loc[asof]
    prices = {t: Decimal(str(last[t])) for t in TRADABLE}
    broker = FakeBroker(cash=Decimal("0"), prices=prices)
    mv = Decimal("0")
    for tkr, w in weights_frac.items():
        price = Decimal(str(last[tkr]))
        qty = (Decimal(str(w)) * equity) / price
        if qty != 0:
            broker.seed_position(tkr, qty, price)
            mv += Decimal(str(w)) * equity
    broker._cash = equity - mv  # type: ignore[attr-defined]
    return broker, equity


# --------------------------------------------------------------------------- #
# 1. PARIDADE pesquisa <-> producao (o gate central)
# --------------------------------------------------------------------------- #
def test_parity_live_equals_backtest_synthetic(synthetic):
    """Os pesos VIVOS de uma data == a linha dessa data no vetor do BACKTEST sobre
    o painel inteiro. Tolerancia ao float (mesmo calculo, sem recomputo divergente)."""
    panel = synthetic
    wf_full = production_weight_frame(panel)
    for d in (panel.index[-1], panel.index[-100], panel.index[-400]):
        live = target_weights_for_date(panel, d)
        bt_row = wf_full.loc[d]
        assert live, f"esperava cesta nao-vazia em {d.date()}"
        for tkr, w_live in live.items():
            assert w_live == pytest.approx(float(bt_row[tkr]), abs=1e-12), (
                f"paridade quebrou em {d.date()} / {tkr}: vivo={w_live} bt={float(bt_row[tkr])}"
            )
        # e nenhum ticker com peso no backtest fica de fora do alvo vivo.
        bt_nonzero = {t for t in TRADABLE if abs(float(bt_row[t])) > 1e-12}
        assert bt_nonzero <= set(live), f"alvo vivo perdeu tickers em {d.date()}"


def test_parity_live_equals_backtest_real_panel(real_panel):
    """Paridade tambem no painel REAL (cache yfinance), varias datas."""
    panel = real_panel
    wf_full = production_weight_frame(panel)
    for d in (panel.index[-1], panel.index[-60], panel.index[-300]):
        live = target_weights_for_date(panel, d)
        bt_row = wf_full.loc[d]
        for tkr, w_live in live.items():
            assert w_live == pytest.approx(float(bt_row[tkr]), abs=1e-9), (
                f"paridade real quebrou em {d.date()} / {tkr}"
            )


def test_weights_sum_to_one_long_only(synthetic):
    """Espec travada: long-only, sem alavancagem — pesos em [0,1] e soma ~1."""
    panel = synthetic
    target = target_weights_for_date(panel, panel.index[-1])
    assert all(0.0 <= w <= 1.0 + 1e-9 for w in target.values())
    assert sum(target.values()) == pytest.approx(1.0, abs=1e-9)


def test_production_config_is_champion():
    """Os parametros de producao sao a config campea do veredito (broad8_L6_N4_AGG)."""
    assert PROD_LOOKBACK_M == 6
    assert PROD_TOP_N == 4
    assert PROD_DEFENSIVE == "AGG"
    assert PROD_DEFENSIVE in TRADABLE


# --------------------------------------------------------------------------- #
# 2. SEM LOOK-AHEAD
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_rows_do_not_change_past_target(synthetic):
    """O alvo de uma data passada nao muda se o painel contiver dados FUTUROS."""
    panel = synthetic
    asof = panel.index[-300]
    with_future = target_weights_for_date(panel, asof)
    without_future = target_weights_for_date(panel.loc[panel.index <= asof], asof)
    assert set(with_future) == set(without_future)
    for k in with_future:
        assert with_future[k] == pytest.approx(without_future[k], abs=1e-12)


def test_no_lookahead_future_price_perturbation_has_no_effect(synthetic):
    """Perturbar precos FUTUROS nao altera o alvo de uma data passada."""
    panel = synthetic
    asof = panel.index[-300]
    base = target_weights_for_date(panel, asof)
    perturbed = panel.copy()
    perturbed.iloc[-50:] = perturbed.iloc[-50:] * 4.0  # choque no futuro
    after = target_weights_for_date(perturbed, asof)
    assert set(base) == set(after)
    for k in base:
        assert base[k] == pytest.approx(after[k], abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. BANDA DE NAO-TRADE
# --------------------------------------------------------------------------- #
def test_no_trade_band_skips_when_at_target(synthetic):
    """Carteira ja no ALVO (desvio ~0 < banda) NAO gera ordem."""
    panel = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, asof)
    broker, equity = _broker_at_weights(panel, asof, target)
    plan = TAARebalancer(broker).compute_plan(panel, asof, equity=equity)
    assert plan.intents == [], f"carteira no alvo nao deveria gerar ordens: {plan.intents}"
    assert set(plan.skipped_in_band) >= {t for t, w in target.items() if w > 1e-6}


def test_below_band_holds_above_band_trades(synthetic):
    """Desvio < banda nao negocia; desvio > banda negocia o ativo."""
    panel = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, asof)
    tkr = max(target, key=lambda k: target[k])
    band = float(DEFAULT_NO_TRADE_BAND)

    cur = dict(target)
    cur[tkr] = target[tkr] - band * 0.5  # desvio 0.75% (< 1.5%)
    broker, equity = _broker_at_weights(panel, asof, cur)
    plan = TAARebalancer(broker).compute_plan(panel, asof, equity=equity)
    assert all(i.symbol != tkr for i in plan.intents)
    assert tkr in plan.skipped_in_band

    cur2 = dict(target)
    cur2[tkr] = target[tkr] - band * 2.0  # desvio 3% (> 1.5%)
    broker2, equity2 = _broker_at_weights(panel, asof, cur2)
    plan2 = TAARebalancer(broker2).compute_plan(panel, asof, equity=equity2)
    matching = [i for i in plan2.intents if i.symbol == tkr]
    assert matching, f"desvio acima da banda deveria gerar ordem em {tkr}"
    assert matching[0].side == OrderSide.BUY  # alvo > atual


# --------------------------------------------------------------------------- #
# 4. ORDENS DE DIFERENCA corretas (lado + quantidade)
# --------------------------------------------------------------------------- #
def test_diff_orders_from_zero_are_all_buys(synthetic):
    """Carteira ZERADA -> tudo vira BUY do alvo cheio; qty ~ peso*equity/preco."""
    panel = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, asof)
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"), prices={t: Decimal(str(last[t])) for t in TRADABLE}
    )
    equity = Decimal("100000")
    plan = TAARebalancer(broker).compute_plan(panel, asof, equity=equity)

    for tkr, w in target.items():
        if w < float(DEFAULT_NO_TRADE_BAND):
            continue
        order = next((i for i in plan.intents if i.symbol == tkr), None)
        assert order is not None, f"esperava ordem p/ {tkr} (peso-alvo {w:.3f})"
        assert order.side == OrderSide.BUY
        expected_qty = (Decimal(str(w)) * equity) / Decimal(str(last[tkr]))
        assert order.qty <= expected_qty + Decimal("1")  # round-down p/ inteiro
        assert order.qty > 0
    # so vende -> nada (carteira vazia); todas as ordens sao BUY (long-only).
    assert all(i.side == OrderSide.BUY for i in plan.intents)


def test_strategy_name_tag(synthetic):
    """As ordens carregam a tag de estrategia (auditoria)."""
    panel = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"), prices={t: Decimal(str(last[t])) for t in TRADABLE}
    )
    plan = TAARebalancer(broker).compute_plan(panel, asof, equity=Decimal("100000"))
    assert plan.intents
    assert all(i.strategy == STRATEGY_NAME for i in plan.intents)


# --------------------------------------------------------------------------- #
# 5. GATILHO MENSAL
# --------------------------------------------------------------------------- #
def test_monthly_trigger_fires_once_per_month(synthetic, state):
    """Devido na 1a vez do mes; nao-devido apos marcar; devido de novo no mes seguinte."""
    panel = synthetic
    # asof bem dentro do mes (>= 2o pregao) p/ o gate de timing passar.
    asof = panel.index[-1]
    assert is_rebalance_due(state, asof, panel=panel) is True
    mark_rebalanced(state, asof)
    assert is_rebalance_due(state, asof, panel=panel) is False

    # um mes a frente (mesmo painel; outra data de mes diferente)
    next_month = asof + pd.offsets.MonthBegin(1) + pd.Timedelta(days=3)
    # garante >= 2 pregoes no mes de next_month dentro do painel sintetico:
    panel_ext = panel.reindex(panel.index.union(pd.bdate_range(asof, next_month, tz="UTC")))
    assert is_rebalance_due(state, next_month, panel=panel_ext) is True


def test_timing_gate_defers_first_trading_day(synthetic):
    """No 1o pregao do mes o rebalance e DEFERIDO (pesos do backtest so vigem no
    2o pregao); a partir do 2o, libera."""
    panel = synthetic
    months = panel.index.tz_localize(None).to_period("M")
    # acha o 1o e 2o pregao de algum mes coberto pelo painel.
    last_m = months[-1]
    days_in_last_m = panel.index[months == last_m]
    assert len(days_in_last_m) >= 2
    first_day, second_day = days_in_last_m[0], days_in_last_m[1]
    assert is_rebalance_due(None, first_day, panel=panel) is False   # deferido
    assert is_rebalance_due(None, second_day, panel=panel) is True   # liberado


def test_force_overrides_trigger(synthetic, state):
    """force=True ignora estado e timing (uso operacional/manual)."""
    panel = synthetic
    asof = panel.index[0]  # 1o pregao -> normalmente deferido
    mark_rebalanced(state, asof)
    assert is_rebalance_due(state, asof, panel=panel) is False
    assert is_rebalance_due(state, asof, panel=panel, force=True) is True
