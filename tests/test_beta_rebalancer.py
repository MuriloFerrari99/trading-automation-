"""Testes do REBALANCEADOR VIVO da estrategia de beta (strategies/beta_rebalancer.py).

O QUE GARANTE (o gate do Planner antes do track record):
  1. PARIDADE pesquisa<->producao: os pesos VIVOS para uma data batem EXATAMENTE
     com a linha dessa data no vetor de pesos do BACKTEST (sobre o painel inteiro).
     E o que prova que o que vai pro paper e o que foi auditado.
  2. SEM LOOK-AHEAD: o alvo de uma data passada NAO muda quando se adicionam (ou
     se perturbam) precos FUTUROS no painel.
  3. BANDA DE NAO-TRADE: desvio pequeno alvo-vs-atual NAO gera ordem (turnover baixo).
  4. ORDENS DE DIFERENCA corretas: lado/quantidade derivados de (alvo - atual) em
     notional, com mapeamento de simbolo cripto (BTC-USD -> BTC/USD).
  5. GATILHO MENSAL: so rebalanceia 1x por mes (estado persistido), sem look-ahead.
  6. RESPEITA kill switch (via Executor) e PortfolioRiskGuard (halt / per-symbol).

Os testes de paridade e no-look-ahead usam o painel REAL (cache yfinance de
data/beta_cache); se o cache nao existir, sao pulados (skip), nunca falham por
ambiente. Os demais usam um painel SINTETICO deterministico — sem rede.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from broker.fake_broker import FakeBroker
from core.kill_switch import KillSwitch
from core.models import OrderIntent, OrderSide, OrderType
from data.audit_log import AuditLog
from data.db import Database
from data.order_repo import OrderRepository
from data.state_repo import StateRepository
from data.trade_logger import TradeLogger
from agents.executor import Executor
from risk.portfolio_guard import PortfolioRiskGuard
from simulation.beta_portfolio import UNIVERSE, _rebalance_mask, load_panel
from strategies.beta_rebalancer import (
    DEFAULT_NO_TRADE_BAND,
    PRODUCTION_LEVERAGE,
    BetaRebalancer,
    beta_sleeve_equity,
    beta_sleeve_nav,
    beta_sleeve_pnl,
    is_rebalance_due,
    mark_rebalanced,
    production_weight_frame,
    resolve_sleeve_capital,
    resolve_sleeve_nav,
    target_weights_for_date,
    to_broker_symbol,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _synthetic_panel(n_days: int = 900, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Painel SINTETICO deterministico no calendario de pregao (sem rede).

    Geometric random walk por ativo, vol distinta por classe (bond < equity <
    metal < crypto) p/ que o 1/vol produza pesos nao-triviais e estaveis."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days, tz="UTC")
    vol_by_class = {"equity": 0.012, "bond": 0.004, "metal": 0.010, "crypto": 0.045}
    classes = {a.ticker: a.klass for a in UNIVERSE}
    cols = {}
    for a in UNIVERSE:
        daily_vol = vol_by_class[a.klass]
        rets = rng.normal(0.0003, daily_vol, size=n_days)
        price = 100.0 * np.exp(np.cumsum(rets))
        cols[a.ticker] = price
    panel = pd.DataFrame(cols, index=idx)
    return panel, classes


@pytest.fixture()
def synthetic():
    return _synthetic_panel()


@pytest.fixture()
def real_panel():
    panel, classes = load_panel()
    if panel.empty:
        pytest.skip("cache de precos (data/beta_cache) ausente — pula paridade real.")
    return panel, classes


@pytest.fixture()
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture()
def state(db) -> StateRepository:
    return StateRepository(db)


# --------------------------------------------------------------------------- #
# 1. PARIDADE pesquisa <-> producao (o gate central)
# --------------------------------------------------------------------------- #
def test_parity_live_equals_backtest_real_panel(real_panel):
    """Para varias datas, os pesos VIVOS == a linha do vetor de pesos do BACKTEST
    sobre o painel INTEIRO. Tolerancia apertada (deve ser identico ao float)."""
    panel, classes = real_panel
    wf_full = production_weight_frame(panel, classes)

    # amostra de datas ao longo do historico recente (onde a cesta e diversificada).
    sample_dates = [panel.index[-1], panel.index[-50], panel.index[-250], panel.index[-600]]
    for d in sample_dates:
        live = target_weights_for_date(panel, classes, d)
        bt_row = wf_full.loc[d]
        for tkr, w_live in live.items():
            assert w_live == pytest.approx(float(bt_row[tkr]), abs=1e-9), (
                f"paridade quebrou em {d.date()} / {tkr}: vivo={w_live} backtest={float(bt_row[tkr])}"
            )


def test_parity_live_equals_backtest_synthetic(synthetic):
    """Paridade tambem no painel sintetico (independe do cache de rede)."""
    panel, classes = synthetic
    wf_full = production_weight_frame(panel, classes)
    for d in (panel.index[-1], panel.index[-100], panel.index[-400]):
        live = target_weights_for_date(panel, classes, d)
        bt_row = wf_full.loc[d]
        maxdiff = max(abs(live[k] - float(bt_row[k])) for k in live)
        assert maxdiff < 1e-12, f"paridade sintetica quebrou em {d.date()} (maxdiff={maxdiff})"


def test_production_leverage_scales_gross(synthetic):
    """A alavancagem de producao escala o gross: gross(2.0x) == 2.0 * gross(1x)."""
    panel, classes = synthetic
    d = panel.index[-1]
    w1 = target_weights_for_date(panel, classes, d, leverage=1.0)
    wL = target_weights_for_date(panel, classes, d, leverage=PRODUCTION_LEVERAGE)
    g1 = sum(abs(v) for v in w1.values())
    gL = sum(abs(v) for v in wL.values())
    assert gL == pytest.approx(PRODUCTION_LEVERAGE * g1, rel=1e-9)
    assert PRODUCTION_LEVERAGE == 2.0  # espec travada (perfil 2.0x escolhido)


# --------------------------------------------------------------------------- #
# 2. SEM LOOK-AHEAD
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_rows_do_not_change_past_target(synthetic):
    """O alvo de uma data passada nao muda se o painel contiver dados FUTUROS."""
    panel, classes = synthetic
    asof = panel.index[-300]

    with_future = target_weights_for_date(panel, classes, asof)
    without_future = target_weights_for_date(panel.loc[panel.index <= asof], classes, asof)
    for k in with_future:
        assert with_future[k] == pytest.approx(without_future[k], abs=1e-12)


def test_no_lookahead_future_price_perturbation_has_no_effect(synthetic):
    """Perturbar um preco FUTURO nao altera o alvo de uma data passada."""
    panel, classes = synthetic
    asof = panel.index[-300]
    base = target_weights_for_date(panel, classes, asof)

    perturbed_panel = panel.copy()
    perturbed_panel.iloc[-1] = perturbed_panel.iloc[-1] * 3.0  # choque no futuro
    perturbed = target_weights_for_date(perturbed_panel, classes, asof)
    for k in base:
        assert base[k] == pytest.approx(perturbed[k], abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. BANDA DE NAO-TRADE
# --------------------------------------------------------------------------- #
def _broker_at_weights(panel, classes, asof, weights_frac, equity=Decimal("100000")):
    """FakeBroker posicionado EXATAMENTE nos pesos `weights_frac` (por ticker do
    backtest), a precos = ultimo fechamento do painel."""
    last = panel.loc[asof]
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    broker = FakeBroker(cash=Decimal("0"), prices=prices)
    for tkr, w in weights_frac.items():
        price = Decimal(str(last[tkr]))
        qty = (Decimal(str(w)) * equity) / price
        if qty != 0:
            broker.seed_position(to_broker_symbol(tkr), qty, price)
    # ajusta o caixa p/ o equity total bater (cash + market value = equity).
    mv = sum(
        (Decimal(str(w)) * equity for w in weights_frac.values()), Decimal("0")
    )
    broker._cash = equity - mv  # type: ignore[attr-defined]
    return broker, prices, equity


def test_no_trade_band_skips_small_deviation(synthetic):
    """Se a carteira ja esta no ALVO (desvio ~0 < banda), NAO gera ordem."""
    panel, classes = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)

    broker, prices, equity = _broker_at_weights(panel, classes, asof, target)
    reb = BetaRebalancer(broker)
    plan = reb.compute_plan(panel, asof, equity=equity)

    assert plan.intents == [], f"carteira no alvo nao deveria gerar ordens: {plan.intents}"
    # todos os ativos com peso-alvo nao-trivial caem na banda.
    assert set(plan.skipped_in_band) >= {t for t, w in target.items() if abs(w) > 1e-6}


def test_below_band_deviation_does_not_trade_but_above_does(synthetic):
    """Desvio < banda nao negocia; desvio > banda negocia."""
    panel, classes = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)
    # escolhe o ativo de maior peso-alvo p/ manipular.
    tkr = max(target, key=lambda k: target[k])

    band = float(DEFAULT_NO_TRADE_BAND)
    # (a) desvio MENOR que a banda -> sem ordem nesse ativo.
    cur = dict(target)
    cur[tkr] = target[tkr] - (band * 0.5)  # 0.75% de desvio (< 1.5%)
    broker, _, equity = _broker_at_weights(panel, classes, asof, cur)
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=equity)
    assert all(i.symbol != to_broker_symbol(tkr) for i in plan.intents)
    assert tkr in plan.skipped_in_band

    # (b) desvio MAIOR que a banda -> ordem nesse ativo (compra, pois alvo > atual).
    cur2 = dict(target)
    cur2[tkr] = target[tkr] - (band * 2.0)  # 3% de desvio (> 1.5%)
    broker2, _, equity2 = _broker_at_weights(panel, classes, asof, cur2)
    plan2 = BetaRebalancer(broker2).compute_plan(panel, asof, equity=equity2)
    sym = to_broker_symbol(tkr)
    matching = [i for i in plan2.intents if i.symbol == sym]
    assert matching, f"desvio acima da banda deveria gerar ordem em {sym}"
    assert matching[0].side == OrderSide.BUY


# --------------------------------------------------------------------------- #
# 4. ORDENS DE DIFERENCA corretas (lado + quantidade)
# --------------------------------------------------------------------------- #
def test_diff_orders_side_and_qty(synthetic):
    """Ordem de diferenca: BUY quando alvo>atual, SELL quando alvo<atual, e a qty
    aproxima |alvo-atual|*equity/preco."""
    panel, classes = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)
    last = panel.loc[asof]

    # carteira ZERADA -> tudo vira BUY do alvo cheio.
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    equity = Decimal("100000")
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=equity)

    for tkr, w in target.items():
        if abs(w) < float(DEFAULT_NO_TRADE_BAND):
            continue
        sym = to_broker_symbol(tkr)
        order = next((i for i in plan.intents if i.symbol == sym), None)
        assert order is not None, f"esperava ordem p/ {sym} (peso-alvo {w:.3f})"
        assert order.side == OrderSide.BUY  # de zero, todo alvo positivo e compra
        expected_qty = (Decimal(str(w)) * equity) / Decimal(str(last[tkr]))
        # equities arredondam p/ inteiro (round down); cripto fraciona.
        assert order.qty <= expected_qty + Decimal("1")
        assert order.qty > 0


def test_crypto_symbol_mapping(synthetic):
    """Cripto do backtest (BTC-USD/ETH-USD) vira simbolo de corretora (BTC/USD)."""
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=Decimal("100000"))
    symbols = {i.symbol for i in plan.intents}
    # nenhum simbolo com sufixo -USD do yfinance deve vazar p/ a ordem.
    assert not any(s.endswith("-USD") for s in symbols)
    # se BTC tem peso, deve aparecer como BTC/USD.
    if abs(target_weights_for_date(panel, classes, asof).get("BTC-USD", 0.0)) > float(
        DEFAULT_NO_TRADE_BAND
    ):
        assert "BTC/USD" in symbols


def test_sell_when_overweight(synthetic):
    """Posicao ACIMA do alvo gera SELL do excedente."""
    panel, classes = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)
    tkr = max(target, key=lambda k: target[k])

    over = dict(target)
    over[tkr] = target[tkr] + 0.10  # 10pp acima do alvo
    broker, _, equity = _broker_at_weights(panel, classes, asof, over)
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=equity)
    sym = to_broker_symbol(tkr)
    order = next((i for i in plan.intents if i.symbol == sym), None)
    assert order is not None and order.side == OrderSide.SELL


# --------------------------------------------------------------------------- #
# 5. GATILHO MENSAL
# --------------------------------------------------------------------------- #
def test_monthly_trigger_due_once_per_month(state):
    asof = pd.Timestamp("2025-03-10", tz="UTC")
    # primeira vez: devido.
    assert is_rebalance_due(state, asof) is True
    mark_rebalanced(state, asof)
    # mesmo mes: nao devido.
    assert is_rebalance_due(state, pd.Timestamp("2025-03-25", tz="UTC")) is False
    # mes seguinte: devido de novo.
    assert is_rebalance_due(state, pd.Timestamp("2025-04-01", tz="UTC")) is True


def test_monthly_trigger_no_state_is_due():
    assert is_rebalance_due(None, pd.Timestamp("2025-03-10", tz="UTC")) is True


def test_plan_if_due_returns_none_when_not_due(synthetic, state):
    panel, classes = synthetic
    asof = panel.index[-1]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(panel.loc[asof][t])) for t in classes},
    )
    reb = BetaRebalancer(broker, state=state)
    mark_rebalanced(state, asof)  # ja rebalanceado este mes
    assert reb.plan_if_due(panel, asof) is None
    # force ignora o gatilho.
    assert reb.plan_if_due(panel, asof, force=True) is not None


# --------------------------------------------------------------------------- #
# 6. GUARDS: kill switch (via Executor) e PortfolioRiskGuard
# --------------------------------------------------------------------------- #
def _executor(broker, tmp_path, monkeypatch):
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    db = Database(":memory:")
    return (
        Executor(
            broker,
            TradeLogger(db),
            KillSwitch(tmp_path / "KILL_SWITCH"),
            order_repo=OrderRepository(db),
            audit=AuditLog(db),
        ),
        db,
    )


def test_kill_switch_blocks_rebalance_execution(synthetic, tmp_path, monkeypatch):
    """Com o kill switch engajado, NENHUMA ordem do rebalance e enviada."""
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=Decimal("100000"))
    assert plan.intents  # ha ordens a enviar

    ks = KillSwitch(tmp_path / "KILL_SWITCH")
    ks.engage("teste rebalance")
    db = Database(":memory:")
    ex = Executor(broker, TradeLogger(db), ks, order_repo=OrderRepository(db), audit=AuditLog(db))
    results = ex.execute_many(plan.intents)
    assert results == []
    assert broker.submitted == []  # nada chegou ao broker
    db.close()


def test_guard_halt_blocks_new_buys(synthetic):
    """Com o guard em HALT, compras (aumentam risco) sao BLOQUEADAS no plano;
    vendas (reduzem risco) continuam permitidas."""
    panel, classes = synthetic
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)

    # carteira: 1 ativo MUITO acima do alvo (gera SELL) e o resto zerado (geraria BUY).
    tkr_over = max(target, key=lambda k: target[k])
    weights = {tkr_over: target[tkr_over] + 0.20}
    broker, _, equity = _broker_at_weights(panel, classes, asof, weights)

    guard = PortfolioRiskGuard(start_equity=equity, max_dd_pct=Decimal("0.10"))
    guard._halt("halt de teste")  # type: ignore[attr-defined]
    assert guard.trading_halted

    plan = BetaRebalancer(broker, guard=guard).compute_plan(panel, asof, equity=equity)
    # a venda do ativo sobre-alocado passa (reduz risco).
    sells = [i for i in plan.intents if i.side == OrderSide.SELL]
    buys = [i for i in plan.intents if i.side == OrderSide.BUY]
    assert any(i.symbol == to_broker_symbol(tkr_over) for i in sells)
    assert buys == []  # nenhuma compra sob halt
    assert plan.blocked  # registrou os bloqueios


def test_guard_per_symbol_cap_blocks_overweight_buy(synthetic):
    """Teto por simbolo do guard bloqueia uma COMPRA cujo peso-alvo excede o teto."""
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    # teto baixo (5%) p/ forcar bloqueio de qualquer alvo grande, carteira zerada.
    guard = PortfolioRiskGuard(
        start_equity=Decimal("100000"),
        max_per_symbol_pct=Decimal("0.05"),
        max_heat_pct=Decimal("10"),
    )
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    plan = BetaRebalancer(broker, guard=guard).compute_plan(panel, asof, equity=Decimal("100000"))
    target = target_weights_for_date(panel, classes, asof)
    big = {t for t, w in target.items() if w > 0.05}
    # nenhum ativo com peso-alvo > 5% deve ter ordem de COMPRA (todos bloqueados).
    bought_syms = {i.symbol for i in plan.intents if i.side == OrderSide.BUY}
    for t in big:
        assert to_broker_symbol(t) not in bought_syms
    assert plan.blocked


def test_equity_zero_aborts_plan(synthetic):
    """Equity <= 0 aborta o plano (fail-safe), sem ordens."""
    panel, classes = synthetic
    asof = panel.index[-1]
    broker = FakeBroker(cash=Decimal("0"), prices={})
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=Decimal("0"))
    assert plan.intents == []


# =========================================================================== #
# PERFIL 2.0x — CALIBRACAO DOS GUARDS (a classe de bug das 4 rodadas: o guard
# nao pode pegar a operacao NORMAL, so anomalia). Anchora nos dados da fronteira
# (vol-alvo 10% x 2.0x): concentracao max legitima por nome ~200%; DD ao-vivo
# ~-36/-39% (halt -43%); vol nominal ~20% (daily-loss -8%).
# =========================================================================== #
def test_2x_per_symbol_cap_allows_legit_200pct_blocks_above_210(synthetic):
    """PER-SYMBOL (paridade-critico): com o teto de PRODUCAO (210%), uma COMPRA
    cujo peso-alvo e a concentracao MAXIMA legitima a 2.0x (~200% — o 1/vol pondo
    toda a alavancagem num nome de baixa vol) PASSA; so um alvo ALEM de 210%
    (anomalia) e bloqueado. Se o teto clipasse <=200%, quebraria a paridade
    vivo==backtest (o backtest gera ~200% por nome em dias de baixa-vol)."""
    from config.risk import get_beta_guard_settings

    bs = get_beta_guard_settings()
    guard = PortfolioRiskGuard(
        start_equity=Decimal("100000"),
        max_per_symbol_pct=bs.max_per_symbol_pct,   # 210% de producao
        max_heat_pct=bs.max_portfolio_heat_pct,
    )
    # concentracao legitima de 200% (toda a alavancagem 2.0x num nome) PASSA.
    ok, reason = guard.can_open(Decimal("2.00"), Decimal("0"))
    assert ok, f"200% (concentracao legitima a 2.0x) NAO deveria ser bloqueado: {reason}"
    # exatamente no teto (210%) ainda PASSA (a condicao e estritamente '>').
    ok_edge, _ = guard.can_open(Decimal("2.10"), Decimal("0"))
    assert ok_edge, "210% (no teto) deveria passar"
    # ALEM de 210% (anomalia) e bloqueado.
    ok_over, reason_over = guard.can_open(Decimal("2.11"), Decimal("0"))
    assert not ok_over and "simbolo" in reason_over, reason_over


def test_2x_per_symbol_does_not_clip_real_panel_concentration(real_panel):
    """PER-SYMBOL (no painel REAL): em NENHUM dia a concentracao legitima por nome
    a 2.0x passa do teto de producao (210%) — i.e., o guard NUNCA clipa a operacao
    normal e a paridade vivo==backtest fica preservada. (O maximo medido e ~200%.)"""
    from config.risk import get_beta_guard_settings

    panel, classes = real_panel
    bs = get_beta_guard_settings()
    wf = production_weight_frame(panel, classes)  # ja em 2.0x (PRODUCTION_LEVERAGE)
    max_per_symbol = float(wf.abs().max().max())
    # a concentracao legitima maxima e ~200% e fica ABAIXO do teto de 210%.
    assert max_per_symbol <= float(bs.max_per_symbol_pct), (
        f"concentracao legitima {max_per_symbol:.3f} excede o teto {bs.max_per_symbol_pct} "
        "-> o guard cliparia a operacao normal e quebraria a paridade"
    )
    # e o teto tem folga real sobre a concentracao legitima (nao e justo demais).
    assert max_per_symbol > 1.6, (
        f"a 2.0x a concentracao por nome ({max_per_symbol:.3f}) deve passar de 160% "
        "(o teto antigo de 1.5x) — senao o teste nao prova que 160% clipava"
    )


def test_2x_normal_rebalance_does_not_block_any_legit_buy(real_panel):
    """PERFIL 2.0x (end-to-end, painel real): uma rebalanceada NORMAL a 2.0x com o
    guard de PRODUCAO (boot sadio) NAO bloqueia NENHUMA compra legitima por
    per-symbol — todo peso-alvo do book 2.0x cabe sob o teto de 210%. Prova que o
    guard recalibrado nao trava a operacao normal do novo perfil."""
    from config.risk import get_beta_guard_settings

    panel, classes = real_panel
    asof = panel.index[-1]
    last = panel.loc[asof]
    bs = get_beta_guard_settings()
    # guard de producao, boot sadio (start=peak=NAV), carteira ZERADA => tudo BUY.
    guard = PortfolioRiskGuard(
        start_equity=Decimal("1000000"),
        daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct,
        max_per_symbol_pct=bs.max_per_symbol_pct,
        max_heat_pct=bs.max_portfolio_heat_pct,
        peak_equity=Decimal("1000000"),
    )
    guard.update(Decimal("1000000"))
    assert not guard.trading_halted
    # buying_power generoso (book 2.0x => gross>1, caixa fica negativo como em margem).
    broker = FakeBroker(
        cash=Decimal("1000000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        buying_power=Decimal("9000000"),
    )
    plan = BetaRebalancer(broker, guard=guard).compute_plan(
        panel, asof, equity=Decimal("1000000")
    )
    # nenhuma compra legitima foi bloqueada pelo per-symbol (operacao normal a 2.0x).
    assert not any("simbolo" in b for b in plan.blocked), (
        f"o guard de producao bloqueou compra(s) legitima(s) a 2.0x: {plan.blocked}"
    )
    # e o book NAO nasce vazio: ha ordens (a 1a rebalanceada constroi o book).
    assert [i for i in plan.intents if i.side == OrderSide.BUY], (
        f"rebalanceada normal a 2.0x deveria gerar BUYs; blocked={plan.blocked}"
    )


# --------------------------------------------------------------------------- #
# 7. Integracao end-to-end do plug vivo (flag), no-op por padrao
# --------------------------------------------------------------------------- #
def test_live_runner_is_noop_when_flag_off(synthetic, monkeypatch, tmp_path):
    """run_beta_rebalance_cycle e NO-OP quando a flag esta desligada (OFF por
    padrao: nem env var, nem `.env`)."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # sem `.env` no cwd => flag default OFF (legado)
    panel, classes = synthetic
    broker = FakeBroker(cash=Decimal("100000"), prices={})
    db = Database(":memory:")
    ex = Executor(broker, TradeLogger(db), KillSwitch(tmp_path / "K"), order_repo=OrderRepository(db))
    out = beta_live.run_beta_rebalance_cycle(broker, ex, panel=panel)
    assert out["enabled"] is False
    assert out["executed"] == 0
    assert broker.submitted == []
    db.close()


def test_live_runner_executes_when_forced(synthetic, monkeypatch, tmp_path, state):
    """Com force=True (ou flag on), o runner calcula e executa o rebalance e
    marca o mes como rebalanceado."""
    from strategies import beta_live
    from strategies.beta_rebalancer import is_rebalance_due

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    db = Database(":memory:")
    ex = Executor(
        broker, TradeLogger(db), KillSwitch(tmp_path / "K"),
        order_repo=OrderRepository(db), audit=AuditLog(db),
    )
    out = beta_live.run_beta_rebalance_cycle(
        broker, ex, state=state, panel=panel, asof=asof, force=True
    )
    assert out["enabled"] is True and out["due"] is True
    assert out["executed"] >= 1
    assert broker.submitted  # ordens chegaram ao broker
    # marcou o mes -> nao mais devido.
    assert is_rebalance_due(state, asof) is False
    db.close()


# --------------------------------------------------------------------------- #
# 7b. FLAG DE ARME: lida via pydantic-settings (.env + env var), OFF por padrao
#     Regressao do go-live: beta_live_enabled() lia os.environ CRU, entao
#     BETA_LIVE_ENABLED=1 no .env era INERTE e o daemon bootava no caminho
#     LEGADO em silencio. Os 3 casos canonicos provados abaixo.
# --------------------------------------------------------------------------- #
def test_arm_flag_off_by_default(monkeypatch, tmp_path):
    """OFF POR PADRAO: sem env var e sem `.env` no cwd, beta_live_enabled() e
    False (caminho vivo do beta inteiramente inerte = comportamento legado)."""
    from strategies.beta_live import beta_live_enabled

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    # cwd sem `.env` => pydantic-settings nao acha a flag em lugar nenhum => default.
    monkeypatch.chdir(tmp_path)
    assert beta_live_enabled() is False  # OFF por padrao (sem flag em lugar nenhum)


def test_arm_flag_off_build_app_is_legacy(monkeypatch):
    """OFF == LEGADO em build_app: com a flag desligada, build_app devolve o
    orquestrador LEGADO (LocalOrchestrator do factory), NAO o wrapper
    _BetaRebalanceOrchestrator. Roda do cwd real (config/watchlist + data
    resolvem); DB em memoria p/ nao tocar o sqlite real. Flag forcada OFF via env
    var (precedencia sobre o `.env`) p/ ser deterministico independ. do `.env`."""
    import main as main_mod
    from data.db import Database
    from orchestration.local_orchestrator import LocalOrchestrator
    from strategies.beta_live import beta_live_enabled

    # env var=0 vence o `.env`: flag deterministicamente OFF (sem chdir, p/ os
    # arquivos de config relativos do build_app continuarem resolvendo).
    monkeypatch.setenv("BETA_LIVE_ENABLED", "0")
    assert beta_live_enabled() is False
    # DB em memoria: build_app chama Database() (path default) — redireciona p/ nao
    # poluir data/trading.sqlite real.
    monkeypatch.setattr(main_mod, "Database", lambda *a, **k: Database(":memory:"))

    broker = FakeBroker(cash=Decimal("100000"), prices={})
    _monitor, orchestrator = main_mod.build_app(broker, orchestrator_name="local")
    assert isinstance(orchestrator, LocalOrchestrator)
    assert not isinstance(orchestrator, main_mod._BetaRebalanceOrchestrator), (
        "flag off NAO deve armar o beta: build_app tem de ficar no caminho legado"
    )


def test_arm_flag_on_via_env_var_inline(monkeypatch, tmp_path):
    """ON VIA ENV VAR INLINE (como o daemon ja rodando foi armado): a env var
    arma a flag mesmo sem `.env`. Comportamento de hoje, preservado."""
    from strategies.beta_live import beta_live_enabled

    monkeypatch.chdir(tmp_path)  # sem `.env` no cwd: prova que a env var sozinha arma
    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    assert beta_live_enabled() is True


def test_arm_flag_on_via_dotenv_file(monkeypatch, tmp_path):
    """ON VIA `.env` (O CASO QUE ESTAVA QUEBRADO): com BETA_LIVE_ENABLED=1 num
    `.env` no cwd e SEM env var, beta_live_enabled() agora retorna True. Antes do
    fix (os.environ cru) isto era False e o daemon bootava no legado em silencio."""
    from strategies.beta_live import beta_live_enabled

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)  # nao ha env var inline
    (tmp_path / ".env").write_text("BETA_LIVE_ENABLED=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # pydantic-settings le o `.env` do cwd
    assert beta_live_enabled() is True


def test_arm_flag_env_var_overrides_dotenv(monkeypatch, tmp_path):
    """PRECEDENCIA: a env var inline tem precedencia sobre o `.env` (igual ao
    resto da config). `.env`=0 + env var=1 => armado."""
    from strategies.beta_live import beta_live_enabled

    (tmp_path / ".env").write_text("BETA_LIVE_ENABLED=0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")  # env var vence o `.env`
    assert beta_live_enabled() is True


# =========================================================================== #
# MUST-FIX da auditoria de producao (data/code_audit_beta_production.txt)
# =========================================================================== #
def _month_boundary(panel):
    """(r_prev, r, r_next) p/ um boundary de mes recente do painel: r = 1o pregao
    do mes (dia do _rebalance_mask), r_next = r+1 (2o pregao = dia em que os pesos
    entram em vigor no backtest), r_prev = ultimo pregao do mes anterior."""
    idx = panel.index
    mask = _rebalance_mask(idx)
    rebal_days = idx[mask.values]
    for r in reversed(rebal_days):
        pos = idx.get_loc(r)
        if 0 < pos < len(idx) - 1:
            return idx[pos - 1], r, idx[pos + 1]
    raise AssertionError("painel sem boundary de mes utilizavel")


# --- MF-2: gatilho mensal em r+1 (NAO em r); NAV vivo == backtest -----------
def test_mf2_trigger_deferred_on_r_and_fires_on_r_plus_1(real_panel, state):
    """MF-2 (ALTO): no DIA do gatilho do backtest (r, 1o pregao do mes) o caminho
    vivo NAO rebalanceia (os pesos ainda sao do mes anterior). So fica devido em
    r+1, o dia em que os pesos do backtest entram em vigor (.shift(1))."""
    panel, classes = real_panel
    r_prev, r, r_next = _month_boundary(panel)
    mark_rebalanced(state, r_prev)  # mes anterior ja resolvido

    # r: deferido pelo gate de TIMING (so 1 pregao do mes ate r).
    assert is_rebalance_due(state, r, panel=panel) is False, (
        f"vivo NAO deveria rebalancear em r={r.date()} (pesos ainda do mes anterior)"
    )
    # r+1: devido (2o pregao do mes; pesos novos em vigor).
    assert is_rebalance_due(state, r_next, panel=panel) is True, (
        f"vivo deveria rebalancear em r+1={r_next.date()}"
    )


def test_mf2_live_target_equals_backtest_r_plus_1_not_stale_r(real_panel):
    """MF-2: o alvo NEGOCIADO no vivo (asof=r+1) == wf.loc[r+1] do backtest, e
    DIFERE do alvo defasado wf.loc[r] (o do mes anterior, que o codigo antigo
    negociava). Esta e a paridade de TIMING (nao so da funcao de peso)."""
    panel, classes = real_panel
    _r_prev, r, r_next = _month_boundary(panel)
    wf = production_weight_frame(panel, classes)

    live_r1 = target_weights_for_date(panel, classes, r_next)
    bt_r1 = wf.loc[r_next]
    bt_r = wf.loc[r]

    # vivo (r+1) bate com backtest (r+1), bit-a-bit.
    for tkr, w in live_r1.items():
        assert w == pytest.approx(float(bt_r1[tkr]), abs=1e-9), (
            f"timing-parity quebrou em {tkr}: vivo(r+1)={w} bt(r+1)={float(bt_r1[tkr])}"
        )
    # e DIFERE do alvo defasado (r) — senao o teste nao prova nada (o boundary tem
    # mudanca real de pesos).
    gap = max(abs(live_r1[k] - float(bt_r[k])) for k in live_r1)
    assert gap > 1e-6, "boundary sem mudanca de pesos — escolha outro p/ provar o lag"


def test_mf2_nav_path_live_matches_backtest_over_boundary(real_panel, state, tmp_path, monkeypatch):
    """MF-2 (caminho de NAV): no DIA do gatilho, rodando o fluxo vivo COMPLETO
    (run_beta_rebalance_cycle, force p/ ignorar a janela de close), a carteira
    resultante fica nos pesos wf[r+1] do backtest — i.e., o NAV vivo passa a seguir
    o MESMO vetor que o backtest mantem em [r+1, proximo rebalance].

    Partimos posicionados em wf[r-1] (estado do mes anterior) e provamos que apos o
    rebalance de r+1 a carteira esta em wf[r+1], nao no alvo velho."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # flag off no cwd; force=True exercita o caminho vivo
    panel, classes = real_panel
    r_prev, r, r_next = _month_boundary(panel)
    wf = production_weight_frame(panel, classes)

    # broker posicionado nos pesos do mes anterior (wf[r-1]) aos precos de r+1.
    w_prev = {k: float(v) for k, v in wf.loc[r_prev].items()}
    equity = Decimal("1000000")
    last = panel.loc[r_next]
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    # buying_power generoso: o book e 1.5x (gross>1 => caixa fica negativo, como
    # numa conta de margem real). Sem isso, o cash-account do FakeBroker barraria
    # as compras por "buying power insuficiente" (artefato do fake, nao do produto).
    broker = FakeBroker(cash=Decimal("0"), prices=prices, buying_power=Decimal("5000000"))
    for tkr, w in w_prev.items():
        if abs(w) < 1e-9:
            continue
        qty = (Decimal(str(w)) * equity) / Decimal(str(last[tkr]))
        broker.seed_position(to_broker_symbol(tkr), qty, Decimal(str(last[tkr])))
    broker._cash = equity - sum(  # type: ignore[attr-defined]
        (Decimal(str(w)) * equity for w in w_prev.values()), Decimal("0")
    )

    db = Database(":memory:")
    ex = Executor(
        broker, TradeLogger(db), KillSwitch(tmp_path / "K"),
        order_repo=OrderRepository(db), audit=AuditLog(db),
    )
    # force=True: prova o resultado do rebalance de r+1 ignorando a janela de close.
    out = beta_live.run_beta_rebalance_cycle(
        broker, ex, state=state, panel=panel, asof=r_next, force=True
    )
    assert out["due"] is True

    # carteira final (pesos atuais lidos do broker) deve casar com wf[r+1].
    final_w, _ = BetaRebalancer(broker)._current_weights(equity)
    bt_r1 = wf.loc[r_next]
    bt_r = wf.loc[r]
    # tolerancia: ordens de equities arredondam p/ inteiro -> erro <= banda.
    band = float(DEFAULT_NO_TRADE_BAND)
    for tkr in classes:
        assert abs(final_w[tkr] - float(bt_r1[tkr])) <= band, (
            f"NAV vivo nao convergiu p/ wf[r+1] em {tkr}: vivo={final_w[tkr]} bt(r+1)={float(bt_r1[tkr])}"
        )
    # e ao menos um ativo se afastou do alvo VELHO (r) por mais que a banda —
    # prova que negociamos para os pesos NOVOS, nao ficamos nos do mes anterior.
    moved = max(abs(final_w[t] - float(bt_r[t])) for t in classes)
    assert moved > band, "rebalance nao saiu do alvo do mes anterior (lag nao corrigido)"
    db.close()


def test_mf2_synthetic_trigger_timing(synthetic, state):
    """MF-2 tambem no painel sintetico (independe do cache): defere em r, dispara
    em r+1."""
    panel, _classes = synthetic
    r_prev, r, r_next = _month_boundary(panel)
    mark_rebalanced(state, r_prev)
    assert is_rebalance_due(state, r, panel=panel) is False
    assert is_rebalance_due(state, r_next, panel=panel) is True


# --- MF-3: cripto EXECUTA (peso != 0) --------------------------------------- #
def _panel_with_crypto_weight(seed=11):
    """Painel sintetico em que a cripto recebe peso-alvo > banda (vol baixa o
    suficiente p/ o 1/vol alocar BTC/ETH). Garante cripto != 0 p/ testar execucao."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 900
    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    # vol de cripto BAIXA aqui (ao contrario do default) p/ o vol-target NAO zerar
    # a cripto pela vol — assim ela ganha peso > banda e gera ORDEM.
    vol_by_class = {"equity": 0.020, "bond": 0.012, "metal": 0.018, "crypto": 0.010}
    classes = {a.ticker: a.klass for a in UNIVERSE}
    cols = {}
    for a in UNIVERSE:
        rets = rng.normal(0.0002, vol_by_class[a.klass], size=n)
        cols[a.ticker] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(cols, index=idx), classes


def test_mf3_crypto_executes_when_weight_nonzero():
    """MF-3 (cripto != 0): com peso-alvo de cripto acima da banda, o rebalancer gera
    ordem com simbolo de corretora 'BTC/USD' (com barra), qty FRACIONARIA, e a
    ordem chega ao broker (executa). Antes, cripto caia em 'sem preco' e nunca
    negociava."""
    panel, classes = _panel_with_crypto_weight()
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)
    # pre-condicao do teste: a cripto realmente tem peso > banda neste painel.
    assert abs(target.get("BTC-USD", 0.0)) > float(DEFAULT_NO_TRADE_BAND), (
        f"painel de teste sem peso de cripto suficiente: {target.get('BTC-USD')}"
    )

    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
    )
    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=Decimal("100000"))

    btc_orders = [i for i in plan.intents if i.symbol == "BTC/USD"]
    assert btc_orders, f"esperava ordem de cripto BTC/USD; intents={[i.symbol for i in plan.intents]}"
    o = btc_orders[0]
    assert o.side == OrderSide.BUY
    assert o.qty > 0
    # cripto e fracionavel: a qty NAO e inteira (lote 0.0001).
    assert o.qty != o.qty.to_integral_value(), f"qty de cripto deveria ser fracionaria: {o.qty}"

    # e a ordem EXECUTA no broker (chega ao submitted).
    db = Database(":memory:")
    ex = Executor(broker, TradeLogger(db), KillSwitch("/tmp/__no_kill__"), order_repo=OrderRepository(db))
    results = ex.execute_many(plan.intents)
    assert any(r.symbol == "BTC/USD" for r in results)
    assert any(i.symbol == "BTC/USD" for i in broker.submitted)
    db.close()


def test_mf3_crypto_position_symbol_normalized_no_double_buy():
    """MF-3c: se o broker devolve a posicao de cripto SEM barra ('BTCUSD'), o
    rebalancer deve reconhece-la como BTC/USD e NAO comprar de novo (sem o fix, o
    lookup erra, le peso 0 e DOBRA a posicao).

    Aqui simulamos isso via _to_position do AlpacaBroker (a normalizacao real) e
    confirmamos que o simbolo normalizado casa com to_broker_symbol('BTC-USD')."""
    from broker.alpaca_broker import AlpacaBroker

    class _RawPos:
        symbol = "BTCUSD"  # forma SEM barra que a Alpaca pode devolver
        qty = "0.5"
        avg_entry_price = "40000"
        current_price = "60000"

    pos = AlpacaBroker._to_position(_RawPos())
    assert pos.symbol == "BTC/USD" == to_broker_symbol("BTC-USD"), pos.symbol

    # consequencia pratica: posicionado no alvo de cripto, NAO gera nova compra.
    panel, classes = _panel_with_crypto_weight()
    asof = panel.index[-1]
    target = target_weights_for_date(panel, classes, asof)
    w_btc = target["BTC-USD"]
    last = panel.loc[asof]
    equity = Decimal("100000")
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    broker = FakeBroker(cash=Decimal("0"), prices=prices)
    qty = (Decimal(str(w_btc)) * equity) / Decimal(str(last["BTC-USD"]))
    # seed na forma COM barra (como o _to_position normalizado entrega ao sistema).
    broker.seed_position("BTC/USD", qty, Decimal(str(last["BTC-USD"])))
    broker._cash = equity - qty * Decimal(str(last["BTC-USD"]))  # type: ignore[attr-defined]

    plan = BetaRebalancer(broker).compute_plan(panel, asof, equity=equity)
    btc_orders = [i for i in plan.intents if i.symbol == "BTC/USD"]
    assert btc_orders == [], f"cripto ja no alvo NAO deveria gerar nova ordem: {btc_orders}"
    assert "BTC-USD" in plan.skipped_in_band


# --- MF-4: guards isolados (sleeve do beta, sem halt cruzado) --------------- #
def test_mf4_beta_sleeve_equity_excludes_non_beta_positions():
    """MF-4: o equity do sleeve do beta soma SO o mv das posicoes do universo do
    beta — posicoes do book de ACOES (ex.: AAPL) NAO entram."""
    broker = FakeBroker(
        cash=Decimal("50000"),
        prices={"SPY": Decimal("500"), "AAPL": Decimal("200"), "BTC/USD": Decimal("60000")},
    )
    broker.seed_position("SPY", Decimal("100"), Decimal("400"))     # beta: 50k
    broker.seed_position("AAPL", Decimal("300"), Decimal("150"))    # acoes: 60k (FORA)
    broker.seed_position("BTC/USD", Decimal("0.5"), Decimal("50000"))  # beta: 30k
    sleeve = beta_sleeve_equity(broker)
    assert sleeve == Decimal("80000"), sleeve  # 50k + 30k, AAPL excluido


def test_mf4_stock_crash_does_not_halt_beta():
    """MF-4: um tombo do book de ACOES (que derruba o equity TOTAL) NAO halta o
    guard do beta, porque o guard do beta mede o SLEEVE do beta (saudavel)."""
    from config.risk import get_beta_guard_settings
    from risk.portfolio_guard import PortfolioRiskGuard

    bs = get_beta_guard_settings()
    # Sleeve do beta estavel em 100k (SPY); o book de acoes (AAPL) e que desaba.
    broker = FakeBroker(
        cash=Decimal("0"),
        prices={"SPY": Decimal("100"), "AAPL": Decimal("100")},
    )
    broker.seed_position("SPY", Decimal("1000"), Decimal("100"))   # beta sleeve 100k
    broker.seed_position("AAPL", Decimal("1000"), Decimal("100"))  # acoes 100k

    sleeve_start = beta_sleeve_equity(broker)
    guard = PortfolioRiskGuard(
        start_equity=sleeve_start,
        daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct,
        peak_equity=sleeve_start,
    )
    guard.update(sleeve_start)
    assert not guard.trading_halted

    # AAPL despenca -40% (equity TOTAL cairia ~20%), mas o SLEEVE do beta (SPY) nao.
    broker.set_price("AAPL", "60")
    guard.update(beta_sleeve_equity(broker))  # alimenta com o SLEEVE, nao o total
    assert not guard.trading_halted, "tombo das ACOES nao deve haltar o beta (MF-4)"


def test_mf4_beta_crash_does_not_halt_stocks():
    """MF-4 (reciproco): um tombo do book de BETA nao deve haltar o guard das
    ACOES, que mede o equity da conta menos... — aqui provamos que o guard das
    acoes (start_equity proprio) so engaja com base no SEU equity-base. Modelamos
    os dois guards com bases SEPARADAS e confirmamos o nao-cruzamento."""
    from config.risk import get_beta_guard_settings
    from risk.portfolio_guard import PortfolioRiskGuard

    bs = get_beta_guard_settings()
    # Guard das ACOES: base = equity das acoes (100k), inalterado quando o beta cai.
    stock_guard = PortfolioRiskGuard(start_equity=Decimal("100000"), max_dd_pct=Decimal("0.20"), peak_equity=Decimal("100000"))
    # Guard do BETA: base = sleeve do beta (100k), DD-halt do perfil 2.0x (-43%).
    beta_guard = PortfolioRiskGuard(start_equity=Decimal("100000"), max_dd_pct=bs.max_drawdown_pct, peak_equity=Decimal("100000"))

    # Beta sangra -45% (sleeve 55k), PIOR que o halt de -43% -> halta o BETA...
    beta_guard.update(Decimal("55000"))
    assert beta_guard.trading_halted
    # ...mas o guard das ACOES, alimentado com o equity das ACOES (intacto, 100k),
    # NAO halta.
    stock_guard.update(Decimal("100000"))
    assert not stock_guard.trading_halted, "tombo do BETA nao deve haltar as ACOES (MF-4)"


def test_mf4_beta_guard_uses_own_state_keys(monkeypatch, tmp_path):
    """MF-4 + NOVO-1: _build_beta_guard usa chaves de estado PROPRIAS ('beta:risk_*'),
    sem pisar nas chaves do guard das acoes ('risk_*'), E ancora o guard no NAV do
    SLEEVE (capital alocado + P&L), nao no equity TOTAL da conta.

    Cenario com os DOIS books coexistindo (acoes 500k + SPY do beta): carvamos um
    sleeve limpo via BETA_SLEEVE_CAPITAL=100000 (a remediacao do latente do MF-4).
    A SPY esta no preco de entrada => P&L 0 => NAV do sleeve = 100k (o capital
    alocado), NAO os 600k do equity total."""
    import main as main_mod
    from data.audit_log import AuditLog
    from data.db import Database
    from datetime import datetime, timezone

    # Capital ALOCADO ao sleeve = 100k (independe do equity da conta de 600k).
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL", "100000")

    db = Database(":memory:")
    st = StateRepository(db)
    audit = AuditLog(db)
    # broker com SPY (beta) no preco de entrada (P&L 0) e uma posicao de acoes a parte.
    broker = FakeBroker(cash=Decimal("0"), prices={"SPY": Decimal("100"), "AAPL": Decimal("100")})
    broker.seed_position("SPY", Decimal("1000"), Decimal("100"))
    broker.seed_position("AAPL", Decimal("5000"), Decimal("100"))  # acoes (nao-beta)

    main_mod._build_beta_guard(broker, st, audit)
    today = datetime.now(timezone.utc).date().isoformat()
    # chaves do BETA existem e refletem o NAV do SLEEVE (100k alocado + 0 P&L),
    # NAO o equity total (600k).
    assert st.get_decimal(f"beta:risk_start_equity:{today}") == Decimal("100000")
    assert st.get_decimal("beta:risk_peak_equity") == Decimal("100000")
    # chaves das ACOES NAO foram criadas por _build_beta_guard.
    assert st.get_decimal(f"risk_start_equity:{today}") is None
    assert st.get_decimal("risk_peak_equity") is None
    db.close()


# --- MF-5: rebalance perto do fechamento ------------------------------------ #
def test_mf5_rebalance_deferred_when_not_near_close(real_panel, state, tmp_path, monkeypatch):
    """MF-5: quando o mes e devido mas estamos LONGE do fechamento (1o tick da
    manha), o rebalance NAO executa (e o mes NAO e marcado) — sera tentado num
    tick perto do close, p/ o fill MARKET casar com o close de r+1."""
    from datetime import datetime, timezone
    from strategies import beta_live

    # flag ON p/ o ciclo chegar ao gate de close (force bypassaria a janela).
    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    panel, classes = real_panel
    _r_prev, _r, r_next = _month_boundary(panel)
    last = panel.loc[r_next]
    # FakeBroker: relogio aberto, mas fechamento daqui a 4h (longe da janela 20min).
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        market_open=True,
        buying_power=Decimal("5000000"),
        today=r_next.tz_convert("UTC").date() if r_next.tzinfo else r_next.date(),
    )
    # timestamp do FakeBroker e 12:00 UTC; next_close 16:00 UTC => 240 min p/ close.
    broker.set_clock(next_close=datetime(broker._today.year, broker._today.month, broker._today.day, 16, 0, tzinfo=timezone.utc))

    db = Database(":memory:")
    ex = Executor(broker, TradeLogger(db), KillSwitch(tmp_path / "K"), order_repo=OrderRepository(db), audit=AuditLog(db))
    out = beta_live.run_beta_rebalance_cycle(broker, ex, state=state, panel=panel, asof=r_next)
    assert out.get("deferred_to_close") is True, out
    assert out["executed"] == 0
    assert broker.submitted == []
    # mes NAO foi marcado -> ainda devido (vai tentar perto do close).
    assert is_rebalance_due(state, r_next, panel=panel) is True
    db.close()


def test_mf5_rebalance_executes_near_close(real_panel, state, tmp_path, monkeypatch):
    """MF-5: dentro da janela perto do fechamento, o rebalance devido EXECUTA e
    marca o mes."""
    from datetime import datetime, timezone
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")  # flag ON p/ exercitar o caminho vivo real
    panel, classes = real_panel
    _r_prev, _r, r_next = _month_boundary(panel)
    last = panel.loc[r_next]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        market_open=True,
        buying_power=Decimal("5000000"),
        today=r_next.tz_convert("UTC").date() if r_next.tzinfo else r_next.date(),
    )
    # next_close daqui a 10 min (12:10 UTC) => DENTRO da janela de 20 min.
    broker.set_clock(next_close=datetime(broker._today.year, broker._today.month, broker._today.day, 12, 10, tzinfo=timezone.utc))

    db = Database(":memory:")
    ex = Executor(broker, TradeLogger(db), KillSwitch(tmp_path / "K"), order_repo=OrderRepository(db), audit=AuditLog(db))
    out = beta_live.run_beta_rebalance_cycle(broker, ex, state=state, panel=panel, asof=r_next)
    assert out["due"] is True
    assert out.get("deferred_to_close") is not True
    assert out["executed"] >= 1
    assert broker.submitted
    assert is_rebalance_due(state, r_next, panel=panel) is False  # mes marcado
    db.close()


# =========================================================================== #
# NOVO-1 (CRITICO) — BOOT em conta FLAT (sleeve=0), o estado de go-live.
# O guard do beta NAO pode nascer haltado: a base e o NAV do sleeve = capital
# ALOCADO (caixa) + P&L do beta, nao o market value das posicoes (0 no boot).
# (Re-auditoria: data/code_audit_beta_production_v2.txt, Parte B / NOVO-1.)
# =========================================================================== #
def test_sleeve_nav_is_allocated_capital_when_flat():
    """NOVO-1: em conta FLAT (sem posicoes do beta), o NAV do sleeve = capital
    alocado (P&L 0), e POSITIVO — nao 0 (que era o market value, a causa do halt)."""
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("500")})
    assert beta_sleeve_pnl(broker) == Decimal("0")          # sem posicoes => P&L 0
    assert beta_sleeve_equity(broker) == Decimal("0")        # mv das posicoes = 0
    assert beta_sleeve_nav(broker, Decimal("100000")) == Decimal("100000")  # alocado + 0


def test_resolve_sleeve_capital_absolute_or_none(monkeypatch):
    """NOVO-3 (simplificacao): resolve_sleeve_capital devolve o CAIXA ABSOLUTO FIXO
    quando setado (> 0), ou None em paper SO-BETA (default). O ramo antigo
    pct*equity foi REMOVIDO (era a fonte do double-count: equity ja inclui P&L)."""
    broker = FakeBroker(cash=Decimal("250000"), prices={})  # equity 250k, conta flat
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    # default (sem capital absoluto) => None: paper so-beta usa o equity da conta.
    assert resolve_sleeve_capital(broker) is None
    # pct legado NAO carva mais um sleeve (DEPRECATED p/ NOVO-3): segue None.
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL_PCT", "0.4")
    assert resolve_sleeve_capital(broker) is None
    # absoluto > 0 vence e devolve o caixa FIXO.
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL", "75000")
    assert resolve_sleeve_capital(broker) == Decimal("75000")


def test_resolve_sleeve_nav_paper_only_is_account_equity(monkeypatch):
    """NOVO-3: em paper SO-BETA (default), o NAV do sleeve E o equity da conta
    (P&L contado UMA vez), NAO pct*equity + P&L (que duplicava o P&L)."""
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    # conta flat: NAV = equity = caixa (positivo, guard nasce sadio).
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("100")})
    assert resolve_sleeve_nav(broker) == Decimal("100000")
    # totalmente investido (gasta o caixa) e SPY -30%: NAV == equity da conta (70k),
    # NAO 40k (o double-count antigo: equity 70k - P&L 30k).
    broker.submit_order(
        OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("1000"),
                    order_type=OrderType.MARKET, strategy="t")
    )
    broker.set_price("SPY", "70")
    assert broker.get_account().equity == Decimal("70000")  # real DD -30%
    assert resolve_sleeve_nav(broker) == Decimal("70000")   # 1x, sem double-count


def test_resolve_sleeve_nav_absolute_is_fixed_cash_plus_pnl(monkeypatch):
    """NOVO-3: com capital ABSOLUTO (book de acoes junto), o NAV = caixa_fixo +
    P&L (uma vez). O caixa fixo NAO inclui P&L, entao nao ha double-count."""
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL", "100000")
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    # SPY do beta investido + uma posicao de ACOES inflando o equity total.
    broker = FakeBroker(
        cash=Decimal("0"),
        prices={"SPY": Decimal("100"), "AAPL": Decimal("100")},
    )
    broker.seed_position("SPY", Decimal("1000"), Decimal("100"))   # beta, P&L 0
    broker.seed_position("AAPL", Decimal("5000"), Decimal("100"))  # acoes (nao-beta)
    # P&L 0 => NAV do sleeve = 100k (o caixa fixo), NAO o equity total (600k).
    assert resolve_sleeve_nav(broker) == Decimal("100000")
    # SPY -30% => P&L beta -30k => NAV = 70k (so o sleeve, isolado das acoes).
    broker.set_price("SPY", "70")
    assert resolve_sleeve_nav(broker) == Decimal("70000")


def test_novo1_flat_boot_guard_not_halted(monkeypatch):
    """NOVO-1 (o nucleo): no BOOT em conta FLAT (sleeve=0, estado de go-live),
    _build_beta_guard NAO halta o guard. Antes: start coagido a 1 + update(0) =>
    daily_pl -100% => halt pegajoso. Agora: NAV = capital alocado (positivo) =>
    guard sadio (sem halt)."""
    import main as main_mod
    from data.audit_log import AuditLog
    from data.db import Database

    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    # Conta FLAT: tem caixa (100k) mas ZERO posicoes do beta.
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("500")})
    db = Database(":memory:")
    st = StateRepository(db)
    guard = main_mod._build_beta_guard(broker, st, AuditLog(db))
    assert guard.trading_halted is False, (
        f"guard NAO deve nascer haltado em conta flat: {guard.halt_reason}"
    )
    assert guard.halt_reason is None
    db.close()


def test_novo1_flat_boot_first_rebalance_generates_orders(synthetic, monkeypatch, tmp_path):
    """NOVO-1 (a prova de go-live): com o guard do BOOT FLAT, a PRIMEIRA
    rebalanceada GERA ordens (BUYs) e NAO bloqueia tudo. Antes, o halt pegajoso
    deixava plan.intents == [] e plan.blocked == todos os longs -> book vazio."""
    import main as main_mod
    from data.audit_log import AuditLog
    from data.db import Database

    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]

    # Conta FLAT (go-live): caixa, zero posicoes; buying_power generoso (book 1.5x).
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        buying_power=Decimal("5000000"),
    )
    db = Database(":memory:")
    st = StateRepository(db)
    guard = main_mod._build_beta_guard(broker, st, AuditLog(db))
    assert guard.trading_halted is False  # pre-condicao: guard sadio

    # 1a rebalanceada com esse guard (equity=None => dimensiona pelo NAV do sleeve).
    reb = BetaRebalancer(broker, guard=guard)
    plan = reb.compute_plan(panel, asof)
    buys = [i for i in plan.intents if i.side == OrderSide.BUY]
    assert buys, f"1a rebalanceada deveria gerar BUYs; blocked={plan.blocked}"
    # nenhum bloqueio por 'trading halted' (o halt fantasma sumiu).
    assert not any("halt" in b.lower() for b in plan.blocked), plan.blocked
    # o dimensionamento usou o NAV do sleeve (capital alocado), POSITIVO.
    assert plan.equity == Decimal("100000")

    # e as ordens EXECUTAM (chegam ao broker) — o book e construido, nao vazio.
    ex = Executor(
        broker, TradeLogger(db), KillSwitch(tmp_path / "K"),
        order_repo=OrderRepository(db), audit=AuditLog(db),
    )
    results = ex.execute_many(plan.intents)
    assert results and broker.submitted, "o book do beta deve ser construido no go-live"
    db.close()


def test_novo1_flat_boot_via_live_runner_executes(synthetic, monkeypatch, tmp_path, state):
    """NOVO-1 end-to-end: pelo runner vivo COMPLETO (run_beta_rebalance_cycle,
    force), partindo de conta FLAT + guard do boot, o rebalance EXECUTA ordens
    (track record nao nasce vazio)."""
    import main as main_mod
    from data.audit_log import AuditLog
    from data.db import Database
    from strategies import beta_live

    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # flag off no cwd; force=True exercita o caminho vivo
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        buying_power=Decimal("5000000"),
    )
    db = Database(":memory:")
    st = StateRepository(db)
    guard = main_mod._build_beta_guard(broker, st, AuditLog(db))
    assert guard.trading_halted is False

    ex = Executor(
        broker, TradeLogger(db), KillSwitch(tmp_path / "K"),
        order_repo=OrderRepository(db), audit=AuditLog(db),
    )
    out = beta_live.run_beta_rebalance_cycle(
        broker, ex, state=state, guard=guard, panel=panel, asof=asof, force=True
    )
    assert out["due"] is True
    assert out["executed"] >= 1, f"go-live nao deveria gravar book vazio: {out}"
    assert out["blocked"] == [] or not any("halt" in b.lower() for b in out["blocked"])
    assert broker.submitted
    db.close()


def test_novo1_real_sleeve_loss_still_halts(monkeypatch):
    """NOVO-1 (o lado complementar): apos PERDAS REAIS do sleeve alem do limite, o
    guard SIM halta. O fix nao desarma o guard — so remove o halt FANTASMA do boot
    flat. Uma posicao do beta que cai abaixo do limite diario engaja o halt."""
    from config.risk import get_beta_guard_settings
    from risk.portfolio_guard import PortfolioRiskGuard

    bs = get_beta_guard_settings()
    allocated = Decimal("100000")
    # boot flat -> guard sadio.
    broker = FakeBroker(cash=allocated, prices={"SPY": Decimal("100")})
    start_nav = beta_sleeve_nav(broker, allocated)
    guard = PortfolioRiskGuard(
        start_equity=start_nav,
        daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct,
        peak_equity=start_nav,
    )
    guard.update(start_nav)
    assert not guard.trading_halted  # boot flat: sadio

    # SPY do beta cai 10% (entrada 100 -> 90); P&L = -10% do alvo. Modelamos uma
    # posicao grande p/ a perda do SLEEVE passar do limite diario (-6%).
    broker.seed_position("SPY", Decimal("1000"), Decimal("100"))  # entrada 100
    broker.set_price("SPY", "90")  # -10% => P&L -10k sobre NAV 100k = -10%
    guard.update(beta_sleeve_nav(broker, allocated))
    assert guard.trading_halted, "perda REAL do sleeve alem do limite deve haltar"


def test_novo1_sizing_uses_sleeve_not_total_account_equity(synthetic, monkeypatch):
    """NOVO-1 / latente MF-4: o DIMENSIONAMENTO usa o NAV do SLEEVE (capital
    alocado + P&L), NAO o equity TOTAL da conta. Com uma posicao de ACOES inflando
    o equity total, o beta dimensiona sobre o sleeve carvado (config), nao sobre a
    conta inteira."""
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]

    # Carva um sleeve de 100k via config absoluto (os dois books coexistem).
    monkeypatch.setenv("BETA_SLEEVE_CAPITAL", "100000")

    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    prices["AAPL"] = Decimal("500")
    broker = FakeBroker(cash=Decimal("100000"), prices=prices, buying_power=Decimal("9000000"))
    broker.seed_position("AAPL", Decimal("1000"), Decimal("500"))  # 500k de acoes (nao-beta)
    # equity total da conta = 100k caixa + 500k AAPL = 600k.
    assert broker.get_account().equity == Decimal("600000")

    # equity=None => o rebalancer resolve o NAV do sleeve do config (100k), nao 600k.
    reb = BetaRebalancer(broker)
    plan = reb.compute_plan(panel, asof)
    assert plan.equity == Decimal("100000"), (
        f"dimensionou sobre o equity total ({plan.equity}) em vez do sleeve (100k)"
    )
    # sanity: o notional-alvo de um nome com peso w e ~ w * 100k (sleeve), nao w*600k.
    for tkr, w in plan.target_weights.items():
        if abs(w) < float(DEFAULT_NO_TRADE_BAND):
            continue
        sym = to_broker_symbol(tkr)
        order = next((i for i in plan.intents if i.symbol == sym), None)
        if order is None:
            continue
        notional = order.qty * Decimal(str(last[tkr]))
        # com sleeve 100k o notional <= ~peso*100k (+1 unidade de arredondamento);
        # com 600k seria 6x maior. Conferimos o teto do sleeve.
        assert notional <= Decimal(str(w)) * Decimal("100000") + Decimal(str(last[tkr])), (
            f"{sym}: notional {notional} > teto do sleeve (peso {w} * 100k)"
        )


# =========================================================================== #
# NOVO-3 (BLOQUEADOR) — DUPLA CONTAGEM de P&L no NAV do sleeve com o DEFAULT
# (paper SO-BETA, sem capital absoluto) E posicoes do beta carregando P&L REAL.
# Era o caminho que NENHUM teste cobria (o sizing-test usava capital ABSOLUTO +
# P&L 0). Estes testes exercitam exatamente o desvio que escondia o bug: default
# (sem capital) + P&L != 0 -> o guard deve ver o drawdown 1x (nao 2x).
# (Re-auditoria: data/code_audit_beta_production_v3.txt, NOVO-3.)
# =========================================================================== #
def _fully_invested_sleeve_broker(monkeypatch) -> FakeBroker:
    """Paper SO-BETA (default), sleeve TOTALMENTE investido: caixa 100k -> compra
    1000 SPY@100 (gasta o caixa). P&L 0 no instante da entrada; mover SPY move o
    NAV. Modela o caminho VIVO real (o sizing/guard veem o equity da conta)."""
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("100")})
    broker.submit_order(
        OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("1000"),
                    order_type=OrderType.MARKET, strategy="beta_rebalancer")
    )
    assert broker.get_account().cash == Decimal("0")        # caixa gasto na compra
    assert broker.get_account().equity == Decimal("100000")  # P&L 0 na entrada
    return broker


def test_novo3_guard_sees_real_1x_drawdown_default_pct(monkeypatch):
    """NOVO-3 (o nucleo): com o DEFAULT (paper so-beta) e posicoes do beta com P&L,
    o NAV do sleeve == equity da conta (P&L UMA vez). O guard ve o drawdown REAL
    (1x), nao 2x. Antes (pct*equity + P&L): -30% real era lido como -60%."""
    broker = _fully_invested_sleeve_broker(monkeypatch)
    for px, true_dd in [("90", "-0.10"), ("80", "-0.20"), ("70", "-0.30")]:
        broker.set_price("SPY", px)
        eq = broker.get_account().equity
        nav = resolve_sleeve_nav(broker)
        assert nav == eq, f"NAV {nav} != equity {eq} (double-count do P&L?)"
        dd = (nav - Decimal("100000")) / Decimal("100000")
        assert dd == Decimal(true_dd), f"DD do guard {dd} != real {true_dd} (2x?)"


def test_novo3_normal_pullback_minus36_does_not_halt(monkeypatch):
    """PERFIL 2.0x (regressao central — a classe de bug das 4 rodadas): um
    drawdown NORMAL do perfil 2.0x ate -36% REAL (= o DD AO-VIVO esperado, mid)
    NAO halta. A queda e LENTA entre dias (cada passo < gate diario -8%), entao so
    o gate de DD pico-a-vale poderia disparar — e -36% fica ABAIXO do halt -43%.

    Esta e a prova de que a operacao normal a 2.0x (DD esperado -25/-36%) nao
    dispara o guard. Com o threshold antigo de -28% (calibrado p/ 1.5x), -36%
    HALTARIA — truncando o track record num drawdown NORMAL do novo perfil."""
    from config.risk import get_beta_guard_settings

    bs = get_beta_guard_settings()
    broker = _fully_invested_sleeve_broker(monkeypatch)
    peak = resolve_sleeve_nav(broker)          # high-water inicial = 100k
    prev_day_nav = peak                          # daily start (reset a cada dia)
    # slide LENTO entre dias ate -36% cumulativo (cada passo <= ~6% < gate -8%).
    for px in ["95", "90", "85", "80", "75", "70", "67", "64"]:
        broker.set_price("SPY", px)
        nav = resolve_sleeve_nav(broker)
        guard = PortfolioRiskGuard(
            start_equity=prev_day_nav,           # gate diario vs fechamento de ontem
            daily_loss_pct=bs.daily_loss_limit_pct,
            max_dd_pct=bs.max_drawdown_pct,
            peak_equity=peak,                    # gate de DD pico-a-vale
        )
        guard.update(nav)
        assert not guard.trading_halted, (
            f"-36% slow slide (DD normal do perfil 2.0x) NAO deve haltar "
            f"(px={px}, nav={nav}): {guard.halt_reason}"
        )
        peak = max(peak, nav)
        prev_day_nav = nav
    # confere que chegamos exatamente a -36% REAL (o DD ao-vivo esperado).
    assert (resolve_sleeve_nav(broker) - Decimal("100000")) / Decimal("100000") == Decimal("-0.36")


def test_novo3_real_crash_beyond_minus43_halts_via_dd_gate(monkeypatch):
    """PERFIL 2.0x (lado complementar): um tombo REAL pico-a-vale ALEM de -43%
    (PIOR que o DD ao-vivo esperado -36/-39% do produto = cisne negro) DISPARA o
    halt de DD. Passos diarios gentis (~1%) p/ o gate diario nao mascarar o teste
    — so o gate de DD pode disparar, e ele dispara em -43% REAL (1x), nao antes.

    Confirma que o guard NAO foi desarmado ao subir o threshold p/ 2.0x: ele
    ainda protege contra um tombo alem do historico do produto."""
    from config.risk import get_beta_guard_settings

    bs = get_beta_guard_settings()
    broker = _fully_invested_sleeve_broker(monkeypatch)
    peak = resolve_sleeve_nav(broker)
    prev_day_nav = peak
    halt_dd = None
    px = 100
    while px > 50:
        px -= 1                                   # ~1%/dia: gate diario nunca dispara
        broker.set_price("SPY", str(px))
        nav = resolve_sleeve_nav(broker)
        guard = PortfolioRiskGuard(
            start_equity=prev_day_nav,
            daily_loss_pct=bs.daily_loss_limit_pct,
            max_dd_pct=bs.max_drawdown_pct,
            peak_equity=peak,
        )
        guard.update(nav)
        if guard.trading_halted:
            halt_dd = (nav - Decimal("100000")) / Decimal("100000")
            assert "drawdown" in guard.halt_reason, guard.halt_reason
            break
        peak = max(peak, nav)
        prev_day_nav = nav
    assert halt_dd is not None and halt_dd <= Decimal("-0.43"), (
        f"DD-halt deveria disparar em <= -43% REAL (1x) no perfil 2.0x, disparou em {halt_dd}"
    )


def test_novo3_single_day_crash_trips_daily_gate_at_real_pct(monkeypatch):
    """PERFIL 2.0x: o gate de perda DIARIA dispara em -8% (1x), nao antes nem em
    2x. Calibrado p/ a vol do book 2.0x (sigma diario ~1.26% nominal): um dia
    tipico/cauda moderada NAO halta; so um choque intradia anomalo (-8% ~ 6 sigmas;
    pior dia do backtest -5.67%) dispara. Confirma tambem que a leitura e REAL
    (1x), nao 2x (no mundo bugado -7% era lido como -14% e disparava cedo)."""
    from config.risk import get_beta_guard_settings

    bs = get_beta_guard_settings()
    broker = _fully_invested_sleeve_broker(monkeypatch)
    start = resolve_sleeve_nav(broker)
    guard = PortfolioRiskGuard(
        start_equity=start, daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct, peak_equity=start,
    )
    guard.update(start)
    # -7% intradia: NAO dispara o gate diario de -8% (dia de cauda moderada do 2.0x).
    broker.set_price("SPY", "93")
    guard.update(resolve_sleeve_nav(broker))
    assert not guard.trading_halted, "-7% REAL nao deve disparar o gate diario de -8% (perfil 2.0x)"
    # -9% intradia: dispara (choque anomalo; e a leitura e -9%, nao -18%).
    broker.set_price("SPY", "91")
    guard.update(resolve_sleeve_nav(broker))
    assert guard.trading_halted and "-9" in guard.halt_reason, guard.halt_reason


def test_novo3_sizing_default_pct_uses_account_equity_with_pnl(monkeypatch, synthetic):
    """NOVO-3: o DIMENSIONAMENTO em paper so-beta (default) usa o equity da conta
    (P&L uma vez), nao pct*equity+P&L. Com o sleeve carregando P&L, plan.equity ==
    equity da conta (nao o dobro do P&L)."""
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    broker = FakeBroker(cash=Decimal("100000"), prices=prices, buying_power=Decimal("9000000"))
    # uma posicao do beta JA aberta carregando P&L (+20% no SPY).
    spy_entry = Decimal(str(last["SPY"]))
    broker.seed_position("SPY", Decimal("100"), spy_entry)
    broker.set_price("SPY", str(spy_entry * Decimal("1.2")))   # +20% => P&L > 0
    expected_equity = broker.get_account().equity              # inclui P&L UMA vez
    reb = BetaRebalancer(broker)
    plan = reb.compute_plan(panel, asof)
    assert plan.equity == expected_equity, (
        f"sizing usou {plan.equity}, esperado equity da conta {expected_equity} "
        f"(double-count do P&L?)"
    )


# =========================================================================== #
# NOVO-4 (MEDIO) — residuo STALE 'beta:risk_start_equity:<hoje>'=0 em disco
# (lixo da versao buggada do NOVO-1). No boot DEVE ser re-derivado do NAV (nao
# coagido p/ 1, que neutralizava o gate de perda diaria do 1o dia) e a chave
# saneada. (Re-auditoria: data/code_audit_beta_production_v3.txt, NOVO-4.)
# =========================================================================== #
def test_novo4_stale_zero_start_is_rederived_not_coerced(monkeypatch):
    """NOVO-4: um start persistido = 0 (residuo pre-fix) NAO neutraliza o gate
    diario. _build_beta_guard re-deriva start do NAV do boot e SANEIA a chave."""
    import main as main_mod
    from datetime import datetime, timezone

    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    db = Database(":memory:")
    st = StateRepository(db)
    today = datetime.now(timezone.utc).date().isoformat()
    start_key = f"beta:risk_start_equity:{today}"
    # semeia o LIXO (a versao buggada do NOVO-1 gravava 0).
    st.set_decimal(start_key, Decimal("0"))

    # boot flat (go-live): 100k de caixa, zero posicoes do beta.
    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("100")})
    guard = main_mod._build_beta_guard(broker, st, AuditLog(db))

    # start re-derivado do NAV do boot (100k), NAO coagido p/ 1; chave saneada.
    assert guard.start_equity == Decimal("100000"), guard.start_equity
    assert st.get_decimal(start_key) == Decimal("100000")   # chave stale re-escrita
    assert not guard.trading_halted
    db.close()


def test_novo4_daily_gate_active_on_day1_despite_stale_zero(monkeypatch):
    """NOVO-4 (a consequencia): com o residuo 0 saneado, o gate de PERDA DIARIA
    fica ATIVO no 1o dia. Antes (start coagido a 1, denominador 1) uma perda
    intradia de -10% NAO disparava o halt diario de -6% — agora dispara."""
    import main as main_mod
    from datetime import datetime, timezone

    monkeypatch.delenv("BETA_SLEEVE_CAPITAL", raising=False)
    monkeypatch.delenv("BETA_SLEEVE_CAPITAL_PCT", raising=False)
    db = Database(":memory:")
    st = StateRepository(db)
    today = datetime.now(timezone.utc).date().isoformat()
    st.set_decimal(f"beta:risk_start_equity:{today}", Decimal("0"))  # lixo pre-fix

    broker = FakeBroker(cash=Decimal("100000"), prices={"SPY": Decimal("100")})
    guard = main_mod._build_beta_guard(broker, st, AuditLog(db))
    assert not guard.trading_halted

    # posicao do beta cai -10% intradia (compra 1000 SPY@100, depois SPY->90).
    broker.submit_order(
        OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("1000"),
                    order_type=OrderType.MARKET, strategy="beta_rebalancer")
    )
    broker.set_price("SPY", "90")  # -10% sobre o NAV de 100k
    guard.update(resolve_sleeve_nav(broker))
    assert guard.trading_halted and "daily loss" in guard.halt_reason, (
        f"gate diario deve disparar em -10% (start re-derivado, nao 1): {guard.halt_reason}"
    )
    db.close()


# =========================================================================== #
# NOVO-2 (ALTO) — sem feed vivo o painel CONGELA. Um job agendado re-baixa o
# painel (mesma fonte do backtest) p/ o asof avancar. Atras da MESMA flag (OFF
# por padrao), com fail-safe de rede.
# (Re-auditoria: data/code_audit_beta_production_v2.txt, Parte B / NOVO-2.)
# =========================================================================== #
def test_novo2_refresh_noop_when_flag_off(monkeypatch, tmp_path):
    """NOVO-2: refresh_production_panel e NO-OP quando a flag esta desligada (OFF
    por padrao: nem env var, nem `.env`)."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # sem `.env` no cwd => flag default OFF (legado)
    out = beta_live.refresh_production_panel()
    assert out["enabled"] is False and out["refreshed"] is False


def test_novo2_schedule_noop_when_flag_off(monkeypatch, tmp_path):
    """NOVO-2: schedule_panel_refresh NAO agenda nada com a flag desligada (OFF
    por padrao: nem env var, nem `.env`)."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # sem `.env` no cwd => flag default OFF (legado)

    class _Sched:
        def __init__(self):
            self.jobs = []

        def add_job(self, *a, **k):
            self.jobs.append(k.get("id"))

    sched = _Sched()
    assert beta_live.schedule_panel_refresh(sched) is False
    assert sched.jobs == []


def test_novo2_schedule_adds_daily_job_when_flag_on(monkeypatch):
    """NOVO-2: com a flag ligada, schedule_panel_refresh agenda UM job diario
    (cron 11:00 UTC, pre-mercado) cujo callback chama o refresher."""
    from strategies import beta_live

    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    calls = {"n": 0}

    def fake_refresher():
        calls["n"] += 1
        return {"refreshed": True, "enabled": True}

    captured = {}

    class _Sched:
        def add_job(self, fn, trigger, *, id=None, replace_existing=False):
            captured["id"] = id
            captured["fn"] = fn
            captured["trigger"] = trigger

    ok = beta_live.schedule_panel_refresh(_Sched(), refresher=fake_refresher)
    assert ok is True
    assert captured["id"] == "beta_panel_refresh"
    # cadencia documentada: cron diario as 11:00 UTC.
    fields = {f.name: str(f) for f in captured["trigger"].fields}
    assert fields["hour"] == "11" and fields["minute"] == "0"
    # o callback agendado dispara o refresher.
    captured["fn"]()
    assert calls["n"] == 1


def test_novo2_refresh_advances_asof(monkeypatch):
    """NOVO-2 (o nucleo): com a flag ligada, refresh_production_panel re-baixa o
    painel e o asof (ultima barra) AVANCA. Mockamos load_production_panel p/ nao
    tocar a rede; provamos que o asof reportado e o ultimo close do painel novo."""
    from unittest import mock

    from strategies import beta_live

    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    new_idx = pd.bdate_range("2018-01-01", periods=10, tz="UTC")
    new_panel = pd.DataFrame({"SPY": range(10)}, index=new_idx)
    with mock.patch(
        "strategies.beta_rebalancer.load_production_panel",
        return_value=(new_panel, {"SPY": "equity"}),
    ) as m:
        out = beta_live.refresh_production_panel()
    m.assert_called_once_with(force=True)  # FORCA o re-download (estende o cache).
    assert out["refreshed"] is True
    assert pd.Timestamp(out["asof"]).date() == new_idx[-1].date()  # asof avancou


def test_novo2_refresh_failsafe_on_network_error(monkeypatch):
    """NOVO-2 (fail-safe): se o download falhar (rede), refresh_production_panel
    NAO levanta — loga e devolve refreshed=False (mantem o ultimo cache)."""
    from unittest import mock

    from strategies import beta_live

    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    with mock.patch(
        "strategies.beta_rebalancer.load_production_panel",
        side_effect=RuntimeError("yfinance indisponivel"),
    ):
        out = beta_live.refresh_production_panel()  # nao deve levantar
    assert out["refreshed"] is False
    assert "fail-safe" in out.get("note", "")


def test_novo2_refresh_preserves_weight_parity(real_panel, monkeypatch):
    """NOVO-2 (paridade preservada): o refresh usa o MESMO load_panel do backtest,
    entao os pesos calculados sobre o painel ATUALIZADO continuam bit-a-bit iguais
    a linha do vetor de pesos do backtest. O refresh so AVANCA a data, nao muda a
    identidade de calculo.

    Provamos a invariante diretamente: para o painel real, os pesos vivos == os do
    backtest na ultima data (a paridade que o refresh deve preservar)."""
    from unittest import mock

    from strategies import beta_live

    panel, classes = real_panel
    monkeypatch.setenv("BETA_LIVE_ENABLED", "1")
    # o refresh devolve o painel real (mesma fonte do backtest).
    with mock.patch(
        "strategies.beta_rebalancer.load_production_panel",
        return_value=(panel, classes),
    ):
        out = beta_live.refresh_production_panel()
    assert out["refreshed"] is True
    asof = pd.Timestamp(out["asof"])
    # paridade peso sobre o painel atualizado: vivo == backtest, bit-a-bit.
    wf_full = production_weight_frame(panel, classes)
    live = target_weights_for_date(panel, classes, asof)
    for tkr, w in live.items():
        assert w == pytest.approx(float(wf_full.loc[asof][tkr]), abs=1e-9), (
            f"refresh quebrou a paridade peso em {tkr}"
        )


# =========================================================================== #
# BLOQUEADOR DA 2.0x (data/code_audit_beta_2x.txt, Parte 4) — CRIPTO cash-only
# REJEITADA quando o book de ACOES a 2.0x consome o caixa. O fix e de EXECUCAO
# (reserva de caixa p/ cripto + AccountInfo com pool nao-marginavel + Executor
# validando cripto contra o caixa), NAO de estrategia: pesos/alavancagem 2.0x
# ficam intactos e as posicoes finais batem com o backtest.
#
# GAP DE TESTE FECHADO: estes testes usam SEMANTICA DE MARGEM REAL (dois pools:
# marginavel + caixa nao-marginavel), NAO buying_power=9M (que sidesteppava o
# gate de margem). FakeBroker(margin_semantics=True) modela a Alpaca fielmente:
# acoes consomem o pool marginavel + caixa (colateral); cripto so o caixa, e o
# broker REJEITA (levanta) cripto que exceda o caixa (server-side da Alpaca).
# (OrderType/OrderIntent/OrderSide ja importados no topo do arquivo.)
# =========================================================================== #
def _exec(broker, tmp_path):
    """Executor real (com order_repo/audit) p/ os testes de margem."""
    db = Database(":memory:")
    ex = Executor(
        broker, TradeLogger(db), KillSwitch(tmp_path / "K"),
        order_repo=OrderRepository(db), audit=AuditLog(db),
    )
    return ex, db


def _worst_case_intents():
    """Pior caso do Coder (2016-06-02): gross 200%, BTC 10%, conta 100k.
    Notional-alvo (do audit): SPY 38.9k, QQQ 30.8k, TLT 38.0k, IEF 82.0k, BTC 10k
    = 199.7k gross sobre 100k equity (~2.0x). Precos normalizados a 100 (acoes) e
    10.000 (BTC) p/ qty exata; o que importa e o NOTIONAL e a ordem de execucao."""
    prices = {
        "SPY": Decimal("100"), "QQQ": Decimal("100"), "TLT": Decimal("100"),
        "IEF": Decimal("100"), "BTC/USD": Decimal("10000"),
    }
    intents = [
        OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("389"), order_type=OrderType.MARKET, strategy="beta_rebalancer"),
        OrderIntent(symbol="QQQ", side=OrderSide.BUY, qty=Decimal("308"), order_type=OrderType.MARKET, strategy="beta_rebalancer"),
        OrderIntent(symbol="TLT", side=OrderSide.BUY, qty=Decimal("380"), order_type=OrderType.MARKET, strategy="beta_rebalancer"),
        OrderIntent(symbol="IEF", side=OrderSide.BUY, qty=Decimal("820"), order_type=OrderType.MARKET, strategy="beta_rebalancer"),
        OrderIntent(symbol="BTC/USD", side=OrderSide.BUY, qty=Decimal("1.0"), order_type=OrderType.MARKET, strategy="beta_rebalancer"),
    ]
    return prices, intents


def test_2x_crypto_rejected_with_old_stocks_first_order_real_margin(tmp_path):
    """REPRODUZ o bug (counterfactual): com a ordem ANTIGA (acoes primeiro, cripto
    por ultimo, como o loop do universo), numa conta de MARGEM REAL 100k/4x, as
    acoes a 2.0x consomem o caixa (colateral) e a cripto cash-only e REJEITADA ->
    book vivo = 189.7% acoes + 0% cripto (a paridade quebra). Prova que o
    bloqueador e real e que e a ORDEM de execucao que importa."""
    prices, intents = _worst_case_intents()
    broker = FakeBroker(
        cash=Decimal("100000"), prices=prices,
        buying_power=Decimal("400000"),   # 4x equity (paper margin)
        margin_semantics=True,            # DOIS POOLS reais
    )
    ex, db = _exec(broker, tmp_path)
    # NAO reordena (ordem antiga, acoes primeiro): a cripto cai.
    results = ex.execute_many(intents)
    filled = {r.symbol for r in results}
    assert "BTC/USD" not in filled, "o bug exige que a cripto caia com a ordem antiga"
    pos = {p.symbol: p.qty for p in broker.get_positions()}
    assert pos.get("BTC/USD", Decimal("0")) == Decimal("0")  # 0% cripto (book errado)
    db.close()


def test_2x_crypto_not_rejected_with_cash_reservation_real_margin(tmp_path):
    """O FIX (data/code_audit_beta_2x.txt, item 1): com a RESERVA DE CAIXA (cripto
    ordenada ANTES das acoes via _order_for_cash_reservation), na MESMA conta de
    margem real 100k/4x, a cripto NAO e rejeitada e as acoes deployam na margem.
    Book vivo == backtest: 189.7% acoes + 10% cripto (NAO 190% + 0%)."""
    prices, intents = _worst_case_intents()
    broker = FakeBroker(
        cash=Decimal("100000"), prices=prices,
        buying_power=Decimal("400000"), margin_semantics=True,
    )
    ordered = BetaRebalancer(broker)._order_for_cash_reservation(intents)
    # cripto vem PRIMEIRO (reserva o caixa).
    assert ordered[0].symbol == "BTC/USD", [i.symbol for i in ordered]

    ex, db = _exec(broker, tmp_path)
    results = ex.execute_many(ordered)
    filled = {r.symbol for r in results}
    assert filled == {"SPY", "QQQ", "TLT", "IEF", "BTC/USD"}, f"cripto rejeitada? {filled}"

    # carteira final == backtest: 189.7% acoes + 10% cripto.
    equity = Decimal("100000")
    pos = {p.symbol: p.qty for p in broker.get_positions()}
    btc_mv = pos["BTC/USD"] * prices["BTC/USD"]
    stock_mv = sum(pos[s] * prices[s] for s in ("SPY", "QQQ", "TLT", "IEF"))
    assert btc_mv / equity == Decimal("0.10"), f"cripto vivo {btc_mv/equity} != 10% backtest"
    assert abs(stock_mv / equity - Decimal("1.897")) < Decimal("0.001"), stock_mv / equity
    db.close()


def test_2x_first_rebalance_with_crypto_end_to_end_real_margin(synthetic, tmp_path, state, monkeypatch):
    """END-TO-END (gap fechado): a 1a rebalanceada a 2.0x com cripto>0, via o
    runner vivo COMPLETO (run_beta_rebalance_cycle, force), numa conta de MARGEM
    REAL (margin_semantics, BP=4x, SEM o buying_power=9M que mascarava o gate),
    executa TODAS as ordens incluindo a cripto, e a carteira final bate com os
    pesos-alvo do backtest a 2.0x (dentro da banda de arredondamento)."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # flag off no cwd; force=True exercita o caminho vivo
    panel, classes = _panel_with_crypto_weight()  # garante cripto-alvo > banda
    asof = panel.index[-1]
    last = panel.loc[asof]
    target = target_weights_for_date(panel, classes, asof)
    assert abs(target.get("BTC-USD", 0.0)) > float(DEFAULT_NO_TRADE_BAND), target.get("BTC-USD")

    equity = Decimal("100000")
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    # Conta de MARGEM REAL: caixa 100k (pool cripto), BP 4x (pool marginavel).
    broker = FakeBroker(
        cash=equity, prices=prices,
        buying_power=Decimal("400000"), margin_semantics=True,
    )
    ex, db = _exec(broker, tmp_path)
    out = beta_live.run_beta_rebalance_cycle(
        broker, ex, state=state, panel=panel, asof=asof, force=True
    )
    assert out["due"] is True
    # a cripto EXECUTOU (nao foi barrada nem rejeitada).
    assert any(i.symbol == "BTC/USD" for i in broker.submitted), "cripto nao chegou ao broker"
    assert "BTC/USD" not in str(out.get("blocked", [])), out.get("blocked")

    # carteira final == pesos-alvo do backtest (cripto incluida), dentro da banda.
    final_w, _ = BetaRebalancer(broker)._current_weights(equity)
    band = float(DEFAULT_NO_TRADE_BAND)
    for tkr, w in target.items():
        assert abs(final_w[tkr] - w) <= band, (
            f"posicao final {tkr} {final_w[tkr]} != alvo backtest {w} (paridade quebrou)"
        )
    # especificamente, a cripto NAO ficou em 0 (o bug): ficou no alvo.
    assert abs(final_w["BTC-USD"] - target["BTC-USD"]) <= band
    assert final_w["BTC-USD"] > 0
    db.close()


def test_2x_leverage_and_weights_unchanged_by_fix(synthetic):
    """O FIX e de EXECUCAO, NAO de estrategia: os pesos-alvo e a alavancagem 2.0x
    ficam INTACTOS. A reserva de caixa so reordena as intents — NAO muda quais
    ordens existem, suas quantidades, nem os pesos-alvo do plano."""
    panel, classes = _panel_with_crypto_weight()
    asof = panel.index[-1]
    last = panel.loc[asof]
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    broker = FakeBroker(cash=Decimal("100000"), prices=prices, buying_power=Decimal("400000"))
    reb = BetaRebalancer(broker)
    plan = reb.compute_plan(panel, asof, equity=Decimal("100000"))

    # alavancagem do plano == 2.0x de producao (parametro intacto).
    assert plan.leverage == PRODUCTION_LEVERAGE == 2.0
    # pesos-alvo == os pesos do backtest a 2.0x (paridade de peso, bit-a-bit).
    bt = target_weights_for_date(panel, classes, asof, leverage=2.0)
    for tkr, w in bt.items():
        assert plan.target_weights[tkr] == pytest.approx(w, abs=1e-12)
    # o CONJUNTO de ordens (simbolo->qty) e o MESMO com e sem a reserva (so a ordem
    # da lista muda): a reserva e uma permutacao, nao adiciona/remove/redimensiona.
    by_symbol = {i.symbol: i.qty for i in plan.intents}
    reordered = reb._order_for_cash_reservation(list(plan.intents))
    assert {i.symbol: i.qty for i in reordered} == by_symbol
    assert sorted(i.symbol for i in reordered) == sorted(i.symbol for i in plan.intents)


def test_2x_cash_reservation_orders_sells_then_crypto_then_stocks(synthetic):
    """A reserva de caixa segue a ordem CORRETA: SELLs (liberam caixa) -> BUYs de
    CRIPTO (reservam o caixa) -> BUYs de ACOES (caixa restante + margem)."""
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    broker = FakeBroker(cash=Decimal("100000"), prices=prices)
    reb = BetaRebalancer(broker)
    raw = [
        OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("10"), order_type=OrderType.MARKET, strategy="t"),
        OrderIntent(symbol="BTC/USD", side=OrderSide.BUY, qty=Decimal("0.1"), order_type=OrderType.MARKET, strategy="t"),
        OrderIntent(symbol="QQQ", side=OrderSide.SELL, qty=Decimal("5"), order_type=OrderType.MARKET, strategy="t"),
        OrderIntent(symbol="ETH/USD", side=OrderSide.BUY, qty=Decimal("0.2"), order_type=OrderType.MARKET, strategy="t"),
    ]
    ordered = reb._order_for_cash_reservation(raw)
    sides_syms = [(i.side, i.symbol) for i in ordered]
    # 1o balde: SELLs.
    assert sides_syms[0] == (OrderSide.SELL, "QQQ")
    # 2o balde: BUYs de cripto (ambos, em ordem estavel).
    crypto_buys = [s for (side, s) in sides_syms if side == OrderSide.BUY and "/" in s]
    stock_buys = [s for (side, s) in sides_syms if side == OrderSide.BUY and "/" not in s]
    assert crypto_buys == ["BTC/USD", "ETH/USD"]
    # 3o balde: BUYs de acoes DEPOIS das criptos.
    first_stock_idx = next(i for i, (side, s) in enumerate(sides_syms) if side == OrderSide.BUY and "/" not in s)
    last_crypto_idx = max(i for i, (side, s) in enumerate(sides_syms) if side == OrderSide.BUY and "/" in s)
    assert last_crypto_idx < first_stock_idx
    assert stock_buys == ["SPY"]


def test_account_info_exposes_non_marginable_pool():
    """AccountInfo (broker/base.py + fake_broker) expoe o pool NAO-MARGINAVEL
    (caixa p/ cripto), separado do pool marginavel (buying_power). Sem margin
    semantics, o nao-marginavel = caixa (compat.). Com, e o caixa vivo."""
    # legado: non_marginable default = cash.
    b1 = FakeBroker(cash=Decimal("50000"), prices={}, buying_power=Decimal("200000"))
    a1 = b1.get_account()
    assert a1.buying_power == Decimal("200000")          # pool marginavel
    assert a1.non_marginable_buying_power == Decimal("50000")  # caixa (cripto)

    # margem real: apos comprar acoes (consome caixa como colateral), o pool de
    # cripto (caixa) ENCOLHE — e o que faz a cripto cair se nao for reservada.
    b2 = FakeBroker(
        cash=Decimal("100000"),
        prices={"SPY": Decimal("100")},
        buying_power=Decimal("400000"), margin_semantics=True,
    )
    b2.submit_order(OrderIntent(symbol="SPY", side=OrderSide.BUY, qty=Decimal("1500"), order_type=OrderType.MARKET, strategy="t"))
    a2 = b2.get_account()
    # comprou 150k de SPY: caixa -> -50k (emprestado), pool cripto -> 0 (nunca <0).
    assert a2.cash == Decimal("-50000")
    assert a2.non_marginable_buying_power == Decimal("0")  # sem caixa p/ cripto!
    assert a2.buying_power == Decimal("250000")            # margem decrementada


def test_2x_crypto_target_exceeding_cash_is_graceful(tmp_path):
    """Item 3 do briefing: se a cripto-alvo EXCEDER o caixa (anomalo, nao esperado
    a ~20%), o sistema trata GRACIOSO — nao quebra. O Executor barra a sobra de
    cripto contra o caixa (cash-only) e segue; nada de excecao nao-tratada."""
    # caixa minusculo (1k) mas alvo de cripto 10k -> excede o caixa.
    prices = {"BTC/USD": Decimal("10000")}
    broker = FakeBroker(
        cash=Decimal("1000"), prices=prices,
        buying_power=Decimal("4000"), margin_semantics=True,
    )
    ex, db = _exec(broker, tmp_path)
    intent = OrderIntent(symbol="BTC/USD", side=OrderSide.BUY, qty=Decimal("1.0"), order_type=OrderType.MARKET, strategy="beta_rebalancer")
    # NAO levanta: o Executor barra na validacao (pre-trade), retorna None.
    result = ex.execute(intent)
    assert result is None                       # barrada com graca (sem crash)
    assert broker.submitted == []               # nem chegou ao broker
    pos = {p.symbol: p.qty for p in broker.get_positions()}
    assert pos.get("BTC/USD", Decimal("0")) == Decimal("0")
    db.close()


def test_2x_crypto_shortfall_noted_in_plan(synthetic):
    """Item 3: quando a cripto-alvo excede o caixa nao-marginavel, o plano REGISTRA
    uma nota auditavel (nao silencioso). Em operacao normal (cripto ~20% << caixa)
    a nota nao aparece."""
    panel, classes = _panel_with_crypto_weight()
    asof = panel.index[-1]
    last = panel.loc[asof]
    target = target_weights_for_date(panel, classes, asof)
    w_btc = target["BTC-USD"]
    assert w_btc > float(DEFAULT_NO_TRADE_BAND)

    # equity 100k mas caixa nao-marginavel ridiculo (forca o shortfall da cripto).
    prices = {to_broker_symbol(t): Decimal(str(last[t])) for t in classes}
    equity = Decimal("100000")
    broker = FakeBroker(cash=Decimal("10"), prices=prices, buying_power=Decimal("400000"))
    reb = BetaRebalancer(broker)
    plan = reb.compute_plan(panel, asof, equity=equity)
    assert any("cripto-alvo" in n and "caixa" in n for n in plan.notes), plan.notes

    # operacao NORMAL: caixa = equity (100% do equity) cobre a cripto (~20%) -> sem nota.
    broker_ok = FakeBroker(cash=equity, prices=prices, buying_power=Decimal("400000"))
    plan_ok = BetaRebalancer(broker_ok).compute_plan(panel, asof, equity=equity)
    assert not any("cripto-alvo" in n for n in plan_ok.notes), plan_ok.notes


def test_2x_guards_intact_under_real_margin(real_panel, tmp_path):
    """Os guards 2.0x continuam INTACTOS sob margem real: a 1a rebalanceada a 2.0x
    com o guard de PRODUCAO (boot sadio) e conta de MARGEM REAL (margin_semantics,
    BP=4x, SEM buying_power=9M) NAO bloqueia nenhuma compra legitima por per-symbol,
    e o per-symbol ainda barra um alvo anomalo > 210%."""
    from config.risk import get_beta_guard_settings

    panel, classes = real_panel
    asof = panel.index[-1]
    last = panel.loc[asof]
    bs = get_beta_guard_settings()
    guard = PortfolioRiskGuard(
        start_equity=Decimal("100000"),
        daily_loss_pct=bs.daily_loss_limit_pct,
        max_dd_pct=bs.max_drawdown_pct,
        max_per_symbol_pct=bs.max_per_symbol_pct,
        max_heat_pct=bs.max_portfolio_heat_pct,
        peak_equity=Decimal("100000"),
    )
    guard.update(Decimal("100000"))
    assert not guard.trading_halted
    # MARGEM REAL: caixa 100k, BP 4x (NAO 9M). Gate de margem ATIVO de verdade.
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        buying_power=Decimal("400000"), margin_semantics=True,
    )
    plan = BetaRebalancer(broker, guard=guard).compute_plan(panel, asof, equity=Decimal("100000"))
    # nenhuma compra legitima bloqueada por per-symbol (operacao normal a 2.0x).
    assert not any("simbolo" in b for b in plan.blocked), plan.blocked
    # e o per-symbol ainda barra um alvo anomalo (> 210%) — guard nao desarmado.
    ok, reason = guard.can_open(Decimal("2.11"), Decimal("0"))
    assert not ok and "simbolo" in reason


def test_2x_boot_flat_with_real_margin_ok(synthetic, tmp_path, state, monkeypatch):
    """BOOT FLAT a 2.0x sob margem real: partindo de conta FLAT (caixa, zero
    posicoes) com margin_semantics, a 1a rebalanceada gera e EXECUTA ordens (book
    nao nasce vazio) — o boot-flat nao regrediu com o fix de margem."""
    from strategies import beta_live

    monkeypatch.delenv("BETA_LIVE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)  # flag off no cwd; force=True exercita o caminho vivo
    panel, classes = synthetic
    asof = panel.index[-1]
    last = panel.loc[asof]
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={to_broker_symbol(t): Decimal(str(last[t])) for t in classes},
        buying_power=Decimal("400000"), margin_semantics=True,
    )
    ex, db = _exec(broker, tmp_path)
    out = beta_live.run_beta_rebalance_cycle(
        broker, ex, state=state, panel=panel, asof=asof, force=True
    )
    assert out["due"] is True
    assert out["executed"] >= 1, f"boot-flat a 2.0x nao deve gravar book vazio: {out}"
    assert broker.submitted
    db.close()


def test_financing_residual_is_zero_crypto_cash_funded(real_panel):
    """DIFERENCA RESIDUAL DE FINANCIAMENTO (briefing): a cripto cash-funded no vivo
    NAO paga o juro que o backtest assumiria? PROVA que o residual e ZERO (nao so
    2a ordem): o capital EMPRESTADO e identico nos dois casos.

    Backtest: borrowed = max(gross_total - 1, 0), com a cripto DENTRO do gross.
    Vivo: a cripto sai do CAIXA (1x); o caixa que ela consome SENAO reduziria o
    emprestado das acoes. Como o caixa cobre 100% do equity nos dois casos,
    borrowed_vivo = max(gross_acoes - (1 - gross_cripto), 0) = max(gross_total-1,0)
    = borrowed_backtest. Logo o custo de juro e IGUAL -> residual de CAGR = 0.
    (Confirma que NAO precisa re-rodar o backtest: a premissa de financiamento ja
    esta correta p/ a cripto cash-funded.)"""
    from strategies.beta_rebalancer import _is_crypto_ticker

    panel, classes = real_panel
    wf = production_weight_frame(panel, classes)  # 2.0x
    crypto_cols = [c for c in wf.columns if _is_crypto_ticker(c)]
    stock_cols = [c for c in wf.columns if not _is_crypto_ticker(c)]
    gross_total = wf.abs().sum(axis=1)
    gross_crypto = wf[crypto_cols].abs().sum(axis=1)
    gross_stock = wf[stock_cols].abs().sum(axis=1)

    borrowed_bt = (gross_total - 1.0).clip(lower=0.0)
    cash_free = (1.0 - gross_crypto).clip(lower=0.0)      # caixa apos reservar cripto
    borrowed_live = (gross_stock - cash_free).clip(lower=0.0)

    max_abs_diff = float((borrowed_bt - borrowed_live).abs().max())
    assert max_abs_diff < 1e-12, (
        f"borrowed difere (residual de financiamento != 0): max|diff|={max_abs_diff}"
    )
    # sinal: o residual (se houvesse) e FAVORAVEL ao vivo (vivo paga <= juro).
    assert float((borrowed_bt - borrowed_live).min()) >= -1e-12
    # ha dias com cripto>0 no painel (senao o teste nao prova nada).
    assert int((gross_crypto > 1e-9).sum()) > 0
