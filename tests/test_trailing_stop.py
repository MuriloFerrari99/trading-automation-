"""Testes da estrategia Trailing Stop (nativo Alpaca)."""

from __future__ import annotations

from decimal import Decimal

from broker.base import BrokerOrder
from core.models import OrderSide, OrderType
from strategies.base import StrategyContext
from strategies.trailing_stop import TrailingStopStrategy


def _ctx(broker, state, watchlist):
    return StrategyContext(broker=broker, state=state, watchlist=watchlist)


def test_no_position_no_intents(broker, state, watchlist):
    broker.set_price("AAPL", "100")
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []


def test_places_native_trailing_stop_when_holding(broker, state, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    intent = intents[0]
    assert intent.side == OrderSide.SELL
    assert intent.order_type == OrderType.TRAILING_STOP
    assert intent.qty == Decimal("10")
    assert intent.trail_percent == Decimal("10")  # 0.10 -> 10%


def test_does_not_duplicate_when_trailing_already_open(broker, state, watchlist):
    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    # ja existe uma trailing stop aberta no broker p/ AAPL
    broker.seed_open_order(
        BrokerOrder(
            broker_order_id="t1", client_order_id="x", symbol="AAPL", side="sell",
            qty=Decimal("10"), filled_qty=Decimal("0"), status="new",
            order_type="trailing_stop",
        )
    )
    intents = TrailingStopStrategy().evaluate(_ctx(broker, state, watchlist))
    assert intents == []  # ja protegido, nao duplica


def test_default_protects_long_without_watchlist_config(broker, state):
    """Long sem config de trailing recebe o trailing stop PADRAO (universal)."""
    from config.watchlist import Watchlist, WatchlistItem

    wl = Watchlist(items=[WatchlistItem(symbol="NVDA")])  # sem trailing_stop_pct
    broker.seed_position("NVDA", qty=Decimal("3"), avg_entry_price=Decimal("100"))
    broker.set_price("NVDA", "120")
    strat = TrailingStopStrategy(default_trailing_stop_pct=Decimal("0.08"))
    intents = strat.evaluate(_ctx(broker, state, wl))
    assert len(intents) == 1
    assert intents[0].order_type == OrderType.TRAILING_STOP
    assert intents[0].trail_percent == Decimal("8")


def test_orphan_long_not_in_watchlist_gets_default_stop(broker, state):
    from config.watchlist import Watchlist, WatchlistItem

    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    broker.seed_position("ORPH", qty=Decimal("2"), avg_entry_price=Decimal("50"))
    broker.set_price("ORPH", "55")
    strat = TrailingStopStrategy(default_trailing_stop_pct=Decimal("0.10"))
    intents = strat.evaluate(_ctx(broker, state, wl))
    assert [i.symbol for i in intents] == ["ORPH"]


def test_ladder_symbol_is_not_trailed(broker, state):
    """Ativo gerido por ladder nao recebe trailing (ladder tem stop proprio)."""
    from config.watchlist import LadderConfig, LadderRung, Watchlist, WatchlistItem

    wl = Watchlist(
        items=[
            WatchlistItem(
                symbol="MSFT",
                ladder=LadderConfig(rungs=[LadderRung(drop_pct=Decimal("0.2"), qty=Decimal("5"))]),
            )
        ]
    )
    broker.seed_position("MSFT", qty=Decimal("5"), avg_entry_price=Decimal("300"))
    broker.set_price("MSFT", "320")
    strat = TrailingStopStrategy(default_trailing_stop_pct=Decimal("0.10"))
    assert strat.evaluate(_ctx(broker, state, wl)) == []


def test_mf1_beta_symbol_never_gets_trailing_stop(broker, state):
    """MF-1 (CRITICO): uma posicao de simbolo do book de BETA NUNCA recebe trailing
    stop deste pipeline-base. Sem a isencao, o trailing liquidaria o book do beta
    numa queda de 10% e corromperia o track record (auditoria MF-1).

    Prova: SPY fora da watchlist (como na watchlist shipada AAPL/MSFT) + SPY no
    set de excluidos => evaluate() NAO emite ordem de stop p/ SPY.
    """
    from config.watchlist import Watchlist, WatchlistItem
    from strategies.beta_rebalancer import UNIVERSE, to_broker_symbol

    excluded = {to_broker_symbol(a.ticker) for a in UNIVERSE}  # forma do broker
    assert "SPY" in excluded

    # watchlist shipada NAO tem SPY (so AAPL); posicao do beta criada pelo rebalancer.
    wl = Watchlist(items=[WatchlistItem(symbol="AAPL", trailing_stop_pct=Decimal("0.10"))])
    broker.seed_position("SPY", qty=Decimal("100"), avg_entry_price=Decimal("400"))
    broker.set_price("SPY", "500")

    strat = TrailingStopStrategy(Decimal("0.10"), excluded_symbols=excluded)
    intents = strat.evaluate(_ctx(broker, state, wl))
    assert intents == [], f"posicao do beta NAO deveria receber trailing: {intents}"
    assert all(i.symbol != "SPY" for i in intents)


def test_mf1_crypto_beta_symbol_excluded_even_24x7(broker, state):
    """MF-1: cripto do beta (BTC/USD) e isenta MESMO com o pregao fechado (cripto e
    24/7, entao o trailing dispararia sem o gate). Sem a isencao, o trailing de
    cripto venderia o book ate fora do pregao de acoes."""
    from config.watchlist import Watchlist
    from strategies.beta_rebalancer import UNIVERSE, to_broker_symbol

    excluded = {to_broker_symbol(a.ticker) for a in UNIVERSE}
    assert "BTC/USD" in excluded

    broker.set_market_open(False)  # pregao de ACOES fechado
    broker.seed_position("BTC/USD", qty=Decimal("0.5"), avg_entry_price=Decimal("40000"))
    broker.set_price("BTC/USD", "60000")

    strat = TrailingStopStrategy(Decimal("0.10"), excluded_symbols=excluded)
    intents = strat.evaluate(_ctx(broker, state, Watchlist(items=[])))
    assert intents == [], f"cripto do beta NAO deveria receber trailing 24/7: {intents}"


def test_mf1_non_beta_long_still_protected_with_exclusion_set(broker, state):
    """MF-1 nao deve afetar o book de ACOES: um long NAO-beta (orfa) ainda recebe o
    trailing padrao mesmo com o set de excluidos do beta presente."""
    from config.watchlist import Watchlist
    from strategies.beta_rebalancer import UNIVERSE, to_broker_symbol

    excluded = {to_broker_symbol(a.ticker) for a in UNIVERSE}
    assert "NVDA" not in excluded  # NVDA nao e do beta

    broker.seed_position("NVDA", qty=Decimal("3"), avg_entry_price=Decimal("100"))
    broker.set_price("NVDA", "120")
    strat = TrailingStopStrategy(Decimal("0.10"), excluded_symbols=excluded)
    intents = strat.evaluate(_ctx(broker, state, Watchlist(items=[])))
    assert [i.symbol for i in intents] == ["NVDA"]
    assert intents[0].order_type == OrderType.TRAILING_STOP


def test_full_protection_flow_places_once(broker, state, watchlist):
    """Em ciclos seguidos, coloca a trailing UMA vez e nao reenvia."""
    from agents.executor import Executor
    from core.kill_switch import KillSwitch
    from data.db import Database
    from data.trade_logger import TradeLogger

    broker.seed_position("AAPL", qty=Decimal("10"), avg_entry_price=Decimal("100"))
    broker.set_price("AAPL", "105")
    ks = KillSwitch("/tmp/__no_kill__")
    ex = Executor(broker, TradeLogger(Database(":memory:")), ks)
    strat = TrailingStopStrategy()

    # ciclo 1: coloca a trailing
    intents = strat.evaluate(_ctx(broker, state, watchlist))
    assert len(intents) == 1
    ex.execute(intents[0])
    # ciclo 2: ja existe trailing aberta no broker => nao reenvia
    assert strat.evaluate(_ctx(broker, state, watchlist)) == []
