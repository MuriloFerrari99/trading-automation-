"""SimBroker — broker de backtest bar-a-bar, SEM look-ahead (doc 06 §2).

Reaproveita as estrategias REAIS (TrailingStop, Ladder) via a mesma interface
que elas usam (get_position/get_last_price/get_open_orders/submit_order), entao
o que medimos e o comportamento do modelo de verdade, nao uma reimplementacao.

Regras anti-look-ahead e realismo:
- Decisao no fechamento do bar t; ordens a mercado preenchem no OPEN de t+1.
- Comissao (bps) e slippage (bps) aplicados em todo fill (nunca custo zero).
- Trailing stop NATIVO emulado: o broker rastreia o high-water (como a Alpaca
  faria) e dispara a venda quando o fechamento cai trail% abaixo da maxima;
  a venda preenche no open seguinte.

Nao subclasse de BrokerClient de proposito (duck typing): implementa so o que
as estrategias chamam, mantendo o engine enxuto.
"""

from __future__ import annotations

from decimal import Decimal

from broker.base import BrokerOrder
from core.models import OrderResult, OrderSide, OrderType, Position


class SimBroker:
    def __init__(
        self,
        symbol: str,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        *,
        cash: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self._symbol = symbol.upper()
        self._open = opens
        self._high = highs
        self._low = lows
        self._close = closes
        self._cash = Decimal(str(cash))
        self._commission = Decimal(str(commission_bps)) / Decimal(10000)
        self._slippage = Decimal(str(slippage_bps)) / Decimal(10000)

        self._i = 0
        self._qty = Decimal(0)
        self._avg = Decimal(0)
        self._pending: list = []  # ordens a mercado p/ preencher no proximo open
        self._trailing: dict | None = None  # {high_water, trail_pct, qty}

        self.trade_pnls: list[float] = []
        self.equity_curve: list[float] = []
        self.bars_in_market = 0

    # --- controle do engine -------------------------------------------------
    def set_index(self, i: int) -> None:
        self._i = i

    def fill_pending_at_open(self, i: int) -> None:
        """Preenche ordens enfileiradas no bar anterior, no OPEN do bar i."""
        if not self._pending:
            return
        open_px = Decimal(str(self._open[i]))
        for intent in self._pending:
            if intent.side == OrderSide.BUY:
                self._fill_buy(open_px, intent.qty)
            else:
                self._fill_sell(open_px, intent.qty)
        self._pending = []

    def update_trailing_and_maybe_trigger(self) -> None:
        """Atualiza high-water com o fechamento atual; se romper, enfileira venda."""
        if self._trailing is None or self._qty <= 0:
            return
        close = Decimal(str(self._close[self._i]))
        if close > self._trailing["high_water"]:
            self._trailing["high_water"] = close
        stop = self._trailing["high_water"] * (Decimal(1) - self._trailing["trail_pct"])
        if close <= stop:
            self._pending.append(_MarketSell(self._symbol, self._qty))
            self._trailing = None

    def mark_equity(self) -> None:
        close = Decimal(str(self._close[self._i]))
        equity = self._cash + self._qty * close
        self.equity_curve.append(float(equity))
        if self._qty > 0:
            self.bars_in_market += 1

    def liquidate_final(self) -> None:
        """Fecha qualquer posicao remanescente no ultimo fechamento (saida limpa)."""
        if self._qty > 0:
            self._fill_sell(Decimal(str(self._close[self._i])), self._qty)

    # --- interface usada pelas estrategias ----------------------------------
    def get_last_price(self, symbol: str) -> Decimal:
        return Decimal(str(self._close[self._i]))

    def get_position(self, symbol: str) -> Position | None:
        if self._qty <= 0:
            return None
        return Position(
            symbol=self._symbol, qty=self._qty, avg_entry_price=self._avg,
            current_price=Decimal(str(self._close[self._i])),
        )

    def get_positions(self) -> list[Position]:
        p = self.get_position(self._symbol)
        return [p] if p else []

    def get_open_orders(self) -> list[BrokerOrder]:
        if self._trailing is None:
            return []
        return [BrokerOrder(
            broker_order_id="sim-trail", client_order_id=None, symbol=self._symbol,
            side="sell", qty=self._trailing["qty"], filled_qty=Decimal(0),
            status="new", order_type="trailing_stop",
        )]

    def get_bars(self, symbol: str, limit: int = 60) -> list[Decimal]:
        lo = max(0, self._i + 1 - limit)
        return [Decimal(str(c)) for c in self._close[lo : self._i + 1]]

    def submit_order(self, intent) -> OrderResult:
        if intent.order_type == OrderType.TRAILING_STOP:
            self._trailing = {
                "high_water": Decimal(str(self._close[self._i])),
                "trail_pct": (intent.trail_percent or Decimal(0)) / Decimal(100),
                "qty": intent.qty,
            }
        else:
            self._pending.append(intent)
        return OrderResult(
            broker_order_id="sim", symbol=self._symbol, side=intent.side, qty=intent.qty,
        )

    # --- contabilidade de fills ---------------------------------------------
    def _fill_buy(self, ref_px: Decimal, qty: Decimal) -> None:
        price = ref_px * (Decimal(1) + self._slippage)
        cost = price * qty
        commission = cost * self._commission
        self._cash -= cost + commission
        new_qty = self._qty + qty
        self._avg = (self._avg * self._qty + price * qty) / new_qty if new_qty > 0 else price
        self._qty = new_qty

    def _fill_sell(self, ref_px: Decimal, qty: Decimal) -> None:
        price = ref_px * (Decimal(1) - self._slippage)
        proceeds = price * qty
        commission = proceeds * self._commission
        self._cash += proceeds - commission
        # P&L realizado liquido da comissao de venda; base = custo medio.
        pnl = float(qty * (price - self._avg) - commission)
        self.trade_pnls.append(pnl)
        self._qty -= qty
        if self._qty <= 0:
            self._qty = Decimal(0)
            self._avg = Decimal(0)
            self._trailing = None


class _MarketSell:
    """Ordem interna de venda a mercado (gatilho do trailing)."""

    __slots__ = ("symbol", "qty", "side", "order_type")

    def __init__(self, symbol: str, qty: Decimal) -> None:
        self.symbol = symbol
        self.qty = qty
        self.side = OrderSide.SELL
        self.order_type = OrderType.MARKET
