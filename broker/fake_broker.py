"""Implementacao em memoria do BrokerClient para testes.

Permite exercitar agentes e estrategias sem nenhuma chamada de rede. Simula
contas, posicoes, precos e preenchimento imediato de ordens a mercado.
"""

from __future__ import annotations

import itertools
from datetime import date, timedelta
from decimal import Decimal

from broker.base import AccountInfo, BrokerClient, BrokerOrder, MarketClockInfo
from core.models import (
    OptionContract,
    OptionOrderIntent,
    OptionType,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
)


class FakeBroker(BrokerClient):
    """Broker simulado, deterministico, para uso em testes."""

    def __init__(
        self,
        *,
        cash: Decimal = Decimal("100000"),
        prices: dict[str, Decimal] | None = None,
        options_level: int = 0,
        market_open: bool = True,
        today: date | None = None,
        partial_fill_ratio: Decimal | float | None = None,
        fail_after_record: bool = False,
        buying_power: Decimal | float | None = None,
        margin_semantics: bool = False,
    ) -> None:
        self._cash = cash
        # buying_power explicito (modela conta de MARGEM, em que o poder de compra
        # excede o caixa). None => legado: buying_power = cash (conta a vista).
        self._buying_power = (
            Decimal(str(buying_power)) if buying_power is not None else None
        )
        # SEMANTICA DE MARGEM REAL (dois pools), opt-in p/ os testes do leg de
        # cripto. Quando ligada, o fake modela a Alpaca fielmente:
        #   - acoes/ETFs (marginaveis): consomem o pool MARGINAVEL (buying_power);
        #     o caixa cai pelo NOTIONAL CHEIO (colateral), mas o BP cai so pelo
        #     notional (margem 1x p/ simplificar — o que importa e que o caixa
        #     some quando o book de acoes deploya);
        #   - cripto (cash-only, 1x): avaliada contra o CAIXA NAO-MARGINAVEL. Se o
        #     notional > caixa disponivel, a Alpaca REJEITA server-side -> aqui o
        #     submit LEVANTA (modela "insufficient non-marginable buying power").
        # Desligada (default): comportamento legado de pool unico (testes antigos).
        self._margin_semantics = margin_semantics
        # Pool marginavel vivo (decrementa a cada fill de acao quando margin_semantics).
        self._margin_bp = self._buying_power if self._buying_power is not None else cash
        self._prices: dict[str, Decimal] = dict(prices or {})
        self._positions: dict[str, Position] = {}
        self._bars: dict[str, list[Decimal]] = {}
        self._options_level = options_level
        self._market_open = market_open
        # Injecao de falhas p/ testar os ramos do caminho de ordem:
        # - partial_fill_ratio: fracao (0..1) preenchida em ordens a mercado
        #   (status "partially_filled" em vez de "filled").
        # - fail_after_record: registra a ordem no broker e LEVANTA (simula
        #   timeout: o broker tem a ordem, mas o cliente nao ve a resposta).
        self._partial_fill_ratio = (
            Decimal(str(partial_fill_ratio)) if partial_fill_ratio is not None else None
        )
        self._fail_after_record = fail_after_record
        self._next_open = None
        self._next_close = None
        self._today = today or date(2026, 6, 15)
        self._order_ids = itertools.count(1)
        self.submitted: list[OrderIntent] = []  # historico p/ asserts em testes
        self.submitted_options: list[OptionOrderIntent] = []
        # Idempotencia + reconciliacao: ordens conhecidas por client_order_id.
        self._orders_by_cid: dict[str, BrokerOrder] = {}
        self._open_orders: dict[str, BrokerOrder] = {}  # por broker_order_id

    # --- helpers de teste ---------------------------------------------------
    def set_price(self, symbol: str, price: Decimal | str | float) -> None:
        self._prices[symbol.upper()] = Decimal(str(price))

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    def set_options_level(self, level: int) -> None:
        self._options_level = level

    def set_fail_after_record(self, fail: bool) -> None:
        self._fail_after_record = fail

    def set_partial_fill_ratio(self, ratio: Decimal | float | None) -> None:
        self._partial_fill_ratio = Decimal(str(ratio)) if ratio is not None else None

    def seed_position(self, symbol: str, qty: Decimal, avg_entry_price: Decimal) -> None:
        symbol = symbol.upper()
        self._positions[symbol] = Position(
            symbol=symbol, qty=qty, avg_entry_price=avg_entry_price
        )

    # --- BrokerClient -------------------------------------------------------
    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            (p.qty * self._prices.get(p.symbol, p.avg_entry_price) for p in self._positions.values()),
            Decimal(0),
        )
        # Pool marginavel: em margin_semantics e o BP vivo (ja decrementado pelos
        # fills de acao); senao o buying_power estatico (ou o caixa, legado).
        if self._margin_semantics:
            buying_power = self._margin_bp
        else:
            buying_power = self._buying_power if self._buying_power is not None else self._cash
        # Pool NAO-MARGINAVEL (cripto cash-only) = caixa liquidado disponivel.
        # Nunca negativo (a Alpaca nao da buying power de cripto sobre caixa devedor).
        nmbp = self._cash if self._cash > 0 else Decimal(0)
        return AccountInfo(
            cash=self._cash,
            buying_power=buying_power,
            equity=equity,
            options_level=self._options_level,
            non_marginable_buying_power=nmbp,
        )

    def get_positions(self) -> list[Position]:
        result = []
        for p in self._positions.values():
            price = self._prices.get(p.symbol)
            result.append(
                Position(
                    symbol=p.symbol,
                    qty=p.qty,
                    avg_entry_price=p.avg_entry_price,
                    current_price=price,
                )
            )
        return result

    def get_position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        p = self._positions.get(symbol)
        if p is None:
            return None
        return Position(
            symbol=p.symbol,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            current_price=self._prices.get(symbol),
        )

    def get_last_price(self, symbol: str) -> Decimal:
        symbol = symbol.upper()
        if symbol not in self._prices:
            raise KeyError(f"FakeBroker sem preco para {symbol}; use set_price()")
        return self._prices[symbol]

    def set_bars(self, symbol: str, closes) -> None:
        self._bars[symbol.upper()] = [Decimal(str(c)) for c in closes]

    def get_bars(self, symbol: str, limit: int = 60, *, timeframe: str = "1Day") -> list[Decimal]:
        # timeframe ignorado no fake (barras pre-carregadas via set_bars).
        bars = self._bars.get(symbol.upper(), [])
        return list(bars[-limit:])

    def submit_order(self, intent: OrderIntent) -> OrderResult:
        # Idempotencia: mesmo client_order_id => devolve a ordem existente sem
        # reaplicar o fill (simula a rejeicao de duplicado pelo broker).
        cid = intent.client_order_id
        if cid is not None and cid in self._orders_by_cid:
            return self._as_result(self._orders_by_cid[cid])

        self.submitted.append(intent)
        broker_id = f"fake-{next(self._order_ids)}"

        # Trailing stop nativo: fica ABERTO no broker (nao preenche na hora).
        if intent.order_type == OrderType.TRAILING_STOP:
            order = BrokerOrder(
                broker_order_id=broker_id,
                client_order_id=cid,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                filled_qty=Decimal(0),
                status="new",
                order_type="trailing_stop",
            )
            if cid:
                self._orders_by_cid[cid] = order
            self._open_orders[broker_id] = order
            return self._as_result(order)

        fill_price = self._prices.get(intent.symbol, intent.limit_price or Decimal("1"))

        # SEMANTICA DE MARGEM REAL (dois pools), so quando ligada. A ordem JA foi
        # registrada no broker (self.submitted) — modelamos a REJEICAO SERVER-SIDE
        # da Alpaca levantando aqui (igual ao que o Coder reproduziu: a ordem chega
        # ao broker e e rejeitada, nao barrada no gate do cliente).
        if self._margin_semantics and intent.side == OrderSide.BUY:
            self._enforce_two_pools(intent, fill_price)

        # Timeout pos-registro: o broker JA conhece a ordem (idempotencia futura
        # a reencontra), mas o cliente recebe uma excecao em vez da resposta.
        if self._fail_after_record:
            pending = BrokerOrder(
                broker_order_id=broker_id,
                client_order_id=cid,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                filled_qty=Decimal(0),
                status="accepted",
                order_type=intent.order_type.value,
            )
            if cid:
                self._orders_by_cid[cid] = pending
            self._open_orders[broker_id] = pending
            raise RuntimeError("timeout simulado apos o broker registrar a ordem")

        # Fill total (default) ou parcial (quando partial_fill_ratio esta setado).
        if self._partial_fill_ratio is not None and Decimal(0) < self._partial_fill_ratio < Decimal(1):
            filled_qty = intent.qty * self._partial_fill_ratio
            status = "partially_filled"
        else:
            filled_qty = intent.qty
            status = "filled"

        if filled_qty > 0:
            self._apply_fill(intent.model_copy(update={"qty": filled_qty}), fill_price)
        order = BrokerOrder(
            broker_order_id=broker_id,
            client_order_id=cid,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=intent.qty,
            filled_qty=filled_qty,
            status=status,
            order_type=intent.order_type.value,
            filled_avg_price=fill_price,
        )
        if cid:
            self._orders_by_cid[cid] = order
        return OrderResult(
            broker_order_id=broker_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=filled_qty,
            filled_avg_price=fill_price,
            status=status,
            client_order_id=cid,
        )

    @staticmethod
    def _as_result(order: BrokerOrder) -> OrderResult:
        return OrderResult(
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=OrderSide(order.side),
            qty=order.qty,
            filled_qty=order.filled_qty,
            filled_avg_price=order.filled_avg_price,
            status=order.status,
            client_order_id=order.client_order_id,
        )

    def seed_open_order(self, order: BrokerOrder) -> None:
        """Insere uma ordem aberta no broker (p/ testes de reconciliacao)."""
        self._open_orders[order.broker_order_id] = order
        if order.client_order_id:
            self._orders_by_cid[order.client_order_id] = order

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Cripto na Alpaca = par com barra ('BTC/USD'). Acoes/ETFs nao tem '/'."""
        return "/" in symbol

    def _enforce_two_pools(self, intent: OrderIntent, fill_price: Decimal) -> None:
        """Modela a checagem de pool REAL da Alpaca p/ uma COMPRA (margin_semantics).

        - CRIPTO (cash-only, 1x): avaliada contra o CAIXA NAO-MARGINAVEL. Se o
          notional > caixa disponivel, a Alpaca REJEITA server-side. Aqui levanta
          (= o leg de cripto que o Coder viu cair). Este e o gate que o fix deve
          PASSAR ao reservar caixa p/ a cripto ANTES das acoes.
        - ACAO/ETF (marginavel): avaliada contra o pool MARGINAVEL (buying_power).
          O fill decrementa o pool marginavel (em _apply_fill); o caixa cai pelo
          notional cheio (colateral), podendo ficar negativo (emprestado)."""
        notional = intent.qty * fill_price
        if self._is_crypto(intent.symbol):
            available = self._cash if self._cash > 0 else Decimal(0)
            if notional > available:
                raise RuntimeError(
                    f"insufficient non-marginable buying power: cripto {intent.symbol} "
                    f"custo~{notional} > caixa {available} (cash-only 1x)"
                )
        else:
            if notional > self._margin_bp:
                raise RuntimeError(
                    f"insufficient buying power: {intent.symbol} custo~{notional} > "
                    f"buying_power {self._margin_bp}"
                )

    def _apply_fill(self, intent: OrderIntent, fill_price: Decimal) -> None:
        symbol = intent.symbol
        signed = intent.qty if intent.side == OrderSide.BUY else -intent.qty
        existing = self._positions.get(symbol)
        if existing is None:
            new_qty = signed
            avg = fill_price
        else:
            new_qty = existing.qty + signed
            if intent.side == OrderSide.BUY:
                total_cost = existing.qty * existing.avg_entry_price + intent.qty * fill_price
                avg = total_cost / new_qty if new_qty != 0 else fill_price
            else:
                avg = existing.avg_entry_price
        cash_delta = -signed * fill_price
        self._cash += cash_delta
        # SEMANTICA DE MARGEM REAL: uma COMPRA de acao/ETF (marginavel) consome o
        # pool marginavel; uma VENDA o devolve. Cripto NAO toca o pool marginavel
        # (cash-only — ja foi debitada do caixa via cash_delta acima).
        if self._margin_semantics and not self._is_crypto(symbol):
            self._margin_bp -= signed * fill_price
        if new_qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                symbol=symbol, qty=new_qty, avg_entry_price=avg
            )

    def cancel_all_orders(self) -> int:
        n = len(self._open_orders)
        self._open_orders.clear()
        return n

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self._open_orders.values())

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return self._orders_by_cid.get(client_order_id)

    def is_market_open(self) -> bool:
        return self._market_open

    def get_clock(self) -> "MarketClockInfo":
        from datetime import datetime, time, timezone

        ts = datetime.combine(self._today, time(12, 0), tzinfo=timezone.utc)
        return MarketClockInfo(
            is_open=self._market_open,
            next_open=self._next_open,
            next_close=self._next_close,
            timestamp=ts,
        )

    def set_clock(self, *, next_open=None, next_close=None) -> None:
        self._next_open = next_open
        self._next_close = next_close

    # --- Opcoes -------------------------------------------------------------
    def select_option_contract(
        self,
        underlying: str,
        option_type: OptionType,
        target_strike: Decimal,
        *,
        min_dte: int,
        max_dte: int,
    ) -> OptionContract | None:
        # Sintetiza um contrato: strike arredondado ao inteiro mais proximo e
        # vencimento no meio da janela de DTE pedida. Suficiente para exercitar
        # a logica da Wheel sem uma cadeia de opcoes real.
        underlying = underlying.upper()
        strike = Decimal(round(target_strike))
        dte = (min_dte + max_dte) // 2
        expiration = self._today + timedelta(days=dte)
        occ = (
            f"{underlying}{expiration:%y%m%d}"
            f"{'P' if option_type == OptionType.PUT else 'C'}"
            f"{int(strike) * 1000:08d}"
        )
        return OptionContract(
            occ_symbol=occ,
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
        )

    def submit_option_order(self, intent: OptionOrderIntent) -> OrderResult:
        cid = intent.client_order_id
        if cid is not None and cid in self._orders_by_cid:
            return self._as_result(self._orders_by_cid[cid])  # idempotente

        self.submitted_options.append(intent)
        # Premio por acao: o limit_price (premio-alvo) quando a ordem e LIMIT;
        # senao um premio simulado de 1% do strike. (100 acoes por contrato.)
        premium_per_share = (
            intent.limit_price
            if intent.order_type == OrderType.LIMIT and intent.limit_price is not None
            else intent.contract.strike * Decimal("0.01")
        )
        premium = premium_per_share * Decimal(100) * intent.qty
        if intent.side == OrderSide.SELL:
            self._cash += premium  # vender premio credita caixa
        broker_id = f"fake-opt-{next(self._order_ids)}"
        if cid:
            self._orders_by_cid[cid] = BrokerOrder(
                broker_order_id=broker_id,
                client_order_id=cid,
                symbol=intent.contract.occ_symbol,
                side=intent.side.value,
                qty=intent.qty,
                filled_qty=intent.qty,
                status="filled",
                order_type="option",
            )
        return OrderResult(
            broker_order_id=broker_id,
            symbol=intent.contract.occ_symbol,
            side=intent.side,
            qty=intent.qty,
            filled_qty=intent.qty,
            filled_avg_price=premium_per_share,
            status="filled",
            client_order_id=cid,
        )
