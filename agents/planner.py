"""Agente Planejador.

Consome dados de mercado (via estrategias) e produz intencoes de ordem.
No MVP roda a(s) estrategia(s) configurada(s) e agrega as OrderIntent.
Nao executa nada — apenas decide.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from broker.base import BrokerClient
from config.watchlist import Watchlist
from core.models import OptionOrderIntent, Signal
from core.rounding import round_price, round_qty
from data.audit_log import PLANNER, AuditLog
from data.signal_repo import SignalRepository
from data.state_repo import StateRepository
from feedback.models import MarketRegime
from feedback.regime import classify_regime
from risk.manager import RiskManager, is_risk_increasing
from strategies.base import Strategy, StrategyContext, TradeIntent
from strategies.signals.base import SignalService

logger = logging.getLogger("agent.planner")


class Planner:
    def __init__(
        self,
        broker: BrokerClient,
        state: StateRepository,
        watchlist: Watchlist,
        strategies: list[Strategy],
        *,
        signal_service: SignalService | None = None,
        signal_repo: SignalRepository | None = None,
        risk_manager: RiskManager | None = None,
        audit: AuditLog | None = None,
        enricher=None,
        assumed_stop_pct: Decimal = Decimal("0.10"),
        bars_limit: int = 60,
    ) -> None:
        self._broker = broker
        self._state = state
        self._watchlist = watchlist
        self._strategies = strategies
        self._signal_service = signal_service
        self._signal_repo = signal_repo
        self._risk = risk_manager
        self._audit = audit
        # DecisionEnricher opcional: computa a CONFIANCA (regime + sinais + ML)
        # que dirige o sizing por conviccao no RiskManager. Ausente => sizing
        # fixo legado. Sem FIMATHE no caminho vivo (nao passamos OHLC).
        self._enricher = enricher
        self._assumed_stop_pct = assumed_stop_pct
        self._bars_limit = bars_limit
        # Sinais coletados no ciclo corrente (gather_signals), reaproveitados no
        # sizing por conviccao sem recoletar. Ambos os orquestradores chamam
        # gather_signals() ANTES de plan() no mesmo ciclo.
        self._last_signals: list[Signal] = []

    def gather_signals(self) -> list[Signal]:
        """Coleta sinais dos providers e os registra como SUGESTOES.

        IMPORTANTE: sinais NUNCA sao convertidos em OrderIntent aqui. Eles sao
        apenas logados e persistidos para revisao humana. Por construcao, o
        retorno deste metodo nao alimenta o Executor — a separacao entre
        `gather_signals()` (sugestao) e `plan()` (execucao) e proposital.
        """
        if self._signal_service is None:
            self._last_signals = []
            return []
        signals = self._signal_service.collect()
        self._last_signals = signals
        for s in signals:
            logger.info(
                "SUGESTAO [%s] %s %s conf=%.2f — %s",
                s.source, s.side.value, s.symbol, s.confidence, s.note,
            )
            if self._signal_repo is not None:
                self._signal_repo.record(s)
        return signals

    def plan(self) -> list[TradeIntent]:
        # Estado do pregao de acoes p/ as estrategias gatearem ativos de acao
        # fora de hora (cripto, 24/7, ignora). Falha de rede => assume fechado
        # (fail-safe: nao gera ordem de acao sem confirmar que o pregao esta aberto).
        try:
            equity_open = self._broker.is_market_open()
        except Exception as exc:
            logger.warning("Falha ao consultar clock de mercado; assumindo fechado: %s", exc)
            equity_open = False
        ctx = StrategyContext(
            broker=self._broker,
            state=self._state,
            watchlist=self._watchlist,
            equity_market_open=equity_open,
        )
        intents: list[TradeIntent] = []
        for strategy in self._strategies:
            try:
                produced = strategy.evaluate(ctx)
            except Exception:  # uma estrategia com erro nao derruba as demais
                logger.exception("Estrategia '%s' falhou ao avaliar", strategy.name)
                continue
            if produced:
                logger.info("Estrategia '%s' gerou %d intencao(oes)", strategy.name, len(produced))
            intents.extend(produced)

        # Camada de decisao/risco: veta/ajusta cada intencao antes de sair.
        if self._risk is not None:
            intents = self._apply_risk(intents)
        return intents

    def _apply_risk(self, intents: list[TradeIntent]) -> list[TradeIntent]:
        account = self._broker.get_account()
        positions = self._broker.get_positions()

        halt = self._risk.begin_cycle(account.equity)
        if halt:
            self._audit_risk("risk_halt", None, {"reason": halt})

        # Cache de preco/regime por simbolo DENTRO do ciclo: varios intents do
        # mesmo ativo nao disparam chamadas REST repetidas (mesma cotacao/barras).
        price_cache: dict[str, object] = {}
        regime_cache: dict[str, MarketRegime] = {}

        approved: list[TradeIntent] = []
        for intent in intents:
            symbol = intent.symbol if hasattr(intent, "symbol") else intent.underlying
            if symbol not in price_cache:
                price_cache[symbol] = self._safe_price(symbol)
            price = price_cache[symbol]
            item = self._watchlist.get(symbol)
            fractional = bool(item.fractional) if item is not None else False

            # Confianca (regime + sinais + ML) apenas para intents que ABREM
            # risco — saidas/protecoes passam direto e nao precisam de sizing.
            confidence = None
            if self._enricher is not None and is_risk_increasing(intent):
                regime = self._regime_for(symbol, regime_cache)
                confidence, ctx = self._confidence_for(
                    intent, symbol, price, account.equity, regime
                )
                if confidence is not None:
                    self._audit_risk(
                        "decision_confidence", symbol,
                        {"confidence": round(confidence, 4), "regime": regime.value, **ctx},
                    )

            decision = self._risk.assess(
                intent, account.equity, positions, price,
                fractional=fractional, confidence=confidence,
            )
            if decision.approved and decision.intent is not None:
                if decision.intent is not intent:
                    self._audit_risk("risk_adjusted", symbol, {"reason": decision.reason})
                final = self._round_to_market_steps(decision.intent)
                if final is None:
                    self._audit_risk("risk_vetoed", symbol, {"reason": "qty arredondada para 0 (lote)"})
                    continue
                approved.append(final)
            else:
                logger.info("Risco vetou %s: %s", symbol, decision.reason)
                self._audit_risk("risk_vetoed", symbol, {"reason": decision.reason})
        return approved

    def _round_to_market_steps(self, intent):
        """Alinha qty/limit ao tick/lote do ativo (cripto/fracionado). Opcoes
        passam direto (contratos sao inteiros e ja vem do broker)."""
        if isinstance(intent, OptionOrderIntent):
            return intent
        item = self._watchlist.get(intent.symbol)
        if item is None or (item.lot_size is None and item.tick_size is None and not item.fractional):
            return intent  # sem metadados de mercado: comportamento legado
        new_qty = round_qty(intent.qty, item.lot_size, fractional=item.fractional)
        if new_qty <= 0:
            return None
        new_limit = round_price(intent.limit_price, item.tick_size)
        if new_qty == intent.qty and new_limit == intent.limit_price:
            return intent
        return intent.model_copy(update={"qty": new_qty, "limit_price": new_limit})

    def _regime_for(self, symbol: str, cache: dict[str, MarketRegime]) -> MarketRegime:
        """Regime do ativo a partir de BARRAS REAIS (confiavel ja no 1o ciclo).
        Cacheado por simbolo no ciclo. Falha de rede => UNKNOWN (fail-safe)."""
        if symbol in cache:
            return cache[symbol]
        try:
            closes = [float(c) for c in self._broker.get_bars(symbol, self._bars_limit)]
            regime = classify_regime(closes)
        except Exception as exc:
            logger.warning("Falha ao obter barras de %s p/ regime: %s", symbol, exc)
            regime = MarketRegime.UNKNOWN
        cache[symbol] = regime
        return regime

    def _confidence_for(
        self, intent, symbol: str, price, equity, regime: MarketRegime
    ):
        """Confianca [0..1] da decisao via DecisionEnricher (regime + sinais +
        ML), SEM FIMATHE (ohlc=None). Devolve (confidence, context_p/_log).
        Qualquer falha cai no sizing fixo (confidence None) — nunca derruba o ciclo."""
        if price is None or equity <= 0:
            return None, {}
        try:
            stop = price * (Decimal(1) - self._assumed_stop_pct)
            enriched = self._enricher.enrich(
                symbol=symbol,
                side=intent.side,
                strategy=intent.strategy,
                regime=regime,
                equity=equity,
                entry_price=price,
                stop_price=stop,
                signals=self._last_signals,
                ohlc=None,  # sem FIMATHE no caminho vivo (decisao de escopo)
            )
        except Exception:
            logger.exception("Enricher falhou para %s; sizing cai no fixo", symbol)
            return None, {}
        return enriched.confidence, enriched.context

    def _safe_price(self, symbol: str):
        try:
            return self._broker.get_last_price(symbol)
        except Exception as exc:
            logger.warning("Falha ao obter preco de %s p/ risco: %s", symbol, exc)
            return None

    def _audit_risk(self, event: str, symbol: str | None, payload: dict) -> None:
        if self._audit is not None:
            self._audit.write(PLANNER, event, symbol=symbol, payload=payload)
