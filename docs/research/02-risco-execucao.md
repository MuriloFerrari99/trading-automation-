---
name: risco-execucao
description: Referência completa e precisa sobre gestão de risco e execução de ordens para um bot de trading automatizado (Alpaca, paper trading first) — trailing stops, bracket/OCO, scaling-in, position sizing, risco de portfólio, kill switch, execução e idempotência.
metadata:
  type: reference
---

# Gestão de Risco e Execução de Ordens

> Documento de referência para o bot de trading multi-agente (Alpaca, paper trading primeiro).
> Termos técnicos mantidos em inglês; explicações em português.
> **Regra de ouro:** todo controle de risco descrito aqui deve ser validado em *paper trading* antes de qualquer capital real. Risco mata conta mais rápido do que sinal ruim.

---

## 0. Princípios gerais

1. **Risk-first, não return-first.** O *position sizing* e os limites de perda definem quanto você pode arriscar; o sinal (alpha) só decide *se* e *em que direção* você entra. Defina o risco antes de pensar em retorno. ([Medium — Position Sizing Frameworks](https://medium.com/@ildiveliu/risk-before-returns-position-sizing-frameworks-fixed-fractional-atr-based-kelly-lite-4513f770a82a))
2. **Defesa em camadas.** Stop por trade → limite de perda diária → circuit breaker de portfólio → kill switch manual/automático. Nenhuma camada sozinha é suficiente.
3. **O broker é a fonte de verdade.** O estado do bot é apenas uma *cache*. Sempre reconcilie contra o que a Alpaca reporta (posições, ordens, equity).
4. **Idempotência em tudo que envia ordem.** Uma reconexão, um retry de rede ou um restart não pode duplicar uma ordem.
5. **Fail-safe, não fail-open.** Em dúvida (erro de API, dado faltando, estado inconsistente), o bot deve *parar de abrir risco novo*, não continuar operando às cegas.

---

## 1. Trailing Stop

### 1.1 Definição

Um **trailing stop** é um stop dinâmico cujo preço de gatilho "persegue" o preço a uma distância fixa, definida em **valor absoluto** (`trail_price`) ou em **percentual** (`trail_percent`). Para uma posição comprada (long, sell trailing stop):

- O bot/broker mantém um **high water mark (hwm)** = maior preço atingido desde que a ordem foi submetida.
- O `stop_price` é recalculado continuamente:
  - Por valor: `stop_price = hwm - trail_price`
  - Por percentual: `stop_price = hwm * (1 - trail_percent/100)`
- **O stop sobe quando o preço sobe, mas NUNCA desce.** Se o preço cai, o hwm não muda, logo o stop fica parado. Isso trava o lucro acumulado e ainda dá espaço para o ativo respirar na alta.

Para uma posição vendida (short, buy trailing stop) é o espelho: rastreia o **low water mark**, e `stop_price = lwm + trail` — o stop só desce, nunca sobe.

Quando o preço toca o `stop_price`, a ordem **vira uma market order** (não há garantia de preço — pode preencher acima ou abaixo do gatilho, especialmente em gaps).

Fontes: [Alpaca — Trailing Stop Orders (blog)](https://alpaca.markets/blog/trailing-stop/), [Alpaca Docs — Placing Orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca), [Alpaca — 13 Order Types](https://alpaca.markets/learn/13-order-types-you-should-know-about).

### 1.2 Trail por % vs por valor

| | `trail_percent` | `trail_price` (valor) |
|---|---|---|
| Distância | Proporcional ao preço | Fixa em dólares |
| Bom para | Ativos de preços muito diferentes; normaliza por escala | Quando você quer um buffer monetário exato |
| Risco | Em ativos caros, % vira muito $ | Em ativos baratos, $ fixo pode ser % enorme |
| Recomendação | Default para a maioria dos casos; combine com ATR (ver §4) | Use quando o risco em $ por share é o que importa |

### 1.3 Server-side (nativo Alpaca) vs gerenciado pelo bot

#### (A) Nativo Alpaca (`type="trailing_stop"`) — server-side

A Alpaca rastreia o hwm e recalcula o stop nos servidores dela. Restrições: `time_in_force` deve ser `day` ou `gtc`; requer `trail_price` **ou** `trail_percent` (não ambos).

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(API_KEY, API_SECRET, paper=True)

# Trailing stop nativo: sai de uma posição long com trail de 3%
req = TrailingStopOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.SELL,
    time_in_force=TimeInForce.GTC,
    trail_percent=3.0,        # OU trail_price=2.50 (não os dois)
)
order = client.submit_order(req)
# No retorno, acompanhe order.hwm e order.stop_price
```

**Prós:** o stop continua valendo mesmo se o bot cair, perder conexão ou reiniciar; sem latência de polling; menos código.
**Contras:** menos flexível (não dá para usar lógica custom de trailing, ex.: trail que aperta perto de uma resistência, ou trailing baseado em ATR recalculado intraday); vira *market order* no gatilho (slippage em gaps); GTC nativo expira em 90 dias.

#### (B) Gerenciado pelo bot — client-side (manual)

O bot faz polling do preço, mantém o hwm em memória/DB, e quando o stop precisa subir ele **cancela e reenvia** (ou usa `replace`) um stop order comum. Necessário quando você quer trailing *custom* (ATR-based, time-based, condicional).

```python
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, ReplaceOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream  # ou polling REST de quotes

class ManualTrailingStop:
    """
    Trailing stop client-side para uma posição LONG.
    Mantém o high water mark e só MOVE O STOP PARA CIMA (nunca para baixo).
    """
    def __init__(self, client, symbol, qty, trail_pct):
        self.client = client
        self.symbol = symbol
        self.qty = qty
        self.trail_pct = trail_pct
        self.hwm = None            # high water mark
        self.stop_order_id = None  # id da stop order ativa no broker

    def _desired_stop(self):
        return round(self.hwm * (1 - self.trail_pct / 100), 2)

    def _place_stop(self, stop_price):
        req = StopOrderRequest(
            symbol=self.symbol, qty=self.qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, stop_price=stop_price,
            client_order_id=f"trail-{self.symbol}-{int(time.time()*1000)}",  # idempotência
        )
        return self.client.submit_order(req)

    def on_price(self, last_price):
        # 1) Atualiza o high water mark
        if self.hwm is None or last_price > self.hwm:
            self.hwm = last_price

        new_stop = self._desired_stop()

        # 2) Primeira colocação
        if self.stop_order_id is None:
            o = self._place_stop(new_stop)
            self.stop_order_id = o.id
            self.current_stop = new_stop
            return

        # 3) REGRA CENTRAL: só mexe se o stop SUBIU.
        #    Se new_stop <= current_stop, NÃO faz nada (stop nunca desce).
        if new_stop > self.current_stop:
            # replace é preferível a cancel+submit: menos chance de janela "sem stop"
            self.client.replace_order_by_id(
                self.stop_order_id,
                ReplaceOrderRequest(stop_price=new_stop),
            )
            self.current_stop = new_stop
```

**Prós:** lógica de trailing totalmente customizável; pode combinar com ATR, suporte/resistência, tempo, breakeven.
**Contras:** **se o bot cai, a proteção é só o stop atualmente colocado no broker** — ele não vai mais subir, mas pelo menos protege no nível atual; cuidado com a **janela sem stop** entre cancelar e reenviar (use `replace` em vez de cancel+submit; `replace` mantém a ordem ativa até a substituição vigorar); custo de polling/rate limit.

**Recomendação para este bot:** comece com o **trailing_stop nativo** da Alpaca para robustez (sobrevive a crash). Migre para client-side só quando precisar de lógica que o nativo não suporta (ex.: trail baseado em ATR recalculado a cada barra). Em ambos os casos, registre `hwm` e `stop_price` no DB para auditoria e reconciliação.

---

## 2. Stop-loss, take-profit, bracket orders e OCO

### 2.1 Conceitos

- **Stop-loss**: ordem que limita a perda. Quando o preço atinge `stop_price`, vira market (stop) ou limit (stop-limit). Stop-limit protege de slippage mas pode **não preencher** num gap — o ativo "fura" o limite e você fica preso na posição.
- **Take-profit**: limit order de saída no alvo de lucro.
- **Bracket order** (`order_class="bracket"`): ordem de entrada + **dois** filhos de saída (take-profit limit e stop-loss). Quando um filho preenche, o outro é cancelado automaticamente. Não suporta extended hours; TIF deve ser `day` ou `gtc`.
- **OCO** (`order_class="oco"`, *One-Cancels-Other*): só os dois filhos de saída, para quando a entrada **já está preenchida** (você já tem a posição). Os dois têm o mesmo lado.
- **OTO** (`order_class="oto"`, *One-Triggers-Other*): entrada + **um** filho (take-profit **ou** stop-loss, não os dois).

Fontes: [Alpaca — OCO & OTO](https://alpaca.markets/blog/oco-oto/), [Alpaca — Bracket Orders](https://alpaca.markets/blog/bracket-orders/), [Alpaca Docs — Placing Orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

### 2.2 Bracket order com alpaca-py

```python
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

req = MarketOrderRequest(
    symbol="MSFT",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,
    order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest(limit_price=520.00),   # alvo de lucro
    stop_loss=StopLossRequest(stop_price=480.00,         # stop
                              limit_price=479.50),        # opcional: vira stop-limit
)
order = client.submit_order(req)
```

**Gotcha conhecido:** os preços dos filhos precisam respeitar a relação correta (take_profit acima da entrada e stop_loss abaixo, para long) ou a ordem é rejeitada. Em bracket/OCO/OTO os campos `take_profit`/`stop_loss` exigidos não podem ser omitidos. Para inspecionar os filhos, use `nested=True` ao consultar a ordem (retorna `legs`). ([Trade Mamba — Bracket Orders gotcha](https://medium.com/@trademamba/bracket-orders-with-alpaca-markets-and-a-key-gotcha-6560d47ad6f4))

### 2.3 Bracket nativo vs gerência manual

Bracket nativo é a forma mais robusta de garantir que **toda posição nasce com stop e alvo**. A alternativa manual (submeter entrada, esperar fill, depois submeter OCO) tem uma janela de risco entre o fill da entrada e a colocação da proteção — evite. Regra do bot: **nenhuma entrada sem proteção atômica** (bracket) sempre que possível.

---

## 3. Ladder buys / scaling in (preço médio)

### 3.1 O que é

**Scaling in** = entrar numa posição em **vários degraus** em vez de tudo de uma vez. **Averaging down** = comprar mais conforme o preço **cai**, reduzindo o preço médio. É um **DCA agressivo discricionário**, diferente do DCA clássico (que compra valor fixo em intervalos de tempo, ignorando preço).

Variantes de dimensionamento de cada degrau ([Medium — Averaging Down strategies](https://medium.com/@FMZQuant/grid-dollar-cost-averaging-strategy-dbc5bbbc1574)):

- **Linear / Fixed amount**: mesmo valor em cada degrau (ex.: $1.000 a cada -3%).
- **Grid**: degraus em níveis de preço pré-definidos, frequentemente com **tamanho crescente** na queda.
- **Martingale**: dobra o tamanho a cada degrau — **muito perigoso**, consome capital exponencialmente; um trade ruim pode destruir a conta.
- **Reverse martingale / decreasing**: reduz o tamanho conforme cai — mais conservador, limita o capital comprometido na cauda.

### 3.2 Averaging down vs grid trading

| | Averaging down (DCA agressivo) | Grid trading |
|---|---|---|
| Tese | Direcional (acredito que vai subir); acumular barato | Neutro/range; lucrar com oscilação |
| Saída | Vende tudo num alvo de recuperação | Vende cada lote num degrau acima do compra |
| Quando funciona | Mercado **recupera** depois | Mercado **lateraliza/oscila** num range |
| Risco principal | *Catching a falling knife*: tendência de baixa prolongada → preço médio acima do mercado, perda crescente | Tendência forte (rompe o range) → fica comprado caindo ou perde a alta |

Fontes: [TradeLink — Grid vs DCA](https://tradelink.pro/blog/grid-trading-vs-dca/), [Gainium — DCA vs Grid vs Combo](https://gainium.io/blog/dca-vs-grid-vs-combo-bots-choosing-the-right-strategy).

### 3.3 Como dimensionar os degraus (boas práticas)

1. **Defina o capital MÁXIMO da escada antes de começar** (full allocation). Cada degrau é uma fração desse total — você nunca pode ser obrigado a aportar além do que reservou.
2. **Espaçamento por volatilidade, não arbitrário.** Use múltiplos de ATR (ex.: degrau a cada 1×ATR de queda) em vez de % fixo, para o range se adaptar ao ativo.
3. **Limite o número de degraus** (ex.: 4–5). Mais degraus = mais convicção de que é range, não tendência.
4. **Pare a escada em invalidação de tese**, não em preço. Se o motivo do trade morreu (quebrou suporte estrutural, mudou o fundamento), **o stop final mata a escada inteira** — não continue comprando.
5. **Reverse-martingale ou linear**, evite martingale puro em produção.

### 3.4 Riscos e quando faz sentido

- **Catching a falling knife**: o maior risco. Em *bear market* sustentado, a escada baixa o preço médio mas a perda total **cresce** porque você comprou mais na descida. ([Medium — Grid DCA](https://medium.com/@FMZQuant/grid-dollar-cost-averaging-strategy-dbc5bbbc1574))
- **Sem stop global = ruína.** Toda escada precisa de um **stop final** que liquida tudo se a tese quebrar.
- **Capital travado**: scaling in compromete caixa que poderia ir para outras oportunidades.
- **Faz sentido** quando: ativo de qualidade em correção (não em colapso), range identificável, capital total pré-orçado, e há **stop de invalidação** definido. **Não faz sentido** como "esperança" de recuperação sem tese.

---

## 4. Position sizing

A pergunta central: *quantas shares / quanto $ por trade?* Quatro abordagens, do mais simples ao mais avançado.

### 4.1 Fixed fractional / % de risco por trade (regra do 1–2%)

Arrisque uma fração fixa do equity por trade (tipicamente **1–2%**). A distância até o stop define o tamanho:

```
risk_$         = equity * risk_pct            # ex.: 100k * 0.01 = $1.000
risk_per_share = entry_price - stop_price     # risco por unidade (long)
qty            = floor(risk_$ / risk_per_share)
```

Arriscar mais de 2% por trade aumenta muito a probabilidade de *drawdown* grande e ruína. ([Medium — Position Sizing](https://medium.com/@ildiveliu/risk-before-returns-position-sizing-frameworks-fixed-fractional-atr-based-kelly-lite-4513f770a82a))

### 4.2 ATR / volatility-based sizing

Em vez de um stop arbitrário, posiciona o stop a um múltiplo do **ATR** e dimensiona para que o risco em $ seja constante. Resultado: **menos shares quando a volatilidade é alta, mais quando é baixa** — normaliza o risco entre ativos e regimes. ([ActivTrades — Volatility-based sizing](https://www.activtrades.com/en/news/how-to-use-volatility-based-position-sizing-when-trading-with-cfds))

```python
def atr_position_size(equity, risk_pct, atr, atr_mult=2.0):
    """Dimensiona para arriscar risk_pct do equity, com stop a atr_mult * ATR."""
    risk_dollars   = equity * risk_pct
    risk_per_share = atr_mult * atr          # distância do stop em $
    qty            = int(risk_dollars // risk_per_share)
    return max(qty, 0)

# Ex.: equity=100_000, risco 1%, ATR=$2.50, stop a 2*ATR
# risk_dollars = 1.000 ; risk_per_share = 5.00 ; qty = 200 shares
```

### 4.3 Kelly criterion (e por que usar uma fração dele)

O **Kelly criterion** dá o tamanho de aposta que maximiza o crescimento geométrico de longo prazo, dado win rate e payoff. Para um ativo com probabilidade de ganho `p`, perda `q = 1-p` e razão ganho/perda `b`:

```
f* = p - q / b          # fração ótima do capital (Kelly completo)
```

([QuantStrategy — Kelly](https://quantstrategy.io/blog/applying-the-kelly-criterion-to-trading-maximizing-growth/))

**Por que NUNCA usar Kelly completo em produção:**
- Kelly assume que você conhece `p` e `b` com **precisão** — em trading eles são **estimados** e instáveis. Erro na estimativa → over-betting → drawdowns brutais.
- Kelly completo gera volatilidade altíssima e *drawdowns* que poucos toleram psicologicamente. A prática é **fractional Kelly: 25%–50% do f\*** (half-Kelly ou quarter-Kelly), reduzindo drasticamente a volatilidade com pouca perda de crescimento. ([JournalPlus — Kelly Guide](https://journalplus.co/learn/guides/kelly-criterion-guide/))

```python
def fractional_kelly(win_rate, win_loss_ratio, kelly_fraction=0.25, cap=0.02):
    """
    win_rate (p): prob. de trade vencedor (0..1)
    win_loss_ratio (b): ganho médio / perda média
    kelly_fraction: fração do Kelly cheio (0.25 = quarter-Kelly)
    cap: teto duro de fração do equity por trade (segurança)
    """
    p, q, b = win_rate, 1 - win_rate, win_loss_ratio
    f_full = p - q / b                 # Kelly completo
    f = max(0.0, f_full) * kelly_fraction
    return min(f, cap)                 # nunca acima do cap (ex.: 2%)
```

> **Default recomendado:** comece com **fixed fractional 1%** + stop por ATR. Kelly só depois de ter histórico estatístico **confiável** de `p` e `b` — e sempre fracionado e com cap duro.

### 4.4 Resumo comparativo

| Método | Complexidade | Quando usar |
|---|---|---|
| Fixed fractional (1–2%) | Baixa | Default, iniciantes, sempre seguro |
| ATR-based | Média | Quando volatilidade varia muito entre ativos/regimes |
| Fractional Kelly | Alta | Só com edge estatístico medido; sempre fracionado + cap |

---

## 5. Risco de portfólio

Position sizing controla o trade; estas regras controlam o **agregado**.

### 5.1 Exposição máxima e portfolio heat

- **Por ativo**: teto de % do equity em um único símbolo (ex.: ≤ 10–20%).
- **Por setor/correlação**: limite a exposição somada de ativos correlacionados (tech, energia, cripto correlacionada ao BTC). Ativos altamente correlacionados contam quase como uma só posição em stress.
- **Portfolio heat** (risco somado em aberto): mantenha o risco total em risco entre todas as posições **abaixo de ~6–10% do equity**. Ex.: conta de $50k → no máximo $3.000–$5.000 em risco somado. ([Take Profit Trader — heat/risk per trade](https://www.quantvps.com/blog/takeprofit-trader-daily-loss-limit))

### 5.2 Max drawdown

**Max Drawdown (MDD)** = maior queda pico-a-vale antes de novo pico: `MDD = (vale - pico) / pico`. Mantê-lo **abaixo de ~20%** é crítico para preservação de capital; recuperação acima disso exige retornos desproporcionais (perder 50% exige +100% para voltar). ([Altrady — Max Drawdown](https://www.altrady.com/blog/risk-management/maximum-drawdown-crypto-trading), [Financial Edge — MDD](https://www.fe.training/free-resources/portfolio-management/maximum-drawdown-mdd/))

### 5.3 Daily loss limit e circuit breakers

- **Daily loss limit**: se a perda do dia atinge X% do equity (ex.: 3%), o bot **para de abrir trades novos** pelo resto do dia (e opcionalmente fecha tudo). Evita "vingança" e cascata.
- **Circuit breaker de portfólio**: gatilhos automáticos que escalam — pausar entradas → reduzir size → liquidar — conforme métricas de stress (drawdown intraday, perdas consecutivas, volatilidade anômala, falha de dados).

```python
class PortfolioRiskGuard:
    def __init__(self, start_equity, daily_loss_pct=0.03,
                 max_dd_pct=0.20, max_per_symbol_pct=0.20,
                 max_heat_pct=0.10):
        self.start_equity = start_equity     # equity no início do dia
        self.peak_equity  = start_equity     # pico para drawdown
        self.daily_loss_pct = daily_loss_pct
        self.max_dd_pct = max_dd_pct
        self.max_per_symbol_pct = max_per_symbol_pct
        self.max_heat_pct = max_heat_pct
        self.trading_halted = False

    def update(self, equity):
        self.peak_equity = max(self.peak_equity, equity)
        daily_pl = (equity - self.start_equity) / self.start_equity
        drawdown = (equity - self.peak_equity) / self.peak_equity

        if daily_pl <= -self.daily_loss_pct:
            self.trading_halted = True
            return "HALT: daily loss limit atingido"
        if drawdown <= -self.max_dd_pct:
            self.trading_halted = True
            return "HALT: max drawdown atingido"
        return None

    def can_open(self, symbol_exposure_pct, current_heat_pct):
        if self.trading_halted:
            return False, "trading halted"
        if symbol_exposure_pct > self.max_per_symbol_pct:
            return False, "exposição por símbolo excedida"
        if current_heat_pct > self.max_heat_pct:
            return False, "portfolio heat excedido"
        return True, "ok"
```

---

## 6. Kill switch

Mecanismo para **parar tudo imediatamente**: parar de abrir risco novo e/ou cancelar ordens e liquidar posições com segurança.

### 6.1 Padrões de implementação da flag

- **Arquivo sentinela**: existência de `KILL` em disco → bot para. Simples, funciona offline, fácil de acionar manualmente (`touch KILL`).
- **Flag em DB**: linha `kill_switch=true`. Bom para múltiplos processos/agentes lendo o mesmo estado.
- **Env var / config**: para parada no deploy/restart.
- **Camadas**: tipicamente combine — flag em DB (estado compartilhado entre agentes) + arquivo (override manual de emergência) + endpoint/healthcheck.

O kill switch deve ser checado **antes de cada nova ordem** e por um **loop de monitoramento** independente.

### 6.2 Cancelar ordens e liquidar com segurança (Alpaca)

A Alpaca expõe:
- `cancel_orders()` — cancela **todas** as ordens abertas.
- `close_all_positions(cancel_orders=True)` — cancela as ordens abertas **e** liquida todas as posições (long e short) via market orders.
- `close_position(symbol)` — fecha uma posição específica.

Ambos os endpoints de "cancel all" e "close all" retornam **HTTP 207 Multi-Status**: o corpo é um array com sub-status por operação — **é preciso checar cada elemento**, não basta o status HTTP. ([Alpaca — Cancel Orders & Liquidations](https://alpaca.markets/blog/position-liquidation-cancel-orders/), [Alpaca Docs — Close All Positions](https://docs.alpaca.markets/reference/deleteallopenpositions-1), [alpaca-py Positions](https://alpaca.markets/sdks/python/api_reference/trading/positions.html))

```python
import os, logging

KILL_FILE = "/var/run/tradingbot/KILL"

def kill_switch_active(db=None) -> bool:
    if os.path.exists(KILL_FILE):
        return True
    if db is not None and db.get_flag("kill_switch"):
        return True
    return False

def emergency_flatten(client, liquidate=True):
    """
    Kill switch: cancela ordens e (opcionalmente) liquida tudo.
    Ordem importa: cancelar PRIMEIRO evita que stops/brackets disparem
    no meio da liquidação. close_all_positions(cancel_orders=True) já faz ambos.
    """
    results = {}
    try:
        if liquidate:
            # cancela ordens abertas E liquida posições
            resp = client.close_all_positions(cancel_orders=True)
            results["close_all"] = resp   # HTTP 207: checar cada item!
        else:
            # apenas para risco novo: cancela ordens, mantém posições
            resp = client.cancel_orders()
            results["cancel_all"] = resp
    except Exception as e:
        logging.critical("KILL SWITCH falhou: %s", e)
        results["error"] = str(e)
    # Verificação pós-condição: reconciliar contra o broker (ver §7.4)
    return results

def trading_loop(client, db, risk_guard):
    while True:
        if kill_switch_active(db):
            logging.critical("KILL SWITCH ATIVO — liquidando e parando")
            emergency_flatten(client, liquidate=True)
            break
        # ... lógica normal de trading só se não-halted ...
```

**Cuidados:**
- **Liquidar via market** sofre slippage; em mercado fechado/illiquido pode preencher feio. Para parada "suave", às vezes é melhor só **cancelar ordens e parar de abrir** (`liquidate=False`) e liquidar com limit orders controladas — decida por política.
- **Sempre reconcilie depois**: confirme via API que não há ordens abertas nem posições residuais. O 207 pode ter falhas parciais.
- O kill switch deve ser **idempotente**: acioná-lo duas vezes não pode quebrar nada.

---

## 7. Execução

### 7.1 Slippage

**Slippage** = diferença entre o preço esperado e o preço de fill, causada pelo mercado se mover entre o envio e a execução. Afeta os dois lados (compra preenche mais caro, venda mais barato). Pior em ativos illíquidos, alta volatilidade, e em market orders grandes que "andam" o order book. ([HeyGoTrade — Slippage](https://www.heygotrade.com/en/blog/slippage-explained-what-is-it/), [Topstep — Slippage](https://help.topstep.com/en/articles/8765442-order-types-fills-and-slippage))

### 7.2 Market vs limit

| | Market | Limit |
|---|---|---|
| Garante | **Execução** (não o preço) | **Preço** (não a execução) |
| Risco | Slippage | Não preencher (preço some) |
| Uso no bot | Saídas urgentes, stops disparados, liquidação | Entradas, scaling-in, alvos — controle de preço |

Default: **limit para entrar, market (ou stop→market) para sair quando precisa de certeza de saída.** Considere *marketable limit* (limit agressivo perto do bid/ask) para equilibrar.

### 7.3 Partial fills

Uma ordem pode preencher em partes (`partially_filled`) — comum em limit e em market grandes que varrem vários níveis do book. O bot **não pode assumir fill total**: trabalhe sempre com `filled_qty` real reportado pela Alpaca, não com `qty` solicitada. Atualize position sizing, stops e PnL com base no preenchido. ([Groww — Partial Fill](https://groww.in/blog/partial-fill-in-trading))

### 7.4 Idempotência (não duplicar ordens)

Toda ordem deve carregar um **`client_order_id`** único e determinístico (gerado pelo bot). Se um retry/reconexão reenviar a mesma intenção, o broker rejeita o duplicado pelo id repetido. Esse é o equivalente da `idempotency-key` do Stripe — gere a partir de algo estável (ex.: hash de `estratégia+símbolo+intenção+timestamp_da_decisão`), não de um `now()` que muda a cada tentativa. (Conceito de `ClOrdID` único por dia vem do FIX protocol: [Charm — FIX ClOrdID](http://www.charm.nl/wordpress/?p=163); a comunidade Alpaca discute o tema em [Idempotency on Order Create](https://forum.alpaca.markets/t/idempotency-on-order-create/15801).)

```python
import hashlib

def make_client_order_id(strategy, symbol, side, decision_ts):
    """
    ID determinístico: o MESMO evento de decisão gera o MESMO id.
    Reenvio por retry => broker rejeita duplicado. NÃO use now() aqui.
    """
    raw = f"{strategy}|{symbol}|{side}|{decision_ts}"
    return "bot-" + hashlib.sha1(raw.encode()).hexdigest()[:24]

# Use em TODA submissão:
# MarketOrderRequest(..., client_order_id=make_client_order_id(...))
```

Regras práticas:
- **Persista a intenção ANTES de enviar** (DB: status `PENDING_SUBMIT`), depois envie, depois marque `SUBMITTED` com o id do broker. Se o processo cair no meio, ao reiniciar você sabe que precisa reconciliar aquela intenção.
- Em retry de timeout de rede, **nunca reenvie cegamente** — primeiro consulte a Alpaca pelo `client_order_id` para ver se a ordem já existe.

### 7.5 Reconciliação de estado (broker vs bot)

O bot mantém uma visão local, mas **o broker é a fonte de verdade**. Reconcilie periodicamente e em todo restart:

1. Buscar do broker: `get_all_positions()`, `get_orders(status="open")`, `get_account()` (equity, buying power).
2. Comparar com o estado local (DB).
3. **Divergências** → resolver a favor do broker: posição que o bot não conhece (registrar e proteger ou liquidar), ordem fantasma local (descartar), stop faltando (recriar).
4. Logar toda divergência — divergência recorrente é sintoma de bug de idempotência ou de partial fill mal tratado.

```python
def reconcile(client, db):
    broker_positions = {p.symbol: p for p in client.get_all_positions()}
    broker_orders    = {o.id: o for o in client.get_orders()}  # abertas
    local            = db.get_state()

    issues = []
    # Posição no broker que o bot não conhece → órfã (proteger ou liquidar)
    for sym, pos in broker_positions.items():
        if sym not in local.positions:
            issues.append(("ORPHAN_POSITION", sym, pos.qty))
    # Posição que o bot acha que tem mas o broker não → estado stale
    for sym in local.positions:
        if sym not in broker_positions:
            issues.append(("STALE_LOCAL_POSITION", sym))
    # Stop esperado mas ausente no broker → recriar proteção
    for sym, exp in local.expected_stops.items():
        if not any(o.symbol == sym and o.type in ("stop", "trailing_stop")
                   for o in broker_orders.values()):
            issues.append(("MISSING_STOP", sym))

    return issues   # broker vence sempre; aja sobre cada issue
```

---

## 8. Boas práticas e erros comuns

**Boas práticas**
- **Paper trading primeiro**, sempre, por tempo suficiente para ver casos de borda (gaps, halts, partial fills).
- **Toda posição nasce com stop** (idealmente bracket atômico). Nunca "vou colocar o stop depois".
- **Cap duro de risco por trade** (1–2%) e **portfolio heat** (≤ 6–10%) impostos em código, não na disciplina humana.
- **Idempotência com `client_order_id`** em toda ordem.
- **Reconciliação no startup e periódica**; broker é a verdade.
- **Kill switch testado** — teste-o em paper, não descubra que está quebrado numa emergência.
- **Log estruturado** de toda decisão, ordem, fill e divergência (auditoria e debugging).
- **Fail-safe**: erro/dado faltando ⇒ não abrir risco novo.
- **Limites de perda diária** e pausa automática após N perdas consecutivas.

**Erros comuns**
- **Over-betting** (Kelly cheio, >2% por trade) → ruína estatística.
- **Averaging down sem stop de invalidação** → *falling knife* destrói a conta.
- **Assumir fill total** ignorando `partially_filled`.
- **Cancel+submit em vez de replace** no trailing manual → janela sem proteção.
- **Ignorar o 207 Multi-Status** em cancel/close all → achar que liquidou quando só liquidou parte.
- **Reenviar ordem em timeout** sem checar duplicidade → ordens duplicadas (já reportado por usuários Alpaca).
- **Não reconciliar** → bot opera sobre estado fantasma após crash.
- **Trailing/stop só client-side sem fallback server-side** → bot cai e a posição fica desprotegida sem proteção mínima no broker.
- **Stop-limit em gap** → preço fura o limite, ordem não preenche, perda ilimitada na prática.
- **Correlação ignorada** → 5 posições "diferentes" que são a mesma aposta macro.

---

## Fontes

- Alpaca — [Trailing Stop Orders](https://alpaca.markets/blog/trailing-stop/)
- Alpaca — [13 Order Types You Should Know](https://alpaca.markets/learn/13-order-types-you-should-know-about)
- Alpaca Docs — [Placing Orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
- Alpaca — [OCO & OTO Orders](https://alpaca.markets/blog/oco-oto/)
- Alpaca — [Bracket Orders](https://alpaca.markets/blog/bracket-orders/)
- Trade Mamba — [Bracket Orders gotcha](https://medium.com/@trademamba/bracket-orders-with-alpaca-markets-and-a-key-gotcha-6560d47ad6f4)
- Alpaca — [Cancel Orders & Position Liquidations](https://alpaca.markets/blog/position-liquidation-cancel-orders/)
- Alpaca Docs — [Close All Positions](https://docs.alpaca.markets/reference/deleteallopenpositions-1)
- alpaca-py — [Positions API](https://alpaca.markets/sdks/python/api_reference/trading/positions.html)
- Medium (I. Veliu) — [Position Sizing Frameworks](https://medium.com/@ildiveliu/risk-before-returns-position-sizing-frameworks-fixed-fractional-atr-based-kelly-lite-4513f770a82a)
- ActivTrades — [Volatility-Based Position Sizing](https://www.activtrades.com/en/news/how-to-use-volatility-based-position-sizing-when-trading-with-cfds)
- QuantStrategy — [Applying the Kelly Criterion](https://quantstrategy.io/blog/applying-the-kelly-criterion-to-trading-maximizing-growth/)
- JournalPlus — [Kelly Criterion Guide](https://journalplus.co/learn/guides/kelly-criterion-guide/)
- FMZQuant — [Grid Dollar-Cost Averaging Strategy](https://medium.com/@FMZQuant/grid-dollar-cost-averaging-strategy-dbc5bbbc1574)
- TradeLink — [Grid Trading vs DCA](https://tradelink.pro/blog/grid-trading-vs-dca/)
- Gainium — [DCA vs Grid vs Combo bots](https://gainium.io/blog/dca-vs-grid-vs-combo-bots-choosing-the-right-strategy)
- Altrady — [Maximum Drawdown](https://www.altrady.com/blog/risk-management/maximum-drawdown-crypto-trading)
- Financial Edge — [Maximum Drawdown (MDD)](https://www.fe.training/free-resources/portfolio-management/maximum-drawdown-mdd/)
- QuantVPS — [Daily Loss Limit / Portfolio Heat](https://www.quantvps.com/blog/takeprofit-trader-daily-loss-limit)
- HeyGoTrade — [Slippage Explained](https://www.heygotrade.com/en/blog/slippage-explained-what-is-it/)
- Topstep — [Order Types, Fills, and Slippage](https://help.topstep.com/en/articles/8765442-order-types-fills-and-slippage)
- Groww — [Partial Fill in Trading](https://groww.in/blog/partial-fill-in-trading)
- Charm — [FIX and the Client Order ID](http://www.charm.nl/wordpress/?p=163)
- Alpaca Forum — [Idempotency on Order Create](https://forum.alpaca.markets/t/idempotency-on-order-create/15801)

---

## Como virar skill

Para transformar este documento numa **skill do Claude Code** (`risco-execucao`):

1. **Local**: criar `~/.claude/skills/risco-execucao/SKILL.md` (ou `.claude/skills/` no repo do projeto para versionar com o bot).
2. **Frontmatter**: manter `name: risco-execucao` e escrever um `description` que dispare nos contextos certos — ex.: *"Use ao implementar ou revisar gestão de risco, position sizing, stops, kill switch, bracket orders ou execução no bot de trading Alpaca."* O `description` é o que o Claude usa para decidir carregar a skill, então cite os gatilhos (trailing stop, Kelly, kill switch, slippage, reconciliação).
3. **Corpo enxuto + arquivos de apoio**: mover os blocos de código longos (ManualTrailingStop, PortfolioRiskGuard, kill switch, reconcile) para `references/` ou `scripts/` dentro da pasta da skill e referenciá-los a partir do SKILL.md, mantendo o corpo principal focado em *quando aplicar cada controle*. Skills carregam progressivamente — o corpo deve ser curto e apontar para os detalhes.
4. **Checklists acionáveis**: converter §8 (boas práticas/erros) num checklist que o agente percorre ao revisar código de ordem/risco (ex.: "toda entrada tem stop atômico? `client_order_id` presente? 207 tratado? reconciliação no startup?").
5. **Validação**: incluir um script de teste em paper que exercite kill switch, partial fill e reconciliação — a skill pode rodá-lo como gate antes de qualquer mudança em produção.
6. **Versionar com o bot** e revisar quando a API da Alpaca mudar (checar mudanças em order types e nos endpoints 207).
