---
name: alpaca-api
description: Consulte ao integrar, autenticar ou negociar (paper/live) com a Alpaca via SDK alpaca-py — ordens, market data (IEX/SIP), opções, rate limits e clock/calendar.
metadata:
  type: reference
---

# Alpaca API + SDK `alpaca-py` — Documento de Referência

> Atualizado em 2026-06. Valida na documentação oficial: <https://docs.alpaca.markets/> e <https://alpaca.markets/sdks/python/>.
> Versão mais recente do `alpaca-py` no momento da escrita: **0.43.x** (abr/2026). Python 3.8+.

Este documento é a fonte de verdade para o nosso bot de trading multi-agente em Python sobre a Alpaca. Foco em **paper trading primeiro**, depois live. Todos os endpoints e nomes de classe foram verificados na documentação oficial — não inventar.

---

## 1. Visão geral da Alpaca

A Alpaca expõe três famílias de APIs distintas:

| API | Para quê | Base URL |
| --- | --- | --- |
| **Trading API** | Negociar ações, opções e cripto para uma conta individual/empresarial (retail, algo, prop). Submeter ordens, consultar posições, conta, clock/calendar. | `https://api.alpaca.markets` (live) / `https://paper-api.alpaca.markets` (paper) |
| **Market Data API** | Dados de mercado em tempo real e histórico (6+ anos) para ações, cripto, opções e notícias. | `https://data.alpaca.markets` (mesma URL para paper e live) |
| **Broker API** | Construir apps de corretagem para **usuários finais** (challenger banks, trading apps). Cria e gerencia contas de terceiros. Não é o que usamos num bot pessoal. | `https://broker-api.alpaca.markets` (prod) / `https://broker-api.sandbox.alpaca.markets` (sandbox) |

### Conta paper vs. live

- **Paper trading** (`paper-api.alpaca.markets`): ambiente de simulação com dinheiro fictício, execução simulada, mesmo modelo de dados e (quase) os mesmos endpoints da live. **As chaves de API do paper são DIFERENTES das chaves da live** — geradas separadamente no dashboard. Comece sempre aqui.
- **Live trading** (`api.alpaca.markets`): dinheiro real, ordens reais roteadas ao mercado.
- A **Market Data API** usa a mesma base URL (`data.alpaca.markets`) independentemente de a conta de trading ser paper ou live. O que muda o acesso aos dados é o **plano de assinatura de dados** (free IEX vs. pago SIP), não o ambiente paper/live.
- No `alpaca-py`, a diferença é controlada pelo parâmetro `paper=True|False` no `TradingClient` — você não monta a URL manualmente.

**Fontes:**
- <https://docs.alpaca.markets/us/docs/getting-started>
- <https://docs.alpaca.markets/us/docs/authentication>
- <https://docs.alpaca.markets/us/docs/paper-trading>
- <https://alpaca.markets/>

---

## 2. Autenticação

A autenticação padrão (legacy/key-based) é feita por **dois headers HTTP**:

```
APCA-API-KEY-ID:     <sua API Key ID>
APCA-API-SECRET-KEY: <sua Secret Key>
```

- A **API Key ID** é pública-ish (identifica a chave); a **Secret Key** é mostrada **uma única vez** na criação — guarde com segurança (ex.: variáveis de ambiente / secret manager, nunca em commit).
- Chaves de **paper** e **live** são distintas e não são intercambiáveis entre os ambientes.
- A Alpaca também oferece um fluxo **OAuth 2.0 Client Credentials** (token endpoint `https://authx.alpaca.markets/v1/oauth2/token`, Bearer token válido por ~15 min, com `client_secret_post` ou `private_key_jwt`). **Importante:** no momento da escrita, o Client Credentials flow **ainda não está disponível para a Trading API** — então, para o nosso bot, usamos as duas chaves key/secret.

### Como o `alpaca-py` lida com isso

O SDK injeta os headers automaticamente a partir das credenciais passadas no construtor de cada client. Você nunca monta os headers à mão:

```python
from alpaca.trading.client import TradingClient

trading_client = TradingClient(
    api_key="PKxxxxxxxxxxxxxxxxxx",
    secret_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    paper=True,  # roteia para paper-api.alpaca.markets
)
```

### `alpaca-py` é o SDK recomendado? (vs. `alpaca-trade-api`)

**Sim.** `alpaca-py` é o **SDK oficial e atual** da Alpaca para Python e é o recomendado. O antigo `alpaca-trade-api` (`alpaca-trade-api-python`) é **legacy** — ainda recebe manutenção mínima, mas não recebe features novas e a Alpaca recomenda migrar. Diferenças centrais:

- `alpaca-py` adota uma abordagem **OOP**: você cria objetos de request tipados (`MarketOrderRequest`, `StockBarsRequest`, `GetCalendarRequest`, etc.) e enums (`OrderSide`, `TimeInForce`, `OrderType`, `OrderClass`) em vez de passar dicts soltos.
- Clients separados por produto (trading, dados de stock, dados de option, streams) em vez de um único objeto `REST`.
- **Para qualquer feature nova (especialmente opções e MLeg) use `alpaca-py`.** Não usar `alpaca-trade-api` em código novo.

**Fontes:**
- <https://docs.alpaca.markets/us/docs/authentication>
- <https://github.com/alpacahq/alpaca-py>
- <https://docs.alpaca.markets/us/docs/sdks-and-tools>
- <https://pypi.org/project/alpaca-py/>

---

## 3. Instalação e setup do `alpaca-py`

```bash
pip install alpaca-py
# upgrade:
pip install alpaca-py --upgrade
```

Requer **Python 3.8+**.

### Classes principais (clients)

Verificadas no SDK oficial:

| Client | Módulo | Uso |
| --- | --- | --- |
| `TradingClient` | `alpaca.trading.client` | Ordens, conta, posições, clock, calendar, assets, watchlists |
| `StockHistoricalDataClient` | `alpaca.data.historical` | Barras, quotes, trades históricos de ações |
| `CryptoHistoricalDataClient` | `alpaca.data.historical` | Dados históricos de cripto (sem chave obrigatória para cripto) |
| `OptionHistoricalDataClient` | `alpaca.data.historical` | Dados históricos/snapshots de opções |
| `NewsClient` | `alpaca.data.historical` | Notícias |
| `StockDataStream` | `alpaca.data.live` | Stream WebSocket de ações (quotes, trades, bars) |
| `CryptoDataStream` | `alpaca.data.live` | Stream WebSocket de cripto |
| `OptionDataStream` | `alpaca.data.live` | Stream WebSocket de opções |
| `NewsDataStream` | `alpaca.data.live` | Stream de notícias |
| `BrokerClient` | `alpaca.broker.client` | Broker API (apps multi-conta) — não usado no bot pessoal |

Objetos de request e enums vivem em `alpaca.trading.requests`, `alpaca.trading.enums`, `alpaca.data.requests`, `alpaca.data.timeframe`.

**Fontes:**
- <https://github.com/alpacahq/alpaca-py>
- <https://alpaca.markets/sdks/python/>
- <https://alpaca.markets/sdks/python/api_reference/data/option/historical.html>

---

## 4. Ordens

### Tipos de ordem suportados

- `market` — executa ao preço de mercado disponível (rápido, sujeito a slippage).
- `limit` — executa no preço informado **ou melhor**.
- `stop` — vira **market** ao atingir o `stop_price` (não garante preço de execução).
- `stop_limit` — vira **limit** ao atingir o `stop_price` (executa só no `limit_price` ou melhor).
- `trailing_stop` — stop que acompanha o preço favorável via `trail_price` (offset em $) **ou** `trail_percent` (offset %). TIF apenas `day` ou `gtc`; só dispara em horário regular de mercado.

> Restrição de sub-penny: para preços ≥ $1.00, no máximo 2 casas decimais; para < $1.00, até 4 casas.

### Classes de ordem (`order_class`)

- `simple` — ordem individual (default).
- `bracket` — entrada + dois exits condicionais (take-profit limit + stop-loss). Só **um** exit executa. **Não suporta extended hours**; TIF deve ser `day` ou `gtc`.
- `oco` (One-Cancels-Other) — dois exits do **mesmo lado**; take-profit é limit, stop-loss é stop ou stop-limit.
- `oto` (One-Triggers-Other) — entrada + **um** exit (take-profit **ou** stop-loss, não ambos).
- `mleg` — multi-leg (estratégias de opções, ver seção 7).

> Trailing stops **não** são suportados como pernas de bracket/OCO (planejado para o futuro).

### Time-in-Force (TIF)

| TIF | Descrição |
| --- | --- |
| `day` | Válida só no pregão; cancela no fechamento se não executar. Pode operar em extended hours se a ordem for marcada elegível. |
| `gtc` | Good-till-canceled. Auto-cancela após 90 dias. |
| `ioc` | Immediate-or-Cancel: executa imediatamente a parte possível, cancela o resto. |
| `fok` | Fill-or-Kill: executa a quantidade total imediatamente ou cancela tudo. |
| `opg` | On-Open: só no leilão de abertura. |
| `cls` | On-Close: só no leilão de fechamento. |

> **Cripto** aceita só `gtc` e `ioc`. **Opções** aceitam só `day` e `gtc`, sem notional, sem extended hours, quantidade inteira.

### Exemplos de código (alpaca-py)

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

trading_client = TradingClient("API_KEY", "SECRET_KEY", paper=True)

# --- Market ---
mkt = MarketOrderRequest(
    symbol="SPY", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
)
trading_client.submit_order(order_data=mkt)

# Market notional (em dólares) ou fracionário
mkt_notional = MarketOrderRequest(
    symbol="SPY", notional=500, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
)

# --- Limit ---
lim = LimitOrderRequest(
    symbol="AAPL", qty=10, limit_price=190.00,
    side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
)
trading_client.submit_order(order_data=lim)

# --- Stop (vira market no stop_price) ---
stp = StopOrderRequest(
    symbol="AAPL", qty=10, stop_price=180.00,
    side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
)
trading_client.submit_order(order_data=stp)

# --- Stop-Limit ---
stp_lim = StopLimitOrderRequest(
    symbol="AAPL", qty=10, stop_price=180.00, limit_price=179.50,
    side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
)
trading_client.submit_order(order_data=stp_lim)

# --- Trailing Stop (5% trailing) ---
trail = TrailingStopOrderRequest(
    symbol="SPY", qty=10, trail_percent=5.0,
    side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
)
trading_client.submit_order(order_data=trail)
# alternativa por offset em dólares: trail_price=2.50

# --- Bracket Order (entrada market + TP + SL) ---
bracket = MarketOrderRequest(
    symbol="SPY", qty=5, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest(limit_price=420.00),
    stop_loss=StopLossRequest(stop_price=390.00),  # opcional: limit_price -> stop-limit
)
trading_client.submit_order(order_data=bracket)

# --- OCO (One-Cancels-Other): só os dois exits, sobre posição existente ---
oco = LimitOrderRequest(
    symbol="SPY", qty=5, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
    order_class=OrderClass.OCO,
    take_profit=TakeProfitRequest(limit_price=430.00),
    stop_loss=StopLossRequest(stop_price=395.00),
)
trading_client.submit_order(order_data=oco)

# --- OTO (One-Triggers-Other): entrada + UM exit ---
oto = MarketOrderRequest(
    symbol="SPY", qty=5, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    order_class=OrderClass.OTO,
    take_profit=TakeProfitRequest(limit_price=420.00),
)
trading_client.submit_order(order_data=oto)
```

> **Gotcha de bracket**: `time_in_force` deve ser `DAY` ou `GTC` e `extended_hours` precisa ser `False`. Ordens notional/fracionárias não combinam com algumas classes (bracket exige `qty` inteira em vários casos). Ordens notional **não podem** ser substituídas (replace) — cancele e reenvie.

### Cancelar / substituir

```python
trading_client.cancel_order_by_id(order_id)
trading_client.cancel_orders()  # cancela todas as ordens abertas
# substituir (replace) campos de uma ordem existente:
from alpaca.trading.requests import ReplaceOrderRequest
trading_client.replace_order_by_id(order_id, ReplaceOrderRequest(qty=3, limit_price=191.0))
```

**Fontes:**
- <https://docs.alpaca.markets/us/docs/orders-at-alpaca>
- <https://alpaca.markets/learn/13-order-types-you-should-know-about>
- <https://alpaca.markets/sdks/python/api_reference/trading/requests.html>
- <https://alpaca.markets/sdks/python/trading.html>
- <https://forum.alpaca.markets/t/bracket-order-code-example-with-alpaca-py-library/12110>

---

## 5. Posições e conta

```python
# Conta: buying power, cash, equity, status
account = trading_client.get_account()
print(account.buying_power, account.cash, account.equity, account.portfolio_value)
print(account.pattern_day_trader, account.trading_blocked, account.account_blocked)

# Posições
positions = trading_client.get_all_positions()
for p in positions:
    print(p.symbol, p.qty, p.avg_entry_price, p.market_value, p.unrealized_pl)

pos = trading_client.get_open_position("SPY")  # uma posição específica

# Fechar posições
trading_client.close_position("SPY")                 # fecha SPY
# fechar parte da posição:
from alpaca.trading.requests import ClosePositionRequest
trading_client.close_position("SPY", ClosePositionRequest(qty="2"))
trading_client.close_all_positions(cancel_orders=True)  # fecha tudo + cancela ordens abertas
```

Campos úteis de `account`: `buying_power`, `non_marginable_buying_power`, `cash`, `equity`, `last_equity`, `portfolio_value`, `daytrading_buying_power`, `regt_buying_power`, `pattern_day_trader`, `trading_blocked`, `transfers_blocked`, `account_blocked`, `options_approved_level`, `options_trading_level`.

**Fontes:**
- <https://alpaca.markets/sdks/python/trading.html>
- <https://alpaca.markets/sdks/python/api_reference/trading_api.html>

---

## 6. Market Data

### Histórico — barras, quotes, trades

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from datetime import datetime

data_client = StockHistoricalDataClient("API_KEY", "SECRET_KEY")

# Barras diárias
bars_req = StockBarsRequest(
    symbol_or_symbols=["AAPL", "MSFT"],
    timeframe=TimeFrame.Day,
    start=datetime(2025, 1, 1),
    end=datetime(2025, 6, 1),
    feed=DataFeed.IEX,   # free; use DataFeed.SIP se tiver assinatura
)
bars = data_client.get_stock_bars(bars_req)
print(bars.df)  # DataFrame multi-index (symbol, timestamp)

# Timeframe customizado (ex.: 15 minutos)
tf_15m = TimeFrame(15, TimeFrameUnit.Minute)

# Latest quote
lq = data_client.get_stock_latest_quote(
    StockLatestQuoteRequest(symbol_or_symbols="AAPL", feed=DataFeed.IEX)
)
print(lq["AAPL"].ask_price, lq["AAPL"].bid_price)

# Latest trade
lt = data_client.get_stock_latest_trade(
    StockLatestTradeRequest(symbol_or_symbols="AAPL", feed=DataFeed.IEX)
)
print(lt["AAPL"].price, lt["AAPL"].size)
```

`TimeFrame` aceita os helpers `TimeFrame.Minute`, `TimeFrame.Hour`, `TimeFrame.Day`, `TimeFrame.Week`, `TimeFrame.Month` e a forma `TimeFrame(amount, TimeFrameUnit.X)`.

### Tempo real — WebSocket

```python
from alpaca.data.live import StockDataStream

stream = StockDataStream("API_KEY", "SECRET_KEY")  # feed default conforme assinatura

async def on_bar(bar):
    print(bar.symbol, bar.close)

async def on_quote(quote):
    print(quote.symbol, quote.bid_price, quote.ask_price)

stream.subscribe_bars(on_bar, "AAPL", "MSFT")
stream.subscribe_quotes(on_quote, "AAPL")
# stream.subscribe_trades(handler, "AAPL")

stream.run()  # bloqueante; gerencia reconexão automaticamente
```

### Free (IEX) vs. pago (SIP) — limitações

- **Plano free → feed IEX**: dados de uma única bolsa (IEX). Cobre fração pequena do volume total (ex.: ~12k trades IEX num dia em que houve ~535k no mercado todo). Bom para protótipo/paper, **enganoso para preços/volume reais**.
- **Plano pago (Algo Trader Plus / Unlimited) → feed SIP**: feeds consolidados CTA (NYSE) + UTP (Nasdaq) = ~100% do volume.
- **Restrições do free**:
  - Endpoints "latest"/snapshot exigem assinatura para usar feed SIP.
  - Em queries **históricas** SIP sem assinatura, o `end` deve ter pelo menos **15 minutos de atraso** (regra dos 15 min).
  - Para WebSocket free há **limite de 1 conexão simultânea** por conta no feed IEX.
- No `alpaca-py` o feed é escolhido via `feed=DataFeed.IEX` / `DataFeed.SIP` nos requests, ou no construtor do stream.

**Fontes:**
- <https://docs.alpaca.markets/us/docs/market-data-faq>
- <https://alpaca.markets/data>
- <https://alpaca.markets/support/data-provider-alpaca>
- <https://alpaca.markets/sdks/python/market_data.html>

---

## 7. Opções na Alpaca (status 2025/2026)

A Alpaca oferece **trading de opções US** com até **3 níveis** de aprovação. Status atual:

### Níveis de aprovação

- **Level 1** — estratégias cobertas/conservadoras: vender covered calls e cash-secured puts (exige ações/colateral suficientes).
- **Level 2** — tudo do Level 1 + comprar calls e puts (long options, direcional).
- **Level 3** — tudo do Level 1-2 + **spreads** (debit/credit spreads de call e put), via ordens multi-leg.

O modelo de conta expõe `options_approved_level` e `options_trading_level`; é possível fazer downgrade para um nível menor via Account Configuration.

### Regras de ordem de opções

- Quantidade **inteira** (sem notional / sem fracionário).
- TIF apenas `day` ou `gtc`.
- **Sem extended hours**.
- Usa o **mesmo endpoint `/v2/orders`** das ações/cripto.
- **Paper trading**: opções vêm **habilitadas por padrão** no ambiente paper. Atividades não-trade (exercício, atribuição, expiração) sincronizam no dia seguinte; saldos/posições atualizam na hora.

### Símbolo OCC

Formato padrão OCC: `UNDERLYING + YYMMDD + C/P + STRIKE(8 dígitos)`.
Exemplo: `AAPL250117C00190000` = AAPL, exp 2025-01-17, Call, strike $190.00.

### Negociar opções via alpaca-py

Ordem de **uma perna** (single-leg) — basta usar o símbolo OCC com os request objects normais:

```python
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Comprar 1 contrato de call (Level 2+)
opt = LimitOrderRequest(
    symbol="AAPL250117C00190000",
    qty=1,
    limit_price=2.50,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
)
trading_client.submit_order(order_data=opt)
```

Ordem **multi-leg (MLeg, Level 3)** — `order_class="mleg"` com `legs`. O SDK alpaca-py expõe `OptionLegRequest` + `LimitOrderRequest(order_class=OrderClass.MLEG, legs=[...])`. Estrutura do payload (referência REST oficial):

```python
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, PositionIntent

# Bull call spread: compra strike 190, vende strike 210
mleg = LimitOrderRequest(
    qty=1,
    limit_price=1.00,           # débito líquido do spread
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.MLEG,
    legs=[
        OptionLegRequest(
            symbol="AAPL250117C00190000",
            side=OrderSide.BUY,
            ratio_qty=1,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        OptionLegRequest(
            symbol="AAPL250117C00210000",
            side=OrderSide.SELL,
            ratio_qty=1,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
    ],
)
trading_client.submit_order(order_data=mleg)
```

> `ratio_qty` deve estar na forma simplificada (GCD dos legs = 1). Confirme os nomes exatos `OptionLegRequest`/`PositionIntent` na versão instalada do alpaca-py — a API REST subjacente usa `order_class: "mleg"`, `legs[].ratio_qty`, `legs[].position_intent` (`buy_to_open`, `sell_to_open`, `buy_to_close`, `sell_to_close`).

### Market data de opções

```python
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, OptionChainRequest

opt_client = OptionHistoricalDataClient("API_KEY", "SECRET_KEY")

# Quote mais recente de um contrato
q = opt_client.get_option_latest_quote(
    OptionLatestQuoteRequest(symbol_or_symbols="AAPL250117C00190000")
)

# Option chain (snapshots) por underlying
chain = opt_client.get_option_chain(OptionChainRequest(underlying_symbol="AAPL"))
```

**Fontes:**
- <https://docs.alpaca.markets/us/docs/options-trading>
- <https://alpaca.markets/options>
- <https://alpaca.markets/learn/how-to-trade-options-with-alpaca>
- <https://alpaca.markets/sdks/python/api_reference/data/option/historical.html>

---

## 8. Rate limits, paginação, erros, horários de mercado

### Rate limits

- Padrão: **200 requests/minuto por API key** (vale tanto para Trading API quanto Market Data API no tier free/básico).
- Excedeu → HTTP **429 Too Many Requests**.
- Pode ser elevado (até ~1.000/min) contatando o suporte — geralmente reclassifica a conta como non-retail.
- O `alpaca-py` tem um wrapper de rate-limit/retry interno, mas em produção **implemente o seu próprio backoff** (ver seção 9).

### Paginação

- Endpoints de market data histórico retornam um `next_page_token` quando o resultado é truncado.
- No `alpaca-py`, os métodos de dados históricos (ex.: `get_stock_bars`) **paginam automaticamente** sob o capô e devolvem o conjunto completo — você normalmente não lida com tokens manualmente. Para grandes janelas, isso pode gerar muitas chamadas; controle o número de símbolos e a janela temporal.
- Endpoints de listagem da Trading API (ex.: `get_orders`) usam parâmetros `limit`, `after`, `until`, `direction` para paginar.

### Clock e Calendar (horários de mercado)

```python
from alpaca.trading.requests import GetCalendarRequest
from datetime import date

clock = trading_client.get_clock()
print(clock.timestamp, clock.is_open, clock.next_open, clock.next_close)

if clock.is_open:
    ...  # operar

# Calendar: dias de pregão (1970–~2029), com horários de abertura/fechamento
cal = trading_client.get_calendar(
    GetCalendarRequest(start=date(2026, 6, 1), end=date(2026, 6, 30))
)
for day in cal:
    print(day.date, day.open, day.close)
```

### Extended hours

- Só **limit orders** com TIF `day` ou `gtc`, marcadas com `extended_hours=True`. Qualquer outro tipo/TIF é **rejeitado**.
- Janelas: overnight 20:00–04:00 ET (dom–sex), pré-market 04:00–09:30 ET, after-hours 16:00–20:00 ET (seg–sex).
- `bracket`/opções **não** suportam extended hours.

```python
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

ext = LimitOrderRequest(
    symbol="AAPL", qty=10, limit_price=190.0,
    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    extended_hours=True,
)
trading_client.submit_order(order_data=ext)
```

### Tratamento de erros

- O `alpaca-py` lança `alpaca.common.exceptions.APIError` em respostas de erro da API. O objeto traz `code` e mensagem; inspecione `e.status_code` / o JSON do corpo.
- Status comuns: `401/403` (auth/permite), `422` (request inválido — ex.: TIF incompatível, sub-penny, buying power), `429` (rate limit), `5xx` (lado Alpaca).

```python
from alpaca.common.exceptions import APIError

try:
    trading_client.submit_order(order_data=mkt)
except APIError as e:
    print("Alpaca rejeitou:", e)  # ex.: insufficient buying power, market closed, etc.
```

**Fontes:**
- <https://alpaca.markets/support/increase-api-rate-limit>
- <https://docs.alpaca.markets/us/docs/orders-at-alpaca>
- <https://alpaca.markets/sdks/python/api_reference/trading/calendar.html>
- <https://alpaca.markets/sdks/python/api_reference/trading/clock.html>

---

## 9. Pitfalls / erros comuns e boas práticas de produção

- **Chaves paper ≠ live.** Trocar de ambiente exige trocar o par de chaves **e** `paper=True/False`. Não reutilize chaves entre ambientes.
- **Feed IEX engana.** No plano free, preços/volume do IEX são parciais. Não calibre estratégia em dados IEX achando que é o mercado todo; para backtest/produção sério, assine SIP.
- **Regra dos 15 minutos (SIP free).** Queries históricas SIP sem assinatura precisam de `end` ≥ 15 min no passado, senão erro/dados vazios.
- **Sub-penny / tick size.** Preços de limit/stop fora da granularidade (2 casas ≥$1, 4 casas <$1) são rejeitados com 422.
- **Bracket gotcha.** TIF deve ser `day`/`gtc`, `extended_hours=False`, e cuidado com `qty` inteira; ordens notional não viram bracket.
- **Wash trade / posições conflitantes.** A Alpaca rejeita ordens que criariam posição short líquida em horário estendido com fracionário, e bloqueia algumas combinações de ordens abertas conflitantes.
- **PDT (Pattern Day Trader).** Contas < $25k são limitadas a 3 day trades em 5 dias úteis; verifique `account.pattern_day_trader` e `daytrading_buying_power` antes de operar intraday.
- **Idempotência.** Use `client_order_id` (UUID seu) em cada `submit_order` para detectar/evitar duplicatas em retries.
- **Backoff em 429/5xx.** Implemente retry com exponential backoff + jitter; respeite o limite de 200 rpm; agregue chamadas de dados (multi-símbolo num único request) em vez de loops.
- **Reconexão de WebSocket.** O `StockDataStream.run()` reconecta sozinho, mas em produção monitore o stream, trate gaps e reconcilie via REST (`get_orders`, posições) após reconexões — não confie só no stream para estado de ordens.
- **Estado de ordem via stream `TradingStream`.** Para acompanhar fills em tempo real use `alpaca.trading.stream.TradingStream` (trade updates), não polling agressivo de `get_orders`.
- **Sempre cheque o clock.** Antes de enviar ordens, valide `get_clock().is_open` (e calendar para feriados) para evitar rejeições por mercado fechado.
- **Segredos fora do código.** Carregue chaves de env vars / secret manager; nunca commit. Rotacione se vazar.
- **Use `alpaca-py`, não `alpaca-trade-api`** em código novo — especialmente para opções e MLeg.

**Fontes:**
- <https://docs.alpaca.markets/us/docs/orders-at-alpaca>
- <https://docs.alpaca.markets/us/docs/market-data-faq>
- <https://medium.com/@trademamba/bracket-orders-with-alpaca-markets-and-a-key-gotcha-6560d47ad6f4>

---

## Fontes (consolidado)

- Alpaca Docs — Getting Started: <https://docs.alpaca.markets/us/docs/getting-started>
- Alpaca Docs — Authentication: <https://docs.alpaca.markets/us/docs/authentication>
- Alpaca Docs — Orders at Alpaca: <https://docs.alpaca.markets/us/docs/orders-at-alpaca>
- Alpaca Docs — Options Trading: <https://docs.alpaca.markets/us/docs/options-trading>
- Alpaca Docs — Market Data FAQ: <https://docs.alpaca.markets/us/docs/market-data-faq>
- Alpaca Docs — SDKs and Tools: <https://docs.alpaca.markets/us/docs/sdks-and-tools>
- Alpaca Learn — Order Types & TIF: <https://alpaca.markets/learn/13-order-types-you-should-know-about>
- Alpaca Learn — How to Trade Options: <https://alpaca.markets/learn/how-to-trade-options-with-alpaca>
- alpaca-py (GitHub): <https://github.com/alpacahq/alpaca-py>
- alpaca-py (PyPI): <https://pypi.org/project/alpaca-py/>
- alpaca-py Docs — Trading: <https://alpaca.markets/sdks/python/trading.html>
- alpaca-py Docs — Market Data: <https://alpaca.markets/sdks/python/market_data.html>
- alpaca-py Reference — Requests: <https://alpaca.markets/sdks/python/api_reference/trading/requests.html>
- alpaca-py Reference — Clock: <https://alpaca.markets/sdks/python/api_reference/trading/clock.html>
- alpaca-py Reference — Calendar: <https://alpaca.markets/sdks/python/api_reference/trading/calendar.html>
- alpaca-py Reference — Option Historical: <https://alpaca.markets/sdks/python/api_reference/data/option/historical.html>
- Alpaca Data: <https://alpaca.markets/data>

---

## Como virar skill

Extrair para uma skill `alpaca-trading`: (1) um cliente fino que encapsula `TradingClient`/data clients com chaves vindas de env e `paper` configurável, mais helpers idempotentes de `submit_order` (market/limit/stop/bracket) com `client_order_id` e backoff em 429/5xx; (2) um "cheat sheet" de enums/TIF/order_class e das regras (sub-penny, bracket gotchas, regras de opções/MLeg, IEX vs SIP) como guardrails que o agente consulta antes de montar uma ordem; (3) um pré-flight de `get_clock`/`get_calendar` + checagem de buying power/PDT que roda antes de qualquer ordem.
