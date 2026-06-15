---
name: wheel-opcoes
description: Referência completa da Wheel Strategy (CSP + covered call) e da mecânica de opções para automação com Alpaca + alpaca-py.
metadata:
  type: reference
---

# Wheel Strategy e Mecânica de Opções para Automação

> Documento de referência para o bot de trading multi-agente (Alpaca, Python). Termos
> técnicos mantidos em inglês. Todas as afirmações sobre a API da Alpaca foram verificadas
> na documentação oficial (links no fim de cada seção e em "Fontes"). Informação atual
> 2025/2026.

---

## 1. Fundamentos de opções

Uma **opção** é um contrato que dá ao **comprador (holder/buyer)** o *direito* — não a
obrigação — de comprar ou vender 100 ações (1 contrato = **100 shares**, o *multiplier*
padrão de equity options nos EUA) de um ativo subjacente (*underlying*) a um preço fixo,
até uma data. O **vendedor (seller/writer)** assume a *obrigação* correspondente e recebe
o **prêmio (premium)** como pagamento por assumir esse risco.

### Calls e puts

- **Call**: direito de **comprar** o underlying ao **strike**. O comprador de call lucra
  se o preço sobe acima de `strike + premium`.
- **Put**: direito de **vender** o underlying ao **strike**. O comprador de put lucra se o
  preço cai abaixo de `strike − premium`.

### Comprador vs. vendedor (writer)

| | Direito/Obrigação | Paga ou recebe prêmio | P&L máximo | Risco máximo |
|---|---|---|---|---|
| **Long call** (comprador) | direito de comprar | paga | ilimitado | prêmio pago |
| **Short call** (writer) | obrigação de vender | recebe | prêmio recebido | ilimitado (se *naked*) |
| **Long put** (comprador) | direito de vender | paga | strike − prêmio | prêmio pago |
| **Short put** (writer) | obrigação de comprar | recebe | prêmio recebido | strike − prêmio (até 0) |

Na Wheel **somos sempre o writer** (vendedor): vendemos puts e calls para coletar prêmio.
Isso inverte a lógica intuitiva — queremos que a opção **expire sem valor (worthless)**
para ficar com o prêmio inteiro.

### Termos centrais

- **Strike**: preço de exercício fixado no contrato.
- **Expiration (vencimento)**: data em que o contrato deixa de existir. Equity options nos
  EUA geralmente expiram nas sextas-feiras (weeklies e monthlies). Liquidação física
  (recebe/entrega ações) é o padrão em equity options.
- **Premium (prêmio)**: preço do contrato, cotado por ação. Um prêmio de `$1.50` custa
  `$150` (×100). É o que o writer recebe.
- **Exercise (exercício)**: o **comprador** aciona seu direito. **Assignment (atribuição)**:
  é o lado do **vendedor** — ele é selecionado (pela OCC, aleatoriamente entre writers) para
  cumprir a obrigação. Quando uma put que vendemos é *assigned*, somos obrigados a **comprar**
  100 ações ao strike; quando uma call coberta é *assigned*, somos obrigados a **vender**.

### Americana vs. europeia

- **Americana**: pode ser exercida **a qualquer momento** até o vencimento. Quase todas as
  **equity/ETF options** dos EUA (AAPL, SPY, etc.) são americanas → **assignment antecipada
  (early assignment) é possível**, especialmente perto de **ex-dividend dates** (calls ITM) ou
  quando o valor extrínseco vira ~0.
- **Europeia**: só pode ser exercida **no vencimento**. Típica de **index options**
  (SPX, NDX), que também são **cash-settled** (liquidação financeira, sem entrega de ações).

> **Para a Wheel automatizada**: opere ativos com opções **americanas e physically-settled**
> (ações e ETFs líquidos). Index options europeias não dão "ações para rodar a roda". E o
> código precisa tratar **early assignment** como evento possível a qualquer momento.
> A Alpaca expõe o estilo via `ExerciseStyle.AMERICAN` / `ExerciseStyle.EUROPEAN`.

### Moneyness: ITM / ATM / OTM

Relação entre **strike** e **preço atual (spot)** do underlying:

| Moneyness | Call | Put |
|---|---|---|
| **ITM** (in-the-money) | spot > strike | spot < strike |
| **ATM** (at-the-money) | spot ≈ strike | spot ≈ strike |
| **OTM** (out-of-the-money) | spot < strike | spot > strike |

Na Wheel vendemos **OTM**: put OTM (strike abaixo do spot) e call OTM (strike acima do custo).

### Valor intrínseco vs. extrínseco

`premium = valor intrínseco + valor extrínseco`

- **Valor intrínseco**: quanto a opção está ITM. Call: `max(spot − strike, 0)`. Put:
  `max(strike − spot, 0)`. Opções OTM têm intrínseco **zero**.
- **Valor extrínseco (time value)**: tudo o que sobra do prêmio. Reflete **tempo até o
  vencimento** e **volatilidade implícita (IV)**. É exatamente o que o writer "vende" e
  espera capturar.

### Decaimento temporal (theta)

O valor extrínseco **derrete** com o tempo — fenômeno chamado **theta decay**. O decaimento
**acelera de forma não linear** nos últimos ~45 dias e é mais intenso na reta final (últimos
~7–10 dias). Como writers, o theta trabalha **a nosso favor** todos os dias: por isso a Wheel
favorece vencimentos de **30–45 DTE** (sweet spot de decaimento eficiente).

*Fontes:* [Alpaca — Options Trading Overview](https://docs.alpaca.markets/us/docs/options-trading-overview) · [Investopedia — Options Basics](https://www.investopedia.com/options-basics-tutorial-4583012)

---

## 2. Cash-Secured Put (CSP)

**Mecânica:** vender (write) uma **put OTM** com strike ~5–10% **abaixo** do preço atual,
em um ativo que você **estaria disposto a possuir**, mantendo **cash suficiente** para comprar
100 ações ao strike caso seja atribuído.

- **Por que "cash-secured"**: o colateral é **cash**, não margem. Reserva-se
  `strike × 100` em dinheiro por contrato. Isso elimina o risco de uma *naked put* (alavancada)
  e é o que permite que a estratégia seja aprovada em nível baixo de opções.
- **Colateral exigido**: `strike × 100 − prêmio recebido` em cash (ex.: put strike $45 →
  ~$4.500 reservados por contrato).
- **Risco máximo**: a ação ir a **zero**. Perda máxima = `(strike × 100) − prêmio`. É o
  **mesmo risco de comprar 100 ações**, porém com cost basis reduzido pelo prêmio.
- **Breakeven**: `strike − prêmio por ação`. Abaixo disso, começa o prejuízo (já possuindo
  as ações após assignment).
- **Quando é atribuído**: tipicamente quando a put está **ITM no vencimento** (spot < strike).
  Em opções americanas, **early assignment** é possível se o extrínseco virar ~0 (put fundo
  ITM). Resultado: você compra 100 ações ao strike, e seu **cost basis efetivo = strike −
  prêmio**.
- **Melhor cenário**: a put expira **OTM/worthless** → você fica com 100% do prêmio e repete.

> **Exemplo numérico**: AAPL a $200. Vende 1 put strike $180 (10% OTM), 35 DTE, prêmio $3.00.
> Recebe $300. Colateral = $18.000. Se AAPL > $180 no vencimento → fica com os $300 (≈1,7% em
> 35 dias). Se AAPL cair para $170 e for assigned → compra 100 ações a $180 (cost basis
> efetivo $177), com prejuízo não realizado de $700, mas agora pronto para a fase de
> covered call.

---

## 3. Covered Call

**Mecânica:** já possuindo **≥100 ações**, vender (write) uma **call OTM** com strike ~5–10%
**acima** do seu cost basis, para coletar prêmio. "Covered" porque a obrigação de entregar
as ações está coberta pelas ações que você já tem (sem risco de short ilimitado).

- **Colateral**: as próprias **100 ações** (não exige cash adicional).
- **Risco de "called away"**: se a ação subir acima do strike, a call é atribuída e você é
  **obrigado a vender** suas 100 ações ao strike — perdendo o upside acima dele. Você ainda
  lucra (vendeu acima do custo + prêmio), mas **abre mão da valorização extra**.
- **Trade-off de upside**: você troca **potencial de alta ilimitado** por **prêmio garantido
  hoje**. Em forte bull market, isso é caro (vende-se o vencedor cedo demais). Em mercado
  lateral/levemente altista, é ideal.
- **Breakeven** (da posição combinada): `cost basis das ações − prêmio da call`. O prêmio
  reduz seu custo e oferece pequena proteção na queda (limitada ao valor do prêmio).
- **Quando é atribuído**: call **ITM no vencimento** (spot > strike), ou **early** antes de
  **ex-dividend** se o extrínseco for menor que o dividendo.

> **Exemplo**: você foi assigned 100 AAPL a $180 (cost basis efetivo $177). Vende 1 call
> strike $195, 30 DTE, prêmio $2.50 → recebe $250. Se AAPL < $195 no vencimento → fica com
> as ações + $250 e repete. Se AAPL > $195 → vende a $195 (lucro $1.800 de capital + $250 de
> prêmio + os $300 da put original), e volta a vender CSP.

---

## 4. A roda completa (The Wheel)

A Wheel é um ciclo mecânico que alterna venda de prêmio coberto:

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
  [1] Vender Cash-Secured Put (OTM, ~10% abaixo)        │
        │                                              │
        ├── put expira worthless ──► fica com prêmio ──┘  (repete passo 1)
        │
        └── put ITM → ASSIGNMENT → compra 100 ações ao strike
                │
                ▼
  [2] Vender Covered Call (OTM, ~10% acima do custo)
                │
        ├── call expira worthless ──► fica com prêmio ──┐ (repete passo 2)
        │                                              │
        └── call ITM → ações CALLED AWAY (vendidas) ────┘
                │
                ▼
            Volta ao passo [1]  ── repetir ──
```

A Alpaca descreve a roda em três fases idênticas: **(1)** vender cash-secured puts, **(2)**
ser atribuído e possuir a ação, **(3)** vender covered calls contra as ações; repetindo
quando as ações são *called away*.

### Por que funciona

- Coleta **theta** continuamente (dois fluxos de prêmio por ciclo).
- Cada assignment reduz o **cost basis efetivo** (prêmios acumulados).
- Você só "compra" ações de empresas/ETFs que **realmente quer possuir**, e a preços
  abaixo do mercado (strike OTM − prêmio).

### Onde funciona melhor

- Mercados **laterais a levemente altistas**.
- Underlyings **de qualidade, líquidos**, sem risco de falência, com IV razoável.

### Onde falha (riscos estruturais)

- **Queda forte / contínua (bagholding)**: você é assigned a put, a ação despenca muito
  abaixo do strike, e agora segura ações em prejuízo. Pior: vender covered call abaixo do
  seu cost basis para coletar prêmio **trava o prejuízo** se for called away. O maior risco
  da Wheel **não é o assignment em si — é *o quê* você é atribuído, a que preço, em que
  mercado**. Quem se machuca vendeu puts em ações que **nunca quis possuir**, perseguindo
  prêmio alto (prêmio alto = risco alto sinalizado pela IV, não "almoço grátis").
- **Forte bull market**: a covered call **trava o teto** — você vende seus vencedores cedo e
  reentra mais caro. A Wheel **não é all-weather**.
- **Cost basis drift**: após vários ciclos, o cost basis efetivo diverge muito do strike
  original; perder o rastro leva a má seleção de strike e P&L incorreto. **O bot deve manter
  cost basis efetivo como estado persistente.**

*Fontes:* [Alpaca — The Options Wheel Strategy (Python)](https://alpaca.markets/learn/options-wheel-strategy) · [Option Alpha — Wheel Strategy](https://optionalpha.com/blog/wheel-strategy)

---

## 5. Seleção de contratos (critérios automatizáveis)

Critérios objetivos e numéricos — ideais para regras de bot.

### Delta como proxy de probabilidade de assignment

O **delta** (em valor absoluto) é uma aproximação prática da **probabilidade de a opção
expirar ITM** (≈ probabilidade de assignment). Um **delta 0.30** ≈ **30% de chance** de
assignment / ~70% de expirar worthless.

- **CSP**: delta-alvo comum **0.20–0.30** (put), correspondendo a ~70–80% de prob. de expirar
  sem valor. A própria Alpaca, no exemplo de código da Wheel, usa **delta entre −0.40 e −0.20**
  para puts e **0.18 a 0.42** para calls.
- Mais conservador → delta menor (0.15–0.20), menos prêmio, menos assignment.
- Mais agressivo → delta maior (0.30–0.40), mais prêmio, mais assignment.

### DTE (days to expiration)

- **Sweet spot consenso: 30–45 DTE** no momento da entrada — região de **theta decay**
  acelerado, com tempo suficiente para a tese funcionar sem comprometer capital por meses.
- O exemplo oficial da Alpaca usa janela mais curta (**14–35 DTE**); ambas são defensáveis. Para
  o bot, **parametrize** (ex.: `DTE_MIN`, `DTE_MAX`).

### Liquidez

Antes de cotar/operar, **filtrar liquidez** (crítico para automação — preencher a market
order num contrato ilíquido é caro):

- **Open interest** ≥ ~**200** contratos (mínimo que a Alpaca usa no exemplo); muitos usam ≥500/1000.
- **Volume** diário > 0 (idealmente dezenas+).
- **Bid-ask spread** estreito: regra prática **spread ≤ 5–10%** do mid, ou ≤ $0.05–$0.10 em
  termos absolutos em contratos baratos. **Use limit orders no mid**, nunca market em opção.

### IV rank / IV percentile

- **IV rank** = onde a IV atual está no range dos últimos 52 semanas (0–100%). **Vender prêmio
  rende mais quando a IV está alta** (prêmios "gordos").
- Regra comum: **só vender CSP/CC quando IV rank > ~30–50%** (alguns exigem >60%). Em IV alta,
  pode-se ir mais OTM (delta menor) e ainda coletar bom prêmio.

### Strike

- CSP: strike ~**5–10% abaixo** do spot (a Alpaca usa **±5% do preço atual** + filtro de delta).
- CC: strike ~**5–10% acima** do cost basis (a Alpaca filtra acima da **Bollinger Band superior**
  = SMA20 + 2σ, como resistência técnica).

> **Cálculo do delta no bot**: a options chain da Alpaca traz `open_interest`, `strike`,
> `expiration` etc., mas o **delta normalmente é calculado pelo cliente** (Black-Scholes) ou
> obtido de snapshots/greeks quando disponível. O notebook oficial da Wheel implementa
> `calculate_implied_volatility` e `calculate_delta` (Black-Scholes) e descarta contratos
> fora da faixa de delta.

*Fontes:* [Alpaca — Options Wheel Strategy](https://alpaca.markets/learn/options-wheel-strategy) · [Wheel Strategy Options — DTE/Delta/Exit](https://wheelstrategyoptions.com/blog/optimizing-the-wheel-strategy-advanced-dte-delta-and-trade-exit-tactics/)

---

## 6. Gestão da posição

Regras de gestão são **a maior parte do alfa** da Wheel automatizada.

### Fechar a 50% do lucro (take-profit)

Convenção amplamente usada (mecânica tastytrade): **recomprar a opção quando ela perde ~50%
do prêmio inicial**. Razão: o restante do prêmio rende cada vez menos por unidade de risco/dia
(theta marginal cai e gamma sobe perto do vencimento). Fechar cedo **libera capital e reduz
risco de cauda** para reabrir um novo contrato a 30–45 DTE.

- Bot: ordem de **buy-to-close limit** quando `mark_price ≤ 0.50 × premium_recebido`.

### Rolar (roll) a opção

**Roll** = recomprar (buy-to-close) a opção atual e vender (sell-to-open) outra, num único
movimento (idealmente **net credit**). Objetivos:

- **Roll out (no tempo)**: mesmo strike, vencimento mais distante → adia o assignment e coleta
  mais prêmio.
- **Roll out & down (put) / out & up (call)**: ajusta o strike para longe do dinheiro, dando
  fôlego à posição.
- **Regra de ouro**: rolar **apenas por net credit** (o prêmio do novo contrato cobre o custo
  de fechar o antigo). Rolar por débito normalmente só posterga e aumenta o prejuízo.
- Na Alpaca, um roll pode ser feito como **duas pernas** (close + open) ou como **ordem
  multi-leg (`order_class="mleg"`)** se o nível de opções for **Level 3**.

### Gestão de assignment

- Assignment **não é falha** — é **rotação** prevista pela estratégia, *desde que* o underlying
  seja de qualidade, o strike apropriado e o sizing correto.
- Após assignment de **put** → registrar **cost basis efetivo** = `strike − prêmio` e iniciar
  fase de **covered call** (strike ≥ cost basis para não travar prejuízo).
- Após **early assignment** inesperada → o bot precisa **reconciliar posição** (a perna de
  opção some, surgem/somem 100 ações) e **recalcular estado** antes do próximo ciclo.
- **Nunca** vender covered call **abaixo do cost basis** só para coletar prêmio se o objetivo
  é não realizar prejuízo (a menos que a regra do bot aceite sair com pequena perda).

*Fontes:* [tastylive — Managing winners at 50%](https://www.tastylive.com/concepts-strategies/managing-winners) · [Alpaca — Options Wheel Strategy](https://alpaca.markets/learn/options-wheel-strategy)

---

## 7. Greeks essenciais para automação

O mínimo necessário para um bot da Wheel:

| Greek | Mede | Uso na Wheel (writer) |
|---|---|---|
| **Delta (Δ)** | sensibilidade do preço da opção a $1 no underlying; proxy de prob. ITM | **Seleção de strike** (delta-alvo 0.20–0.30) e prob. de assignment. |
| **Theta (Θ)** | ganho/perda por dia pela passagem do tempo | É **positivo para o writer** — a fonte de renda. Maximizado em 30–45 DTE e fechando a 50%. |
| **Vega (ν)** | sensibilidade à IV | Writer é **short vega**: lucra se a IV cai. Vender com **IV rank alto** melhora o edge; cuidado com *vol spikes* (perda mark-to-market). |
| **Gamma (Γ)** | taxa de variação do delta | Risco que **explode perto do vencimento / quando ATM** ("gamma risk"). Motivo extra para **não segurar até o fim** e fechar a 50%. |

Para a maioria das regras do bot, **delta** (entrada) e **theta** (timing/exit) são os dois
indispensáveis; **vega** filtra *quando* vender (IV rank); **gamma** justifica *quando sair*.

*Fonte:* [Investopedia — Option Greeks](https://www.investopedia.com/trading/getting-to-know-the-greeks/)

---

## 8. Suporte a opções na Alpaca (status 2025/2026)

**Confirmado na documentação oficial:** a Alpaca oferece **trading de opções via API** com
níveis **1, 2 e 3**, incluindo **multi-leg (mleg)** no Level 3.

### Níveis de aprovação

Da página oficial de suporte ("What option levels or tiers do you provide?"):

- **Level 1** — **Covered Call** e **Cash-Secured Put**. (Validações: o sistema verifica que
  você possui as ações suficientes / cash/options buying power.)
- **Level 2** — tudo do Level 1 **+ Long Call** e **Long Put**.
- **Level 3** — **Spreads, Straddles, Butterfly Spreads, Iron Condors, outras multi-leg** e
  todas as estratégias dos Levels 1 e 2.

> **Boa notícia para a Wheel**: ela usa **apenas Level 1** (CSP + covered call). É o nível mais
> baixo e mais fácil de aprovar.

**Aprovação/configuração:**
- Para **live trading** é preciso aplicar e assinar o options agreement; o endpoint de aprovação
  retorna `APPROVED`, `LOWER_LEVEL_APPROVED`, `PENDING` ou `REJECTED`. Você pode submeter até
  **duas aplicações** iniciais.
- O nível é controlado por `max_options_trading_level` no endpoint de **account configurations**;
  o nível efetivo (`options_trading_level`) é sempre ≤ `options_approved_level`.
- **Paper trading**: contas paper têm acesso amplo. ✅ **Confirmado na conta paper deste projeto
  (`PA30UJ3ZJK1Q`, jun/2026): `options_approved_level = 3` e `options_trading_level = 3`** — ou seja,
  a Wheel inteira (Level 1) e até multi-leg (Level 3) estão liberadas para testar sem capital real.

### Options market data (feeds)

- A data de opções vem diretamente do **OPRA**. No SDK escolhe-se o feed:
  - `OptionsFeed.OPRA` — feed completo, requer plano pago **Algo Trader Plus**.
  - `OptionsFeed.INDICATIVE` — disponível sem assinatura (capacidades reduzidas).
- Real-time via **WebSocket**; histórico via endpoints de historical option data; **greeks via
  snapshots** quando disponíveis (caso contrário, calcular delta com Black-Scholes no cliente).

### Symbol format e restrições de ordem

- Contratos usam **OCC symbol** (ex.: `AAPL240119C00100000` = AAPL, 2024-01-19, Call, strike
  $100.000).
- Restrições de ordem de opção: `qty` inteiro; **`notional` proibido**; `time_in_force` apenas
  `day` ou `gtc`; **`extended_hours` deve ser false**; `type` ∈ {`market`, `limit`, `stop`,
  `stop_limit`}.
- Multi-leg: `order_class="mleg"` com array `legs` (cada leg: `symbol`, `side`,
  `ratio_qty`, `position_intent`) — **requer Level 3**.

### Código alpaca-py — obter a chain, cotar e enviar ordens

> Imports e nomes de classe **verbatim** do notebook oficial `options-trading-basic.ipynb`.

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import (
    AssetStatus, ExerciseStyle, ContractType,
    OrderSide, OrderType, TimeInForce,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from datetime import datetime, timedelta

API_KEY, SECRET_KEY = "...", "..."

# --- Clients (paper=True para testar) ---
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
option_data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

# --- 0) Checar nível de opções e buying power ANTES de operar ---
acct = trade_client.get_account()
print(acct.options_approved_level, acct.options_trading_level, acct.options_buying_power)
# Wheel exige options_trading_level >= 1

# --- 1) Obter a options chain (puts ~10% OTM, 30-45 DTE) ---
today = datetime.now()
req = GetOptionContractsRequest(
    underlying_symbols=["AAPL"],
    status=AssetStatus.ACTIVE,
    type=ContractType.PUT,
    style=ExerciseStyle.AMERICAN,
    expiration_date_gte=(today + timedelta(days=30)).date(),
    expiration_date_lte=(today + timedelta(days=45)).date(),
    strike_price_lte=str(round(spot * 0.90, 2)),  # ~10% OTM
    limit=100,
)
contracts = trade_client.get_option_contracts(req).option_contracts

# --- 2) Cotar (latest quote) e filtrar liquidez/spread ---
candidates = []
for c in contracts:
    if (c.open_interest or 0) < 200:          # filtro de liquidez
        continue
    q = option_data_client.get_option_latest_quote(
        OptionLatestQuoteRequest(symbol_or_symbols=c.symbol)
    )[c.symbol]
    bid, ask = q.bid_price, q.ask_price
    if bid <= 0 or ask <= 0:
        continue
    mid = (bid + ask) / 2
    if (ask - bid) / mid > 0.10:              # spread <= 10% do mid
        continue
    candidates.append((c, mid))
    # delta: calcular via Black-Scholes (IV implícita do mid) e exigir 0.20-0.30

# --- 3a) Vender CASH-SECURED PUT (limit no mid, sell-to-open) ---
contract, mid = candidates[0]
put_order = LimitOrderRequest(
    symbol=contract.symbol,        # OCC symbol
    qty=1,                          # 1 contrato = 100 ações de colateral
    side=OrderSide.SELL,            # writer
    type=OrderType.LIMIT,
    limit_price=round(mid, 2),
    time_in_force=TimeInForce.DAY,
)
trade_client.submit_order(put_order)

# --- 3b) Vender COVERED CALL após possuir 100 ações (sell-to-open) ---
# (selecionar call ~10% acima do cost basis com a mesma lógica de chain/quote acima)
call_order = LimitOrderRequest(
    symbol=call_contract_symbol,
    qty=1,                          # coberto pelas 100 ações em carteira
    side=OrderSide.SELL,
    type=OrderType.LIMIT,
    limit_price=round(call_mid, 2),
    time_in_force=TimeInForce.DAY,
)
trade_client.submit_order(call_order)
```

> **Resposta direta à pergunta "a API suporta vender put/call?"**: **Sim.** Vender CSP e
> covered call exige apenas **Level 1**, e a venda é uma ordem normal `POST /orders` com
> `side=SELL` sobre o **OCC symbol** do contrato. **Não há limitação** para a Wheel single-leg.
> O **delta não vem pronto** na chain básica — calcule via Black-Scholes (o próprio exemplo
> oficial da Alpaca faz isso) ou via snapshots/greeks quando disponíveis.

*Fontes:* [Alpaca — Options Trading Overview](https://docs.alpaca.markets/us/docs/options-trading-overview) · [Alpaca — Options Trading (API)](https://docs.alpaca.markets/us/docs/options-trading) · [Alpaca Support — Option levels/tiers](https://alpaca.markets/support/what-option-levels-or-tiers-do-you-provide) · [Alpaca Support — Multi-leg orders](https://alpaca.markets/support/what-are-multi-leg-orders) · [alpaca-py — exemplo options-trading-basic.ipynb](https://github.com/alpacahq/alpaca-py/blob/master/examples/options) · [alpaca-py — Option Market Data](https://alpaca.markets/sdks/python/api_reference/data/option.html)

---

## 9. Riscos e erros comuns ao automatizar a Wheel

**Riscos da estratégia (já cobertos)**: bagholding em queda forte, upside travado em bull
market, cost basis drift. Abaixo, os erros **de automação**.

### Erros comuns de automação

1. **Operar contrato ilíquido** → market order com slippage enorme. Sempre **limit no mid**
   e filtrar OI/volume/spread antes.
2. **Ignorar early assignment (americana)** → o bot assume estado errado. Tratar assignment
   (inclusive antecipada, perto de ex-dividend) como evento assíncrono e **reconciliar posição**.
3. **Não rastrear cost basis efetivo** → vende covered call abaixo do custo e trava prejuízo.
4. **Vender call abaixo do cost basis** após queda, só por prêmio.
5. **Capital insuficiente para 100 ações** no momento do assignment → ordem rejeitada ou margin
   call. Garantir `cash ≥ strike × 100` por CSP aberta.
6. **Perseguir prêmio/IV alta** em ações que você não quer possuir → maior causa de perdas.
7. **Sizing demais** → concentração. Cap recomendado pela Alpaca: **≤10% do buying power** por
   posição.
8. **Confundir feed** (`INDICATIVE` vs `OPRA`) → dados defasados/parciais para decisão.
9. **Ignorar dividendos e earnings** → IV spike e risco de gap; muitos pulam o ciclo de earnings.
10. **Testar só em paper e ir direto a live** sem reconciliação robusta de fills/posições.

### Checagens obrigatórias antes de habilitar live (pre-flight do bot)

- [ ] **Nível de opções aprovado ≥ 1**: `acct.options_approved_level >= 1` e
      `options_trading_level >= 1`.
- [ ] **Options buying power suficiente**: `acct.options_buying_power` cobre o colateral.
- [ ] **Cash ≥ strike × 100** por CSP (cash-secured de verdade; não usar margem).
- [ ] **Possui ≥100 ações** antes de vender covered call (`qty` em carteira).
- [ ] **Liquidez do contrato**: OI ≥ limiar, spread ≤ limiar, volume > 0.
- [ ] **Delta dentro da faixa-alvo** (ex.: 0.20–0.30) e **DTE** na janela (ex.: 30–45).
- [ ] **IV rank** acima do mínimo configurado.
- [ ] **Feed correto** configurado (`OPRA` se assinante; senão `INDICATIVE`).
- [ ] **Sizing** ≤ limite por posição (ex.: 10% do buying power) e exposição agregada.
- [ ] **Reconciliação de assignment** implementada e testada em paper.
- [ ] **Sem earnings/ex-dividend** dentro da janela (se a regra do bot evitar).

*Fontes:* [Alpaca — Options Wheel Strategy](https://alpaca.markets/learn/options-wheel-strategy) · [SteadyOptions — Wheel explained](https://steadyoptions.com/articles/the-options-wheel-strategy-wheel-trade-explained-r632/)

---

## Fontes

- Alpaca — Options Trading Overview: https://docs.alpaca.markets/us/docs/options-trading-overview
- Alpaca — Options Trading (API): https://docs.alpaca.markets/us/docs/options-trading
- Alpaca Support — What option levels/tiers do you provide: https://alpaca.markets/support/what-option-levels-or-tiers-do-you-provide
- Alpaca Support — What are Multi-leg Orders: https://alpaca.markets/support/what-are-multi-leg-orders
- Alpaca — Options Level 3 Trading: https://docs.alpaca.markets/docs/options-level-3-trading
- Alpaca — Real-time Option Data: https://docs.alpaca.markets/docs/real-time-option-data
- Alpaca Learn — The Options Wheel Strategy (Python): https://alpaca.markets/learn/options-wheel-strategy
- Alpaca Learn — How To Trade Options with Alpaca: https://alpaca.markets/learn/how-to-trade-options-with-alpaca
- alpaca-py — exemplos de opções (GitHub): https://github.com/alpacahq/alpaca-py/tree/master/examples/options
- alpaca-py — Option Market Data (API reference): https://alpaca.markets/sdks/python/api_reference/data/option.html
- Option Alpha — Wheel Strategy: https://optionalpha.com/blog/wheel-strategy
- Wheel Strategy Options — DTE/Delta/Exit tactics: https://wheelstrategyoptions.com/blog/optimizing-the-wheel-strategy-advanced-dte-delta-and-trade-exit-tactics/
- tastylive — Managing winners (50% rule): https://www.tastylive.com/concepts-strategies/managing-winners
- Investopedia — Option Greeks: https://www.investopedia.com/trading/getting-to-know-the-greeks/

---

## Como virar skill

Este documento é a base de conhecimento de uma skill `wheel-opcoes` para o bot multi-agente.
Para promovê-lo a skill:

1. **Estrutura**: criar `skills/wheel-opcoes/SKILL.md` com o frontmatter já presente
   (`name`, `description`, `metadata.type: reference`) e um corpo enxuto: **quando acionar**
   (gerar/avaliar trades da Wheel, selecionar contratos, decidir roll/exit, checar pré-requisitos
   Alpaca) e **o que carregar** (este `.md` como referência completa).
2. **Encapsular regras automatizáveis** como parâmetros/constantes (não prosa):
   `DELTA_MIN/MAX`, `DTE_MIN/MAX`, `OI_MIN`, `MAX_SPREAD_PCT`, `IV_RANK_MIN`,
   `MAX_POSITION_PCT`, `TAKE_PROFIT_PCT=0.50`, `ROLL_ONLY_FOR_CREDIT=True`.
3. **Helpers de referência** (apontar para módulos do bot): `get_chain()`, `quote_and_filter()`,
   `calc_delta_bs()`, `submit_csp()`, `submit_covered_call()`, `roll()`, `reconcile_assignment()`.
4. **Pre-flight checklist** da Seção 9 vira uma função `assert_ready_to_trade(acct, contract)`
   que **bloqueia** a execução se qualquer item falhar (nível, buying power, liquidez,
   cash ≥ strike×100, sizing).
5. **Separar paper de live**: a skill exige rodar a bateria de reconciliação em **paper**
   (Level 3 automático) antes de habilitar live (Level 1).
6. **Disclaimer**: incluir que opções têm risco substancial e remeter ao documento
   "Characteristics and Risks of Standardized Options".
