---
name: smart-money-signals
description: Referência sobre sinais de "Smart Money" (copy trading de congressistas e grandes fundos), suas fontes de dados, APIs, limitações de latência e como normalizá-los em um sinal acionável para um bot de trading.
metadata:
  type: reference
---

# Smart Money Signals: Copy Trading de Congressistas e Grandes Fundos

> **TL;DR honesto para um bot EXEMPLAR.** Os dados de "smart money" (congressistas, 13F, insiders)
> são **públicos, legais e ótimos como _features_**, mas chegam com **latência de semanas a meses**.
> O "edge" mais agudo geralmente **já passou** quando o dado vira público. Trate isso como **um sinal
> entre vários** — nunca como gatilho de execução automática cega. O valor real está em **consenso,
> tamanho de posição, ranking de quem está negociando e confirmação cruzada**, não em copiar 1:1.

---

## 1. Trades do Congresso dos EUA (STOCK Act)

### O que é o STOCK Act
O **STOCK Act** (Stop Trading on Congressional Knowledge Act, 2012) obriga membros do Congresso,
certos funcionários e cônjuges/dependentes a divulgarem publicamente transações de valores
mobiliários (ações, bonds, futures, opções etc.) **acima de US$ 1.000** por meio dos
**Periodic Transaction Reports (PTRs)**.

### Prazo de divulgação e o que isso faz com a latência do sinal
- O filer deve reportar **dentro de 30 dias após tomar ciência** da transação, **mas nunca depois de
  45 dias** da data da transação.
- Tradução prática: um trade executado em **1º de janeiro** só precisa aparecer publicamente em
  **~15 de fevereiro**. Na prática, **muitos membros filam ainda mais tarde** que o limite de 45 dias.
- **Consequência direta para o bot:** o sinal é **estrutural e atrasado**. Quando você lê o PTR, o
  preço já reagiu à notícia/decisão que motivou o trade (orçamento, contrato, regulação, earnings).
  A janela de 45 dias é **mais longa do que a maioria dos catalisadores que movem o preço**.

### Eficácia real — seja honesto
- Estudos acadêmicos (Senado e Câmara) **encontraram retornos anormais** em carteiras "buy-minus-sell"
  de congressistas, e pesquisa do **NBER** aponta que ganhos se concentram em **líderes do Congresso**
  (que superariam pares comparáveis em dezenas de pontos percentuais ao ano após assumirem a liderança).
- **PORÉM**: grande parte do valor informacional **dissipa-se** no atraso de 45 dias. Replicar trades
  via disclosure público **não é o mesmo** que ter tido a informação na hora. Os ETFs que copiam (ver §2)
  tiveram bom desempenho recente, mas isso é **fortemente influenciado pela concentração em megacaps de
  tecnologia** (NVDA, MSFT, AAPL, AMZN, GOOGL) — ou seja, parte do "alfa" é **exposição a beta de Big Tech**,
  não necessariamente skill informacional replicável.
- **Enforcement é fraco:** a multa por atraso é simbólica (~US$ 200, frequentemente perdoada), então
  a **qualidade/completude dos dados varia** e há ruído (faixas de valor amplas, não valores exatos).

> **Conclusão para o design do bot:** congressistas servem melhor como **sinal de viés/tilt de médio prazo
> e como confirmação de consenso**, não como gatilho de timing. Pondere pelo **player** (líderes de comitê
> relevantes > membro mediano) e pelo **consenso** (vários membros comprando o mesmo nome).

**Fontes:**
- [Congress.gov — Laws Governing Financial Disclosure (R48641)](https://www.congress.gov/crs-product/R48641)
- [Campaign Legal Center — Oversight of Congressional Stock Trades](https://campaignlegal.org/cases-actions/we-need-stronger-oversight-congressional-stock-trades)
- [ScienceDirect — What explains trading behaviors of members of Congress? (100k+ trades)](https://www.sciencedirect.com/science/article/abs/pii/S1059056024005835)
- [Congress vs S&P 500 — análise de desempenho](https://congressflow.com/learn/congress-vs-sp500)

---

## 2. Fontes de dados de trades de políticos

> **Fonte primária e 100% gratuita:** os feeds oficiais de disclosure.
> Todo o resto é **valor agregado** (parsing, normalização, API limpa) cobrado por cima de um dado público.

| Fonte | Tipo | Grátis? | API pública? | Formato | Notas |
|---|---|---|---|---|---|
| **House Clerk disclosures** | Oficial primária | Sim | Não (sem API; ZIP/XML + PDFs) | XML índice + **PDF dos PTRs** | PDFs frequentemente escaneados → exige OCR/parsing pesado |
| **Senate eFD** | Oficial primária | Sim | Não (busca web, exige aceitar termos) | HTML/PDF | Scraping mais hostil; sessão/cookies |
| **Capitol Trades** | Agregador | Sim (web + API interna) | **Sem API _oficial_; existe BFF interno não-documentado** (`bff.capitoltrades.com/trades`) | JSON | Melhor custo-benefício gratuito; endpoint instável/ToS — ver §2.1 |
| **Quiver Quantitative** | Agregador + API | Tier grátis (atrasado/limitado) | **Sim** | JSON | Pago a partir de ~US$ 25–30/mês; endpoints separados House/Senate |
| **Unusual Whales** | Agregador + API | Tier grátis limitado | **Sim** (100+ endpoints) | JSON / OpenAPI | API paga (Basic ~US$150/mês, Advanced ~US$375/mês em 2025); inclui cônjuge/dependente |
| **ETFs NANC / GOP (ex-KRUZ)** | Produto investível | Compra na bolsa | N/A | — | Replicação "pronta", taxa 0,75% |

### Detalhamento

**Feeds oficiais (gratuitos, sem API amigável):**
- **House:** índice anual em XML + PDFs dos PTRs em
  `https://disclosures-clerk.house.gov/PublicDisclosure/FinancialDisclosure`
  (downloads em `.../FD.zip`). O índice é tratável; os PTRs em si costumam ser **PDF (às vezes escaneado)**,
  exigindo OCR.
- **Senate:** `https://efdsearch.senate.gov/search/` — busca web que **exige aceitar termos** antes de
  retornar resultados (dificulta automação). Sem API JSON oficial.
- **Realidade:** consumir o oficial **direto é trabalhoso** (PDF/OCR/scraping). É exatamente o gap que
  Capitol Trades, Quiver e Unusual Whales preenchem.

**Capitol Trades** — `https://www.capitoltrades.com`
- Agregador gratuito com ótima UI (filtros por político, ticker, comitê), já com **ticker normalizado**
  e **faixas de valor parseadas** — exatamente o trabalho pesado que o feed oficial não entrega.
- **Sem API pública oficial documentada**, mas o site é alimentado por uma **API interna do tipo BFF
  (Backend-for-Frontend)** que retorna JSON limpo. É a forma mais prática de usar o Capitol Trades como
  **banco de dados** do bot. Detalhes completos em **§2.1** abaixo. Provider pronto em **§7**.

---

### 2.1. Capitol Trades como banco de dados (API interna BFF)

> ⚠️ **Status (verificado em jun/2026):** este endpoint é **não-oficial e não-suportado**. Ele é
> protegido por CloudFront e, num teste direto, **retornou `503` para `curl` simples** — ou seja, exige
> headers de browser corretos e pode ser bloqueado, ter rate-limit agressivo ou mudar sem aviso. **Não é
> uma dependência de produção confiável.** Trate como *best-effort*, com cache local e fallback. O uso
> programático pode também conflitar com os **Termos de Serviço** do site — revise antes de usar comercialmente.

**Endpoint principal**
```
GET https://bff.capitoltrades.com/trades?page=<n>&pageSize=<=100>
```
- `pageSize` máximo **100**; pagine com `page` até cobrir `meta.paging.totalItems`.
- Outros parâmetros observados na UI (passados como repetição de query string):
  `txDate=<YYYY-MM-DD>`, `politician=<politicianId>`, `issuer=<issuerId>`, `assetType`,
  `txType=buy|sell`, `chamber=house|senate`, `sortBy=-txDate` (prefixo `-` = desc).

**Headers necessários** (sem eles → 403/503):
```
User-Agent: Mozilla/5.0 (... browser real ...)
Accept: application/json, text/plain, */*
Referer: https://www.capitoltrades.com/
Origin:  https://www.capitoltrades.com
```

**Formato da resposta**
```jsonc
{
  "data": [ /* lista de trades */ ],
  "meta": { "paging": { "page": 1, "size": 100, "totalItems": 12345, "totalPages": 124 } }
}
```

**Schema de um trade** (campos reconstruídos a partir de wrappers comunitários — *confirme contra a
resposta real ao implementar, pois nomes podem mudar*):

| Campo | Significado | Mapeia p/ `Signal` |
|---|---|---|
| `_txId` | ID único da transação | `raw` / dedup |
| `txDate` | Data da transação (`YYYY-MM-DD`) | `traded_at` |
| `pubDate` | Data da publicação (`...Z`) | `filed_at` |
| `txType` | `buy` / `sell` / `exchange` / `receive` | `side` |
| `owner` | self / spouse / child / joint | ponderação de `confidence` |
| `politician.fullName` (`firstName`+`lastName`) | Nome do congressista | `actor` |
| `politician.party` | `democrat` / `republican` / ... | `raw` (ranking) |
| `politician.chamber` (ou `chamber`) | `house` / `senate` | `raw` |
| `issuer.issuerTicker` | Ticker (ex.: `NVDA`) | `ticker` |
| `issuer.issuerName` | Nome da empresa | `raw` |
| `asset.assetType` | `stock` / `stock-options` / `etf` / ... | filtro |
| `size` | Bucket textual (ex.: `"1K–15K"`) | — |
| `sizeRangeLow` / `sizeRangeHigh` | Faixa numérica em USD | `size_usd_low` / `size_usd_high` |
| `value` | Valor estimado (mid da faixa) | `raw` |
| `price` | Preço do ativo na data (quando disponível) | `raw` |

**Estratégias de acesso (em ordem de preferência):**
1. **BFF direto** (§7) — mais simples; sujeito a bloqueio. Mantenha cache em SQLite e *backoff*.
2. **Wrappers/MCP comunitários** — ex.: servidor MCP `mcp-capitol-trades`, libs como `CongressionalTrader`.
   Úteis como referência de headers/parsing, mas herdam a mesma fragilidade do BFF.
3. **Serviços de scraping gerenciado** (Apify "Capitol Trades Scraper", ScrapingBee) — pagos, mas
   absorvem anti-bot/manutenção. Bom se o BFF ficar instável.
4. **Fallback robusto:** se o objetivo é confiabilidade contratual, prefira **Quiver/Unusual Whales (pagos,
   com SLA)** ou a **fonte oficial** (House/Senate) — Capitol Trades é o melhor *gratuito*, não o mais estável.

**Fontes (acesso programático ao Capitol Trades):**
- [ericz1803/CongressionalTrader — cliente Python do BFF](https://github.com/ericz1803/CongressionalTrader/blob/master/capitoltrades/CapitolTrades.py)
- [anguslin/mcp-capitol-trades — servidor MCP (sem API key)](https://github.com/anguslin/mcp-capitol-trades)
- [Apify — Capitol Trades Scraper API](https://apify.com/saswave/capitol-trades-scraper/api)
- [Lambda Finance — Best Capitol Trades APIs (2026)](https://www.lambdafin.com/articles/capitol-trades-api)

**Quiver Quantitative** — `https://www.quiverquant.com/congresstrading/`
- **API documentada e oficial:** `https://api.quiverquant.com/`
  (dataset: `https://api.quiverquant.com/datasets/congress-trades`).
- **Tier grátis** existe, porém **com dados atrasados e profundidade histórica limitada**; **API completa
  é paga** (a partir de ~US$ 25–30/mês).
- **Formato:** JSON estruturado com nome do membro, partido, ticker, tipo de transação e **faixa de valor**.
- **Endpoints (base `https://api.quiverquant.com/beta`):**
  - `GET /live/congresstrading` — trades recentes de **todos** os membros (o que o bot usa).
  - `GET /bulk/congresstrading` — histórico completo (todos os membros).
  - `GET /historical/congresstrading/{ticker}` — por ticker.
- **Auth:** o pacote oficial usa `Authorization: Token <API_KEY>` (a doc web menciona `Bearer` — se
  `Token` falhar, tente `Bearer`). Provider pronto em **§7**.
- **Campos** (PascalCase): `Representative`/`Name`, `Ticker`, `Transaction` (`Purchase`/`Sale`),
  `TransactionDate`, `ReportDate`, `Range`, `Amount`, `House`, `Party`, `TickerType`.

**Unusual Whales** — `https://unusualwhales.com/`
- **API documentada:** `https://api.unusualwhales.com/docs`
  (OpenAPI YAML: `https://api.unusualwhales.com/api/openapi`; dev portal: `https://unusualwhales.com/developers`).
- Mesmo núcleo de dados do STOCK Act (House + Senate), **incluindo transações de cônjuge/dependente** e
  **visões de portfólio por político**.
- **Endpoints de Congresso (base `https://api.unusualwhales.com`):**
  - `GET /api/congress/recent-trades` — trades recentes de todos os membros (o que o bot usa).
    Params: `limit`, `date`, `ticker`. Resposta em `{"data": [...]}`.
  - `GET /api/congress/congress-trader` — relatórios por trader · `GET /api/congress/late-reports`.
- **Auth:** `Authorization: Bearer <API_KEY>` (chave em unusualwhales.com/settings/api-dashboard). Provider em **§7**.
- **Campos** (snake_case): `ticker`, `txn_type` (`Buy`/`Sell`), `transaction_date`, `filed_at_date`,
  `amounts` (string tipo `"$1,001 - $15,000"` → parsear), `name`, `reporter`, `issuer`.
- **Tier grátis limitado**; **API paga** com tiers (em 2025: Trial ~US$50/semana, Basic ~US$150/mês,
  Advanced ~US$375/mês). A API cobre também options flow, dark pool, 13F, insiders.

**ETFs que replicam (replicação "pronta"):**
- **NANC** — *Unusual Whales Subversive Democratic Trading ETF*: compra o que **democratas** (e cônjuges)
  divulgam.
- **GOP** (anteriormente **KRUZ**) — *Unusual Whales Subversive Republican Trading ETF*: lado **republicano**.
  Atenção: o ticker **KRUZ foi alterado para GOP**; valide o ticker corrente antes de codar.
- Lançados em 07/02/2023 na Cboe BZX, **taxa de administração 0,75%**, gestão ativa.
- Úteis como **benchmark do sinal** ("o sinal supera só comprar NANC?") e como sanity check de que sua
  replicação não está só vendendo beta de Big Tech.

**Fontes:**
- [Quiver — Congress Trades API](https://api.quiverquant.com/datasets/congress-trades)
- [Unusual Whales API Docs](https://api.unusualwhales.com/docs) · [Developers](https://unusualwhales.com/developers)
- [Unusual Whales — Congress Trading Report 2025](https://unusualwhales.com/congress-trading-report-2025)
- [Lambda Finance — Best Capitol Trades APIs (2026)](https://www.lambdafin.com/articles/capitol-trades-api)
- [Subversive ETFs (NANC/GOP)](https://subversiveetfs.com/) · [Morningstar — os 2 ETFs](https://www.morningstar.com/funds/2-etfs-that-track-congressional-stock-trades)

---

## 3. 13F Filings (grandes fundos / "whales")

### O que é um 13F
O **Form 13F** é o relatório trimestral que **gestores institucionais com ≥ US$ 100 milhões** em
"13(f) securities" (basicamente **ações long listadas nos EUA**) devem enviar à SEC, listando suas posições
no fim do trimestre.

### Prazo
**Até 45 dias após o fim do trimestre.** Ex.: posições de **31/mar** só precisam ser publicadas até
**~15/mai**. De novo: **latência grande** — você vê um retrato **defasado em até 45 dias** de uma carteira
que pode já ter mudado.

### Limitações críticas (não ignore no design)
- **Só posições _long_ em US equities** (e alguns ADRs/opções listadas/conversíveis). **Não mostra shorts**,
  caixa, renda fixa, FX, commodities, posições internacionais nem **a maior parte das opções/derivativos**.
- É um **snapshot trimestral**, não um fluxo: você não sabe *quando* dentro do trimestre o gestor entrou/saiu.
- **Confidential treatment:** gestores podem pedir para **omitir** posições por um período (acumulação),
  então o 13F pode estar **incompleto** justamente nos nomes mais "alfa".
- Para hedge funds long/short, um 13F **só mostra metade do livro** — copiar cegamente pode ser **o oposto**
  da tese real (a perna long pode ser hedge de um short maior).

> **Conclusão para o bot:** 13F é bom para **temas e consenso de "whales" em large/mega caps**, e para
> **detecção de novas posições/aumentos relevantes**, não para timing nem para fundos cujo edge está em
> shorts/derivativos.

### Fontes

| Fonte | Grátis? | API | Formato |
|---|---|---|---|
| **SEC EDGAR** | **Sim, 100%** | **Sim, gratuita, sem API key** (User-Agent obrigatório) | JSON (submissions) + **XML** do 13F |
| **WhaleWisdom** | Muito conteúdo grátis no site | API **paga** (planos por assinatura) | JSON/CSV |
| **Quiver / Unusual Whales** | Tier grátis limitado | Sim (pago p/ completo) | JSON |

**SEC EDGAR (a fonte canônica e gratuita):**
- **Sem API key.** Exige **header `User-Agent`** com nome + e-mail de contato (sem ele → **403**).
- **Rate limit:** **~10 req/s por IP**; estourar → **429** e bloqueio temporário (~10 min). Boa prática:
  ~100 ms entre requests, cache local, e usar bulk data para varreduras grandes.
- Endpoints úteis:
  - **Submissions por entidade:** `https://data.sec.gov/submissions/CIK##########.json` (CIK com 10 dígitos,
    zero-padded). Lista filings recentes (form type, data, accession number, doc primário).
  - **Full-Text Search:** `https://efts.sec.gov/LATEST/search-index?q=...` (UI: `https://www.sec.gov/edgar/search/`).
    Filings indexados em < 60 s após publicação.
  - **Documento 13F:** o "information table" vem em **XML** (lista de issuers, CUSIP, valor, nº de ações,
    tipo de investimento, discricionariedade).
- Fluxo típico: achar o **CIK** do gestor → ler `submissions/CIK....json` → filtrar `form == "13F-HR"`
  → baixar o **XML da information table** do accession → parsear holdings → fazer **diff trimestre-a-trimestre**
  (novas posições, aumentos, zeradas).

**WhaleWisdom** — `https://whalewisdom.com/`
- Muito conteúdo navegável **grátis** no site (rankings, históricos de 13F, "13F clones").
- **API é paga** (planos de assinatura). Útil se você não quer manter o pipeline de parsing do EDGAR.

**Quiver / Unusual Whales:** ambos expõem 13F via API (pago para acesso completo), já normalizado em JSON —
conveniência sobre o EDGAR cru.

**Fontes:**
- [SEC — Accessing EDGAR Data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [SEC — EDGAR Full Text Search](https://www.sec.gov/edgar/search/)
- [tldrfiling — EDGAR rate limits & User-Agent](https://tldrfiling.com/blog/sec-edgar-api-rate-limits-best-practices)
- [WhaleWisdom — pricing](https://whalewisdom.com/pricing)

---

## 4. Outros sinais "smart money" (menção breve)

- **Insider trading — Form 4 (e 3/5):** executivos/diretores/donos de ≥10% reportam compras/vendas
  **até 2 dias úteis** após a transação → **muito mais rápido** que 13F/PTR. **Compras de insiders**
  ("cluster buys" por vários insiders) são historicamente o subsinal **mais informativo** desta família;
  vendas são ruidosas (diversificação, impostos, RSU vesting). Fontes:
  **SEC EDGAR (grátis, XML)**, [OpenInsider](http://openinsider.com) (grátis, web),
  [SECForm4.com](https://www.secform4.com/), e APIs pagas (EODHD, sec-api.io, Quiver, Unusual Whales).
- **Unusual options activity (UOA):** volume/posicionamento anômalo em opções (grandes blocos, sweeps,
  vol implícita). Sinal **rápido** mas **muito ruidoso** (hedge ≠ aposta direcional). Fonte típica:
  **Unusual Whales** (paga), Cheddar Flow, FlashAlpha.
- **Dark pool:** prints de grandes blocos fora de bolsa (ATS). Indica **acumulação/distribuição
  institucional**, mas atribuição direcional é incerta. Fonte: **Unusual Whales** (paga),
  feeds FINRA ATS (agregados, atrasados, grátis).

> Form 4 (compras em cluster) é o que melhor complementa congressistas/13F porque **fecha o gap de latência**.
> UOA e dark pool são **confirmatórios**, não primários.

**Fontes:**
- [Unusual Whales — Insider trades](https://unusualwhales.com/insiders/trades) · [Form 4 dataset (sec-api.io)](https://sec-api.io/datasets/form-4)
- [EODHD — Insider Transactions API](https://eodhd.com/financial-apis/insider-transactions-api)

---

## 5. Como transformar em sinal acionável

O objetivo **não** é copiar trades 1:1. É produzir um **score normalizado por ticker** que o Agente Planejador
combina com outros sinais (técnicos, fundamentais, macro, risco).

1. **Ingestão e normalização** — cada fonte vira um `Signal` padronizado (ver §7): `ticker`, `side`
   (buy/sell), `size_bucket`, `actor`, `actor_class` (congress/whale/insider), `filed_at`, `traded_at`,
   `latency_days`, `confidence`.
2. **De-duplicação** — o mesmo trade aparece em EDGAR, Quiver e Unusual Whales. Deduplique por
   `(actor, ticker, traded_at, side, size_bucket)` e mantenha a **fonte mais precisa/oficial** como cânon.
3. **Ponderação por ator (ranking de quem negocia):**
   - Congressistas: peso maior para **membros de comitês relevantes ao setor** (ex.: membro do comitê de
     defesa comprando defense), líderes, e quem tem **track record** mensurável.
   - 13F: peso por **convicção** (tamanho da posição vs AUM, **novas** posições > carryover) e por gestor;
     descarte fundos puramente long/short onde o 13F é meia-foto.
   - Insiders: **compras > vendas**; **cluster buys** > insider solitário.
4. **Tamanho da posição / materialidade** — normalize a faixa de valor (PTR/13F dão **ranges**) e exija um
   **mínimo de materialidade** relativo ao ator. Ignore "ruído" pequeno.
5. **Consenso entre players** — o sinal **forte** é **vários atores independentes** (e idealmente de classes
   diferentes: congressista + whale + insider) convergindo no mesmo nome/direção em janela próxima.
   Modele isso explicitamente (ex.: `consensus_count`, `cross_class_bonus`).
6. **Decaimento por latência** — aplique **decay** em função de `latency_days` (o sinal vale menos quanto
   mais velho). Para 13F/PTR, o decay é agressivo; para Form 4, mais suave.
7. **Filtro de exposição/beta** — neutralize ou penalize sinais que são só **"comprar Big Tech de novo"**
   se o portfólio já tem essa exposição. Compare o sinal contra o benchmark NANC/GOP para não pagar por
   alfa que é, na verdade, beta.
8. **Saída = score, não ordem.** O provider entrega um **score por ticker** com metadados; **quem decide
   risco/sizing/execução é o Agente Planejador**, sempre com gestão de risco por cima.

> **Por que NUNCA execução automática cega:** latência alta + dados em faixas + 13F meia-foto + risco de
> erro de parsing (PDF/OCR) + possibilidade de o trade ser hedge, não aposta. Um único sinal smart money
> mal-interpretado pode te colocar exatamente do lado errado de um gestor long/short. **Sempre um voto,
> nunca o juiz.**

---

## 6. Considerações legais e éticas

- **É legal.** Todos esses dados são **divulgações públicas obrigatórias** (STOCK Act, 13F, Section 16).
  Consumi-los e negociar com base neles **não** é insider trading — você está usando informação **já pública**.
- **Cuidados:**
  - **Termos de uso (ToS):** scraping de Capitol Trades, Senate eFD, House Clerk ou de APIs pagas pode
    **violar ToS**. Prefira **APIs oficiais/licenciadas** e respeite robots/ToS. Para EDGAR, cumpra o
    **User-Agent** e o **rate limit (~10 req/s)** — caso contrário você é bloqueado e tecnicamente viola a
    política de uso aceitável.
  - **Não retransmita dados pagos** (Quiver/Unusual Whales/WhaleWisdom) além do que a licença permite.
  - **Qualidade do dado = risco financeiro:** PTRs em PDF escaneado e faixas de valor amplas geram **erro de
    parsing**. Trate dados malformados como **inválidos**, não como zero. Logue proveniência de cada sinal.
  - **Sem aconselhamento/representação:** um bot que opera capital de terceiros pode atrair obrigações
    regulatórias (advisor/broker). Para uso **próprio/educacional** (este projeto EXEMPLAR), mantenha o
    escopo claro e documentado.
  - **Ética de "front-running de políticos":** é legalmente permitido, mas mantenha o sistema **auditável**
    e os disclaimers explícitos de que é **um sinal estatístico atrasado**, não uma garantia.

---

## 7. Esboço da interface `SignalProvider` (Python)

Interface abstrata que **normaliza** sinais de fontes heterogêneas (congresso, 13F, insiders) para um
formato único que o **Agente Planejador** consome. Inclui um provider concreto consumindo a **SEC EDGAR API
(gratuita)**.

```python
from __future__ import annotations

import abc
import datetime as dt
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import requests  # pip install requests


# ----------------------------- Modelo normalizado ----------------------------- #

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ActorClass(str, Enum):
    CONGRESS = "congress"      # PTR / STOCK Act
    WHALE = "whale"            # 13F institucional
    INSIDER = "insider"        # Form 4 (Section 16)
    OPTIONS_FLOW = "options"   # UOA
    DARK_POOL = "dark_pool"


@dataclass(frozen=True)
class Signal:
    """Sinal smart money normalizado. Imutável e serializável."""
    ticker: str
    side: Side
    actor: str                       # ex.: "Nancy Pelosi", "Berkshire Hathaway"
    actor_class: ActorClass
    traded_at: dt.date | None        # data da transação (pode faltar em 13F)
    filed_at: dt.date                # data da divulgação pública
    size_usd_low: float | None       # faixas (PTR/13F dão ranges)
    size_usd_high: float | None
    source: str                      # proveniência (ex.: "sec_edgar:13F-HR")
    confidence: float = 0.5          # 0..1, definido por regras de ponderação
    raw: dict = field(default_factory=dict)  # payload bruto p/ auditoria

    @property
    def latency_days(self) -> int | None:
        if self.traded_at is None:
            return None
        return (self.filed_at - self.traded_at).days

    @property
    def dedup_key(self) -> tuple:
        return (self.actor.lower(), self.ticker.upper(),
                self.traded_at, self.side, self.size_usd_low)


# ----------------------------- Interface abstrata ----------------------------- #

class SignalProvider(abc.ABC):
    """Contrato comum. Cada fonte (EDGAR, Quiver, Unusual Whales...) implementa isto."""

    name: str
    actor_class: ActorClass

    @abc.abstractmethod
    def fetch(self, since: dt.date) -> Iterable[Signal]:
        """Retorna sinais com filed_at >= since. NÃO executa trades — só normaliza dados."""
        raise NotImplementedError

    def healthcheck(self) -> bool:
        """Sobrescreva para validar credenciais/conectividade antes do loop principal."""
        return True


# ------------------- Provider concreto: SEC EDGAR (GRATUITO) ------------------- #

class SecEdgarProvider(SignalProvider):
    """
    Consome a SEC EDGAR API (gratuita, sem API key).
    REQUISITOS DA SEC:
      - Header User-Agent com nome + e-mail (sem ele -> 403).
      - Respeitar ~10 req/s por IP (aqui usamos delay defensivo).
    Docs: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
    """

    name = "sec_edgar_13f"
    actor_class = ActorClass.WHALE
    BASE = "https://data.sec.gov"

    def __init__(self, contact_email: str, app_name: str = "ExemplarBot",
                 min_interval_s: float = 0.15):
        # A SEC pede formato "AppName email@dominio.com"
        self.headers = {"User-Agent": f"{app_name} {contact_email}"}
        self.min_interval_s = min_interval_s

    def _get(self, url: str) -> requests.Response:
        time.sleep(self.min_interval_s)  # rate-limit defensivo (<10 req/s)
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp

    def healthcheck(self) -> bool:
        try:
            self._get(f"{self.BASE}/submissions/CIK0001067983.json")  # Berkshire
            return True
        except requests.RequestException:
            return False

    def fetch(self, since: dt.date, ciks: list[str] | None = None) -> Iterable[Signal]:
        """
        Para cada gestor (CIK), lê submissions, filtra 13F-HR recentes e emite
        um Signal por holding. O diff trimestre-a-trimestre (novas posições /
        aumentos) fica a cargo do agregador downstream.
        """
        for cik in (ciks or []):
            cik10 = str(int(cik)).zfill(10)
            sub = self._get(f"{self.BASE}/submissions/CIK{cik10}.json").json()
            actor = sub.get("name", f"CIK{cik10}")
            recent = sub.get("filings", {}).get("recent", {})

            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accns = recent.get("accessionNumber", [])
            primary = recent.get("primaryDocument", [])

            for form, fdate, accn, doc in zip(forms, dates, accns, primary):
                if form != "13F-HR":
                    continue
                filed_at = dt.date.fromisoformat(fdate)
                if filed_at < since:
                    continue

                # URL do information table XML (parsing real omitido por brevidade):
                accn_nodash = accn.replace("-", "")
                base_dir = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}"
                # for holding in parse_13f_xml(self._get(f"{base_dir}/{doc}").text):
                #     yield Signal(...)

                yield Signal(
                    ticker="<parse_from_xml>",
                    side=Side.BUY,                 # 13F não tem "side"; derivar do diff QoQ
                    actor=actor,
                    actor_class=self.actor_class,
                    traded_at=None,                # 13F é snapshot, sem data de trade
                    filed_at=filed_at,
                    size_usd_low=None,
                    size_usd_high=None,
                    source=f"sec_edgar:{form}",
                    confidence=0.5,
                    raw={"accession": accn, "cik": cik10, "doc_url": f"{base_dir}/{doc}"},
                )


# --------------- Provider concreto: Capitol Trades (BFF não-oficial) ---------- #

class CapitolTradesProvider(SignalProvider):
    """
    Consome a API interna (BFF) do Capitol Trades como banco de dados de trades
    do Congresso. NÃO-OFICIAL: endpoint protegido por CloudFront, exige headers de
    browser, pode retornar 503/403 ou mudar sem aviso. Use cache + backoff e
    trate como best-effort (ver §2.1). Para produção com SLA, prefira Quiver/UW.
    """

    name = "capitol_trades"
    actor_class = ActorClass.CONGRESS
    BASE = "https://bff.capitoltrades.com"

    # txType do Capitol Trades -> Side normalizado
    _SIDE = {"buy": Side.BUY, "sell": Side.SELL}

    def __init__(self, min_interval_s: float = 1.0, max_pages: int = 10):
        self.min_interval_s = min_interval_s   # rate-limit defensivo (endpoint frágil)
        self.max_pages = max_pages             # teto de segurança de paginação
        self.headers = {
            # Use um User-Agent de browser real e atual:
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.capitoltrades.com/",
            "Origin": "https://www.capitoltrades.com",
        }

    def _get(self, page: int) -> dict:
        time.sleep(self.min_interval_s)
        resp = requests.get(
            f"{self.BASE}/trades",
            params={"page": page, "pageSize": 100, "sortBy": "-txDate"},
            headers=self.headers, timeout=30,
        )
        resp.raise_for_status()          # 503/403 -> RequestException (tratar no chamador)
        return resp.json()

    def healthcheck(self) -> bool:
        try:
            self._get(page=1)
            return True
        except requests.RequestException:
            return False                 # caiu? agregador segue com os outros providers

    def fetch(self, since: dt.date) -> Iterable[Signal]:
        """Emite Signals com filed_at (pubDate) >= since, paginando do mais recente."""
        for page in range(1, self.max_pages + 1):
            payload = self._get(page)
            rows = payload.get("data", [])
            if not rows:
                break

            stop = False
            for tx in rows:
                filed_at = _parse_date(tx.get("pubDate"))
                if filed_at is None:
                    continue
                if filed_at < since:     # como vem ordenado desc, podemos parar
                    stop = True
                    break

                side = self._SIDE.get((tx.get("txType") or "").lower())
                if side is None:         # ignora exchange/receive p/ sinal direcional
                    continue

                issuer = tx.get("issuer") or {}
                pol = tx.get("politician") or {}
                ticker = (issuer.get("issuerTicker") or "").upper().strip()
                if not ticker or ticker in {"--", "N/A"}:
                    continue             # opções/ativos sem ticker negociável -> pula

                yield Signal(
                    ticker=ticker,
                    side=side,
                    actor=pol.get("fullName")
                          or f"{pol.get('firstName','')} {pol.get('lastName','')}".strip()
                          or "unknown",
                    actor_class=self.actor_class,
                    traded_at=_parse_date(tx.get("txDate")),
                    filed_at=filed_at,
                    size_usd_low=tx.get("sizeRangeLow"),
                    size_usd_high=tx.get("sizeRangeHigh"),
                    source="capitol_trades:bff",
                    # owner != self (cônjuge/filho) costuma carregar menos sinal:
                    confidence=0.55 if (tx.get("owner") == "self") else 0.45,
                    raw=tx,
                )

            page_meta = payload.get("meta", {}).get("paging", {})
            if stop or page >= page_meta.get("totalPages", page):
                break


def _parse_date(s: str | None) -> dt.date | None:
    """Aceita 'YYYY-MM-DD' e ISO com 'Z' (ex.: 2026-01-15T00:00:00Z)."""
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _parse_amount_range(s: str | None) -> tuple[float | None, float | None]:
    """'$1,001 - $15,000' -> (1001.0, 15000.0). Aceita também valor único."""
    if not s:
        return (None, None)
    nums = re.findall(r"[\d,]+(?:\.\d+)?", s)
    vals = [float(n.replace(",", "")) for n in nums] or [None]
    return (vals[0], vals[-1] if len(vals) > 1 else vals[0])


# ---------- Provider concreto: Quiver Quantitative (PAGO, com SLA) ------------ #

class QuiverProvider(SignalProvider):
    """
    Fallback estável (pago) ao Capitol Trades. Usa /live/congresstrading
    (recentes de todos os membros). Docs: https://api.quiverquant.com/docs
    """

    name = "quiver_congress"
    actor_class = ActorClass.CONGRESS
    BASE = "https://api.quiverquant.com/beta"
    _SIDE = {"purchase": Side.BUY, "buy": Side.BUY, "sale": Side.SELL, "sell": Side.SELL}

    def __init__(self, api_key: str, min_interval_s: float = 0.3):
        # O pacote oficial usa "Token <key>"; se 401, troque por "Bearer <key>".
        self.headers = {"Authorization": f"Token {api_key}",
                        "Accept": "application/json"}
        self.min_interval_s = min_interval_s

    def _get(self, path: str) -> list[dict]:
        time.sleep(self.min_interval_s)
        resp = requests.get(f"{self.BASE}{path}", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def healthcheck(self) -> bool:
        try:
            self._get("/live/congresstrading")
            return True
        except requests.RequestException:
            return False

    def fetch(self, since: dt.date) -> Iterable[Signal]:
        for tx in self._get("/live/congresstrading"):
            filed_at = _parse_date(tx.get("ReportDate") or tx.get("Filed"))
            if filed_at is None or filed_at < since:
                continue
            side = self._SIDE.get((tx.get("Transaction") or "").strip().lower())
            if side is None:                       # ignora "Exchange" etc.
                continue
            ticker = (tx.get("Ticker") or "").upper().strip()
            if not ticker:
                continue
            low, high = _parse_amount_range(tx.get("Range"))
            yield Signal(
                ticker=ticker,
                side=side,
                actor=tx.get("Representative") or tx.get("Name") or "unknown",
                actor_class=self.actor_class,
                traded_at=_parse_date(tx.get("TransactionDate") or tx.get("Traded")),
                filed_at=filed_at,
                size_usd_low=low,
                size_usd_high=high,
                source="quiver:live/congresstrading",
                confidence=0.55,
                raw=tx,
            )


# --------- Provider concreto: Unusual Whales (PAGO, com SLA) ------------------ #

class UnusualWhalesProvider(SignalProvider):
    """
    Fallback estável (pago). Usa /api/congress/recent-trades.
    Inclui cônjuge/dependente. Docs: https://api.unusualwhales.com/docs
    """

    name = "unusual_whales_congress"
    actor_class = ActorClass.CONGRESS
    BASE = "https://api.unusualwhales.com"
    _SIDE = {"buy": Side.BUY, "purchase": Side.BUY,
             "sell": Side.SELL, "sale": Side.SELL}

    def __init__(self, api_key: str, limit: int = 200, min_interval_s: float = 0.3):
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Accept": "application/json"}
        self.limit = limit
        self.min_interval_s = min_interval_s

    def _get(self) -> list[dict]:
        time.sleep(self.min_interval_s)
        resp = requests.get(f"{self.BASE}/api/congress/recent-trades",
                            params={"limit": self.limit},
                            headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", [])         # resposta vem em {"data": [...]}

    def healthcheck(self) -> bool:
        try:
            self._get()
            return True
        except requests.RequestException:
            return False

    def fetch(self, since: dt.date) -> Iterable[Signal]:
        for tx in self._get():
            filed_at = _parse_date(tx.get("filed_at_date"))
            if filed_at is None or filed_at < since:
                continue
            side = self._SIDE.get((tx.get("txn_type") or "").strip().lower())
            if side is None:
                continue
            ticker = (tx.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            low, high = _parse_amount_range(tx.get("amounts"))
            yield Signal(
                ticker=ticker,
                side=side,
                actor=tx.get("name") or tx.get("reporter") or "unknown",
                actor_class=self.actor_class,
                traded_at=_parse_date(tx.get("transaction_date")),
                filed_at=filed_at,
                size_usd_low=low,
                size_usd_high=high,
                source="unusual_whales:recent-trades",
                confidence=0.55,
                raw=tx,
            )


# ------------------------- Agregador (consenso/ranking) ----------------------- #

def aggregate(providers: list[SignalProvider], since: dt.date) -> dict[str, float]:
    """
    Coleta de todos os providers, deduplica e produz um score por ticker.
    SAÍDA = score (não ordens). O Agente Planejador decide risco/sizing/execução.
    """
    seen: set[tuple] = set()
    scores: dict[str, float] = {}
    actors_per_ticker: dict[str, set] = {}

    for p in providers:
        for sig in p.fetch(since):
            if sig.dedup_key in seen:        # de-duplicação cross-source
                continue
            seen.add(sig.dedup_key)

            # decaimento por latência (sinal velho vale menos)
            lat = sig.latency_days or 45
            decay = max(0.1, 1.0 - lat / 90.0)
            direction = 1.0 if sig.side is Side.BUY else -1.0
            scores[sig.ticker] = scores.get(sig.ticker, 0.0) + direction * sig.confidence * decay
            actors_per_ticker.setdefault(sig.ticker, set()).add((sig.actor, sig.actor_class))

    # bônus de consenso: mais atores independentes (e classes distintas) -> sinal mais forte
    for ticker, actors in actors_per_ticker.items():
        n_actors = len(actors)
        n_classes = len({c for _, c in actors})
        scores[ticker] *= (1.0 + 0.1 * (n_actors - 1) + 0.15 * (n_classes - 1))

    return scores
```

> **Notas de implementação:**
> - `parse_13f_xml(...)` e o **diff trimestre-a-trimestre** (novas posições / aumentos) ficam de fora por
>   brevidade — é onde mora o trabalho real do 13F.
> - Para congressistas, um `QuiverCongressProvider` / `UnusualWhalesCongressProvider` implementa a mesma
>   interface `SignalProvider`, mapeando `traded_at`/`filed_at`/faixas para o `Signal` — assim o Planejador
>   **não sabe nem se importa** de onde veio o sinal.
> - `confidence` deve ser preenchido pelas **regras de ponderação por ator** (§5), não fixo em 0.5.

---

## Como virar skill

Para promover este documento a uma **skill** reutilizável do bot:

1. **Estrutura:** criar `skills/smart-money-signals/` com `SKILL.md` (este conteúdo condensado em
   instruções operacionais) + `providers/` (implementações concretas: `sec_edgar.py`, `quiver.py`,
   `unusual_whales.py`) + `tests/`.
2. **Frontmatter da skill:** manter `name: smart-money-signals` e uma `description` clara de **quando** o
   agente deve acionar (ex.: "quando o Planejador pedir sinais de smart money/copy trading de congresso ou 13F").
3. **Contrato:** expor `aggregate(providers, since) -> dict[ticker, score]` como **única** superfície pública;
   o Planejador consome **scores**, nunca ordens.
4. **Segredos/config:** API keys (Quiver/Unusual Whales) e `contact_email` (EDGAR) via env/secret manager;
   nunca hardcoded. Documentar **grátis vs pago** por provider.
5. **Guardrails embutidos na skill:** rate-limit do EDGAR, de-duplicação, decay por latência, exigência de
   `User-Agent`, e o disclaimer de que isto é **um sinal entre vários — nunca execução automática cega**.
6. **Testes/cache:** fixtures com payloads reais (1 PTR, 1 13F-HR, 1 Form 4) para testar o parser offline;
   cache local para respeitar limites das APIs e dar reprodutibilidade aos backtests.
7. **Backtest de validação:** antes de ligar em produção, validar o score contra o benchmark **NANC/GOP**
   para confirmar que há alfa **além** de beta de Big Tech.
