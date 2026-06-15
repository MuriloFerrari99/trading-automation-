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
| **Capitol Trades** | Agregador | Sim (web) | **Não há API pública oficial documentada** | Web (HTML) | Excelente UI; consumo programático = scraping ou wrappers de terceiros |
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
- Agregador gratuito com ótima UI (filtros por político, ticker, comitê).
- **Sem API pública oficial documentada.** Acesso programático = scraping do HTML ou wrappers
  não-oficiais de terceiros (ex.: pacotes comunitários, scrapers do tipo Lambda Finance). Use com cautela:
  pode quebrar e violar ToS.

**Quiver Quantitative** — `https://www.quiverquant.com/congresstrading/`
- **API documentada e oficial:** `https://api.quiverquant.com/`
  (dataset: `https://api.quiverquant.com/datasets/congress-trades`).
- **Tier grátis** existe, porém **com dados atrasados e profundidade histórica limitada**; **API completa
  é paga** (a partir de ~US$ 25–30/mês).
- **Formato:** JSON estruturado com nome do membro, partido, ticker, tipo de transação e **faixa de valor**.
- Endpoints separados para **House**, **Senate**, além de insider, lobbying, contratos governamentais,
  net worth de políticos etc. Autenticação por **Bearer token**.

**Unusual Whales** — `https://unusualwhales.com/`
- **API documentada:** `https://api.unusualwhales.com/docs`
  (OpenAPI YAML: `https://api.unusualwhales.com/api/openapi`; dev portal: `https://unusualwhales.com/developers`).
- Mesmo núcleo de dados do STOCK Act (House + Senate), **incluindo transações de cônjuge/dependente** e
  **visões de portfólio por político**.
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
