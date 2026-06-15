---
name: backtesting-testes
description: Referência completa de backtesting, paper trading e estratégia de testes (pytest, mocks da Alpaca, CI) para o bot de trading multi-agente em Python.
metadata:
  type: reference
---

# Backtesting, Paper Trading e Estratégia de Testes

> Documento de referência para o bot de trading exemplar (Alpaca, paper-first, multi-agente).
> Tudo em português; código e termos técnicos em inglês.
> Data da pesquisa: junho/2026. Sempre reconfira o status de manutenção dos frameworks antes de adotar (ver "Incertezas" no final).

---

## 0. TL;DR para este projeto

- **Backtest primeiro, paper depois, live por último.** Pular etapas é a forma número 1 de perder dinheiro.
- **Framework de backtest recomendado:** **VectorBT (Pro se houver budget; open-source se não)** para pesquisa/varredura de parâmetros + **Backtesting.py** para validação rápida de estratégias individuais. Para arquitetura *research-to-production* unificada (mesmo código no backtest e no live), avaliar **NautilusTrader**.
- **Testes:** `pytest` + `pytest-mock`/`responses` para **nunca** bater na API real da Alpaca em testes unitários.
- **CI:** `ruff` (lint+format) + `mypy` (type-check) + `pytest`, com secrets via GitHub Actions Secrets e secret-scanning (gitleaks/push protection).
- **Critério de live:** só liga dinheiro real depois de backtest realista + walk-forward + semanas em paper batendo com o esperado + risk controls e kill-switch testados.

---

## 1. Backtesting

### 1.1 O que é e por que é essencial

**Backtesting** é simular uma estratégia sobre dados históricos para estimar como ela teria performado. É a primeira linha de defesa: barato, rápido e sem risco de capital. Permite rejeitar estratégias ruins antes de gastar tempo de paper trading (que roda em tempo real) ou dinheiro de verdade.

Ordem canônica do pipeline:

```
ideia → backtest (in-sample) → walk-forward / out-of-sample → paper trading (semanas) → live com capital pequeno → scale gradual
```

O backtest **não prova** que a estratégia é lucrativa no futuro — ele só elimina as que claramente não funcionam e dá uma estimativa otimista (sempre otimista, por causa dos vieses da seção 2) do desempenho.

### 1.2 Frameworks Python — comparação (2025/2026)

| Framework | Modelo | Velocidade | Live trading | Manutenção (2025/26) | Melhor para |
|---|---|---|---|---|---|
| **VectorBT** (OSS) / **VectorBT Pro** | Vetorizado (NumPy/pandas/Numba) | Altíssima | Nenhum nativo (OSS); Pro tem mais recursos | **Ativo, porém novas features só no Pro (pago)** | Varredura massiva de parâmetros, portfolio analytics, pesquisa de alpha |
| **Backtesting.py** | Event-driven, single-asset | Boa | Não (research only, sem broker) | **Ativamente mantido** | Validar 1 estratégia rápido, prototipagem, ensino |
| **Backtrader** | Event-driven, rico em features | Lenta em escala (single-thread) | Sim (tem brokers, incl. integrações) | **Maturidade/estagnado** — última release 1.9.74.123 em mai/2019; só PRs de fix | Curva fácil idea→execução, comunidade/tutoriais grandes |
| **Zipline-reloaded** | Event-driven (fork do Quantopian) | Média | Mínimo | **Ativamente mantido** por Stefan Jansen (livro *ML for Algorithmic Trading*); Python 3.9+ | Pesquisa acadêmica, pipelines de fundamental data, ex-usuários Quantopian |
| **NautilusTrader** | Event-driven, core em Rust | Institucional (muito rápido) | **Sim — mesmo código backtest e live** | **Ativo e crescendo** | Produção institucional, fechar o gap research→production |

#### Notas por framework

- **VectorBT / VectorBT Pro** — o mais rápido para backtest vetorizado, ideal para testar milhares de combinações de parâmetros em segundos. Ponto de atenção: o desenvolvimento ativo de novas features migrou para a versão **Pro (paga)**; a versão open-source (`vectorbt`) ainda funciona mas recebe menos atenção. Sem execução live nativa — você precisa de um sistema de execução separado (que neste projeto é a camada Alpaca do bot).
- **Backtesting.py** — simples, documentação clara, mantido. Não tem suporte a live trading nem broker integration; é "research and testing, not execution". Excelente para o passo de validar cada estratégia isoladamente antes de plugar no orquestrador.
- **Backtrader** — event-driven, comunidade enorme, ótimos tutoriais e tem integração com brokers. **Mas a última release é de 2019**; só aceita PRs de correção. Funciona, é maduro, mas não espere features novas. Risco de longo prazo num projeto "exemplar".
- **Zipline-reloaded** — fork moderno e mantido do Zipline original do Quantopian; compatível com Python/pandas/numpy atuais. Forte em dados fundamentais e metodologia acadêmica, mas suporte a live trading é mínimo e o ritmo de desenvolvimento é mais lento.
- **NautilusTrader** — core em Rust, arquitetura unificada backtest↔live (o mesmo código de estratégia roda nos dois). É a melhor opção quando o objetivo é produção. Custo: curva de aprendizado maior, comunidade menor, setup mais complexo. "Most fund failures happen at the bridge between research and reality" — Nautilus existe para fechar essa ponte.

#### Recomendação para este projeto

Como o bot é **multi-agente, paper-first, e quer ser exemplar (research→production limpo)**:

1. **Pesquisa/otimização de parâmetros:** **VectorBT** (OSS para começar; Pro se a varredura de parâmetros virar gargalo).
2. **Validação de estratégia individual:** **Backtesting.py** — rápido de escrever um teste por estratégia.
3. **Se/quando quiser unificar backtest e live no mesmo código:** **NautilusTrader**. Vale a avaliação porque elimina a classe inteira de bugs "o backtest não bate com o live porque são duas implementações".
4. **Evitar depender de Backtrader** como base de longo prazo justamente pela estagnação — embora seja ótimo para aprender.

Fontes: [quantvps](https://www.quantvps.com/blog/best-python-backtesting-libraries-for-trading), [autotradelab](https://autotradelab.com/blog/backtrader-vs-nautilusttrader-vs-vectorbt-vs-zipline-reloaded), [qmr.ai](https://www.qmr.ai/best-backtesting-library-for-python/), [zipline-reloaded GitHub](https://github.com/stefan-jansen/zipline-reloaded), [python.financial 2026](https://python.financial/)

### 1.3 Exemplo de backtest simples (Backtesting.py)

SMA crossover, com **custo de transação realista** (`commission`) — nunca rode com custo zero.

```python
# backtest_sma_cross.py
# pip install backtesting
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def SMA(values: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    n_fast = 10
    n_slow = 30

    def init(self) -> None:
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, self.n_fast)
        self.sma_slow = self.I(SMA, close, self.n_slow)

    def next(self) -> None:
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
            self.sell()


# data: DataFrame com colunas Open, High, Low, Close, Volume e DatetimeIndex
# Use dados AJUSTADOS por split/dividendo para avaliar performance (ver seção 4).
data = pd.read_parquet("data/AAPL_1d_adjusted.parquet")

bt = Backtest(
    data,
    SmaCross,
    cash=100_000,
    commission=0.0005,   # 5 bps por trade — proxy de corretagem/taxas
    trade_on_close=False,  # evita look-ahead: executa na abertura do próximo bar
    exclusive_orders=True,
)

stats = bt.run()
print(stats)  # Return [%], Sharpe, Max. Drawdown [%], Win Rate [%], Profit Factor, etc.

# Otimização in-sample — CUIDADO com overfitting (seção 2).
# Sempre valide o melhor parâmetro OUT-OF-SAMPLE / walk-forward.
# stats = bt.optimize(n_fast=range(5, 20, 5), n_slow=range(20, 60, 10),
#                     maximize="Sharpe Ratio")
```

> Equivalente vetorizado em VectorBT seria `vbt.Portfolio.from_signals(...)` com `fees=0.0005` e `slippage=...`. Use VectorBT quando precisar varrer muitos parâmetros de uma vez.

---

## 2. Vieses e armadilhas de backtest

Estes são os erros que fazem um backtest "lindo" virar prejuízo no live. Trate cada um como um checklist.

- **Look-ahead bias** — usar informação que não estaria disponível no momento da decisão. Exemplos: executar no `Close` do mesmo bar que gerou o sinal; selecionar parâmetros/símbolos olhando o resultado do período inteiro; usar dados revisados (ex.: fundamentais restated). **Mitigação:** decida no bar `t`, execute no `open` de `t+1`; nunca toque em dados futuros; cuidado com indicadores que "vazam" (ex.: normalização com média do dataset inteiro).
- **Survivorship bias** — testar só nos ativos que sobreviveram, ignorando os que foram delistados/faliram. Infla retorno em ~1–4% a.a. e melhora artificialmente Sharpe e drawdown. **Mitigação:** usar datasets *survivorship-bias-free* (que incluem delisted) ou ao menos reconhecer a inflação esperada.
- **Overfitting / curve-fitting** — otimizar tanto os parâmetros que a estratégia decora o ruído do passado e quebra no futuro. Sinal de alerta: performance espetacular in-sample que evapora out-of-sample. **Mitigação:** poucos parâmetros; faixas amplas e robustas (plateau, não pico); penalizar complexidade.
- **Data snooping / multiple testing** — testar centenas de variantes e ficar com a "melhor". Quanto mais testes, maior a chance de achar algo bom por puro acaso. **Mitigação:** corrigir para múltiplas comparações (ex.: Deflated Sharpe Ratio, Probability of Backtest Overfitting — PBO).
- **Custos de transação e slippage irrealistas** — backtest com custo zero é ficção. Inclua corretagem/taxas, spread bid-ask e slippage (preço pior que o teórico, especialmente em ordens grandes ou ativos ilíquidos). **Mitigação:** sempre setar `commission`/`fees` e `slippage`; estressar com custos maiores e ver se a edge sobrevive.
- **In-sample (IS) vs out-of-sample (OOS)** — separe os dados: ajuste/otimize no IS, valide no OOS que o modelo nunca viu. Se cair muito do IS para o OOS → overfitting.
- **Walk-forward analysis** — "gold standard" (Pardo): re-otimiza em janelas rolantes (otimiza na janela, testa na próxima, anda, repete). Obriga a estratégia a provar-se repetidamente em regimes diferentes. **Alternativa mais robusta contra false discovery:** Combinatorial Purged Cross-Validation (CPCV), que costuma ter PBO menor e melhor Deflated Sharpe que walk-forward simples.

Fontes: [luxalgo survivorship](https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/), [fortraders bias](https://www.fortraders.com/blog/how-to-avoid-bias-in-backtesting), [analystprep CFA](https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/), [arxiv walk-forward 2025](https://arxiv.org/html/2512.12924v1), [ScienceDirect overfitting OOS](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)

---

## 3. Métricas de performance

Leia as métricas **em conjunto** — nenhuma isolada conta a história inteira.

| Métrica | O que mede | Como calcular | Interpretação |
|---|---|---|---|
| **Total Return** | Ganho total no período | `(equity_final / equity_inicial) - 1` | Bruto; não comparável entre períodos de tamanhos diferentes |
| **CAGR** | Crescimento anualizado composto | `(equity_final / equity_inicial)^(1/anos) - 1` | Permite comparar estratégias de durações diferentes |
| **Sharpe Ratio** | Retorno por unidade de **volatilidade total** | `(retorno - risk_free) / desvio_padrão_retornos`, anualizado | A mais usada. > 1 ok, > 2 forte. Penaliza volatilidade boa e ruim igualmente |
| **Sortino Ratio** | Retorno por unidade de **risco de queda** | `(retorno - risk_free) / desvio_padrão_dos_retornos_negativos` | Só pune downside. > 2 preferível. Mais justo para estratégias assimétricas |
| **Max Drawdown (MDD)** | Maior queda pico→vale do equity | `min((equity_t - peak_t) / peak_t)` | Quanto você teria sofrido no pior momento. < 15% = boa preservação de capital |
| **Win Rate** | % de trades vencedores | `n_trades_lucro / n_trades_total` | Sozinho engana: win rate alto com perdas grandes pode perder dinheiro |
| **Profit Factor** | Lucro bruto ÷ perda bruta | `soma_ganhos / abs(soma_perdas)` | > 1 lucrativo; > 1.75 forte |
| **Exposure** | % do tempo com posição aberta | `tempo_em_mercado / tempo_total` | Mede uso de capital e risco de exposição a mercado |

Cálculo de Sharpe/Sortino anualizados (a partir de retornos diários):

```python
import numpy as np

def sharpe(returns: np.ndarray, rf_daily: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf_daily
    return np.sqrt(periods) * excess.mean() / excess.std(ddof=1)

def sortino(returns: np.ndarray, rf_daily: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf_daily
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    return np.sqrt(periods) * excess.mean() / dd if dd > 0 else np.nan

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return (equity / peak - 1.0).min()
```

> Atenção: um Sharpe alto obtido por data snooping é ilusório — confira com **Deflated Sharpe Ratio** quando você testou muitas variantes.

Fontes: [QuantifiedStrategies métricas](https://www.quantifiedstrategies.com/trading-performance/), [optionalpha](https://optionalpha.com/learn/performance-metrics), [luxalgo top metrics](https://www.luxalgo.com/blog/top-7-metrics-for-backtesting-results/), [QuantifiedStrategies profit factor](https://www.quantifiedstrategies.com/profit-factor/)

---

## 4. Dados históricos

A qualidade do backtest está limitada pela qualidade dos dados ("garbage in, garbage out").

### Fontes

- **Alpaca Historical Data (`alpaca-py`)** — fonte primária natural, pois é o broker do bot. Bars/quotes/trades. Atenção aos *feeds*: contas elegíveis acessam **IEX** (gratuito, cobertura parcial); **SIP** (consolidado, cobertura completa) exige assinatura **Algo Trader Plus**. Dados com mais de 15 min estão disponíveis em todos os feeds.
- **yfinance** — gratuito, fácil, cobertura ampla de equities; bom para protótipo e validação cruzada. Qualidade variável, sem garantias de SLA, sujeito a quebras quando o Yahoo muda a API. Não usar como fonte de produção crítica.
- **Outras** (para evoluir): Polygon.io, Databento, Tiingo, Norgate (survivorship-bias-free).

### Ajuste de splits e dividendos

- No `alpaca-py`, parâmetro `adjustment`: `"raw"`/`None`, `"split"`, `"dividend"`, `"all"`.
- **Regra:** use dados **ajustados** (`all`) para **avaliar performance histórica**; use **raw** (não ajustado) quando a decisão é sobre o preço corrente do bar atual (para não introduzir look-ahead pelos ajustes retroativos).
- **Armadilha conhecida:** já houve relatos no fórum da Alpaca de o parâmetro de ajuste **não** aplicar o split corretamente (raw/split/all retornando o mesmo). **Sempre valide** o ajuste manualmente em um símbolo que sofreu split conhecido antes de confiar.

### Granularidade

Escolha a granularidade compatível com a estratégia: daily bars para swing; minute bars para intraday; tick/quote para microestrutura. Quanto mais fina, mais dados, mais custo e mais sensível a slippage/latência.

Fontes: [Alpaca fetch historical](https://alpaca.markets/learn/fetch-historical-data), [alpaca-py GitHub](https://github.com/alpacahq/alpaca-py), [Alpaca stockbars ref](https://docs.alpaca.markets/us/reference/stockbars), [fórum: ajuste não funciona](https://forum.alpaca.markets/t/v2-api-bars-request-adjustment-not-working/12893)

---

## 5. Paper trading como etapa intermediária

Backtest é otimista por construção; paper trading expõe a realidade do mundo em tempo real **sem arriscar capital**. É a ponte entre teoria e dinheiro de verdade.

### Por que rodar semanas em paper antes do live

- Valida **infraestrutura**: API timeouts, reconexões, agendamento, ordens parciais, comportamento do orquestrador multi-agente sob mercado real.
- Valida **comportamento real vs backtest**: o paper roda em tempo real, então você vê se o sinal/execução batem com a simulação.
- Pega bugs que o backtest nunca pega: race conditions, ordens duplicadas, estados inconsistentes, falhas de logging.

### O que validar em paper

- **Latência** de submissão e de fill.
- **Fills**: preço, quantidade, parciais.
- **Comportamento vs backtest**: equity curve do paper deve ser razoavelmente próxima do esperado; divergência grande = bug ou premissa irreal no backtest.
- **Risk controls e kill-switch** funcionando de verdade.

### Diferenças entre paper e live na Alpaca (importante)

Em paper trading, as ordens **não** são roteadas para exchanges reais — o sistema **simula** o preenchimento com base em quotes em tempo real. Consequências:

- **Sem checagem de liquidez real**: a quantidade da ordem **não** é validada contra a quantidade no NBBO. Você pode receber fill de uma ordem muito maior do que a liquidez real disponível.
- **Fills parciais artificiais**: ordens elegíveis recebem partial fill ~10% das vezes (simulação, não realidade de mercado).
- **Sem slippage real, sem erros de teclado, sem problemas de plataforma**: o paper é "limpo demais".
- **Latência diferente**: relatos da comunidade indicam latência **muito maior** no paper do que no live para as mesmas ordens — ou seja, latência de paper **não** é proxy confiável de latência live.

**Conclusão:** paper valida lógica, integração e robustez de software — mas **não** valida slippage/liquidez/latência reais. Por isso o live começa com capital pequeno.

Fontes: [Alpaca paper vs live (data-backed)](https://alpaca.markets/learn/paper-trading-vs-live-trading-a-data-backed-guide-on-when-to-start-trading-real-money), [Alpaca docs paper trading](https://docs.alpaca.markets/us/docs/paper-trading), [fórum: latência paper vs live](https://forum.alpaca.markets/t/massive-paper-trading-latency-vs-live-trading/9053), [Alpaca support](https://alpaca.markets/support/difference-paper-live-trading)

---

## 6. Testes de software

Regra de ouro: **testes unitários NUNCA batem na API real da Alpaca.** Mock sempre. A API real é lenta, não-determinística, com rate limits, e exige credenciais — tudo péssimo para CI.

### Estrutura de `tests/`

```
tests/
├── conftest.py              # fixtures compartilhadas (clients mockados, dados de mercado)
├── fixtures/
│   ├── bars_aapl_1d.parquet # dados de mercado canônicos para testes determinísticos
│   └── orders.json          # respostas de exemplo da API
├── unit/
│   ├── test_strategy_sma.py     # cada estratégia testada ISOLADAMENTE
│   ├── test_strategy_meanrev.py
│   ├── test_risk_manager.py
│   └── test_position_sizing.py
├── integration/
│   ├── test_order_flow.py       # orquestrador → broker (mockado), end-to-end lógico
│   └── test_data_pipeline.py
└── backtest/
    └── test_backtest_regression.py  # equity/métricas conhecidas não regridem
```

### Ferramentas

- `pytest` — runner.
- `pytest-mock` (`mocker`) — wrapper do `unittest.mock`; faz patch escopado e reverte automaticamente após cada teste.
- `responses` ou `requests-mock` — interceptam chamadas HTTP (o `alpaca-trade-api-python` oficial usa `requests_mock` nos próprios testes).
- `freezegun` — congela o relógio para testes determinísticos de lógica dependente de tempo.

### Princípios

- **Cada estratégia isolada:** dado um DataFrame fixo de barras, a estratégia deve produzir exatamente os sinais esperados. Determinístico, sem rede.
- **Testes de integração:** orquestrador + broker **mockado** — valida o fluxo (sinal → ordem → atualização de posição) sem tocar a Alpaca.
- **Fixtures de dados de mercado:** guarde um pequeno dataset canônico (parquet/csv) e respostas JSON de exemplo. Reuse via `conftest.py`.
- **Regression test do backtest:** fixe semente/dados e asserte que métricas-chave (Sharpe, retorno, nº de trades) não mudam inesperadamente.

### Exemplo: teste pytest com mock da Alpaca

Mockando o cliente `alpaca-py` (`TradingClient.submit_order`) — sem rede:

```python
# tests/unit/test_order_execution.py
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Código sob teste (exemplo): bot/execution.py
# from bot.execution import OrderExecutor


@pytest.fixture
def fake_trading_client(mocker):
    """Cliente Alpaca totalmente mockado — nunca toca a rede."""
    client = mocker.MagicMock(name="TradingClient")
    # simula a resposta de submit_order
    fake_order = MagicMock()
    fake_order.id = "test-order-id-123"
    fake_order.status = "accepted"
    fake_order.symbol = "AAPL"
    fake_order.qty = "10"
    client.submit_order.return_value = fake_order
    return client


def test_executor_submits_market_buy(fake_trading_client):
    from bot.execution import OrderExecutor  # import local p/ facilitar o patch

    executor = OrderExecutor(client=fake_trading_client)
    result = executor.buy(symbol="AAPL", qty=10)

    # 1) chamou a API exatamente uma vez
    fake_trading_client.submit_order.assert_called_once()

    # 2) montou os parâmetros certos (inspeciona o request enviado)
    sent = fake_trading_client.submit_order.call_args.args[0]
    assert sent.symbol == "AAPL"
    assert sent.qty == 10
    assert sent.side.value == "buy"

    # 3) propagou o resultado
    assert result.id == "test-order-id-123"
    assert result.status == "accepted"


def test_executor_rejects_oversized_order(fake_trading_client):
    """Risk control deve barrar ANTES de chamar a Alpaca."""
    from bot.execution import OrderExecutor

    executor = OrderExecutor(client=fake_trading_client, max_qty=100)
    with pytest.raises(ValueError, match="exceeds max_qty"):
        executor.buy(symbol="AAPL", qty=1_000)

    # não pode ter chamado a API
    fake_trading_client.submit_order.assert_not_called()
```

Alternativa em nível HTTP com `responses` (intercepta o `requests` por baixo do SDK):

```python
import responses

@responses.activate
def test_get_account_http_mock():
    responses.add(
        responses.GET,
        "https://paper-api.alpaca.markets/v2/account",
        json={"id": "acc-1", "cash": "100000", "status": "ACTIVE"},
        status=200,
    )
    # ... instanciar client apontando para a URL acima e chamar get_account()
    # assert client.get_account().cash == "100000"
```

Fontes: [alpaca-trade-api tests (requests_mock)](https://github.com/alpacahq/alpaca-trade-api-python/blob/master/tests/test_rest.py), [pytest-mock PyPI](https://pypi.org/project/pytest-mock/), [requests-mock pytest](https://requests-mock.readthedocs.io/en/latest/pytest.html), [responses + pytest (2025)](https://oneuptime.com/blog/post/2025-01-06-python-mock-external-apis/view), [pytest external APIs](https://pytest-with-eric.com/api-testing/pytest-external-api-testing/)

---

## 7. CI e segurança

### Não vazar chaves

- **Nunca** comitar chaves. Use `.env` local (no `.gitignore`) e `os.environ`/`pydantic-settings` para ler.
- No CI, use **GitHub Actions Secrets**: criptografados em repouso, mascarados nos logs, não expostos a forks. Variáveis de config (não-sensíveis, ex.: região) podem ser plaintext.
- Habilite **secret scanning**: GitHub **push protection** bloqueia commits com segredos; **gitleaks**/**TruffleHog** rodando no CI detectam credenciais vazadas antes do merge.
- Para escala, migrar para um secrets manager (HashiCorp Vault, AWS/Azure Secrets Manager) com rotação e auditoria.
- Os testes unitários **não precisam de chaves** porque tudo é mockado — isso é também uma defesa de segurança.

### Testes determinísticos

- Sem rede, sem relógio real (`freezegun`), sem aleatoriedade não-semeada (`np.random.seed`/`random.seed`).
- Dados de teste fixos (fixtures versionadas).
- Sem dependência de ordem entre testes.

### Lint e type-check

- **`ruff`** — linter + formatter rapidíssimo (substitui flake8/isort/black).
- **`mypy`** — type-checking estático; pegue erros de tipo antes do runtime (crítico num bot que move dinheiro).

### Exemplo de workflow GitHub Actions

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"   # ruff, mypy, pytest, pytest-mock, responses
      - name: Lint
        run: ruff check .
      - name: Format check
        run: ruff format --check .
      - name: Type check
        run: mypy bot/
      - name: Tests (offline, mocked)
        run: pytest -q --cov=bot
        # NÃO injetar ALPACA_API_KEY aqui: testes unitários são mockados.

  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: gitleaks/gitleaks-action@v2
```

Fontes: [GitHub Actions secrets vs env](https://env.dev/guides/github-actions-secrets-env), [secrets/API keys .env (KDnuggets)](https://www.kdnuggets.com/managing-secrets-and-api-keys-in-python-projects-env-guide), [lint+test com GitHub Actions](https://medium.com/django-unleashed/3-lint-and-test-with-github-actions-efa80197b303), [ruff docs](https://docs.astral.sh/ruff/settings/), [mypy action](https://github.com/marketplace/actions/mypy-action)

---

## 8. Checklist "pronto para live"

Só ligue dinheiro real quando **todos** os itens estiverem marcados:

**Estratégia**
- [ ] Backtest com premissas **realistas** (custos, slippage, dados ajustados).
- [ ] Validação **out-of-sample** e/ou **walk-forward** com queda aceitável vs in-sample.
- [ ] Testada em **mais de um regime de mercado** (alta, baixa, lateral, alta vol).
- [ ] Poucos parâmetros, em **plateau** robusto (não pico) — sem cara de overfitting.
- [ ] Métricas lidas **em conjunto** (Sharpe + Sortino + MDD + profit factor + exposure).

**Paper trading**
- [ ] **Semanas** em paper, não dias.
- [ ] Equity curve do paper **bate razoavelmente** com o esperado do backtest.
- [ ] Fills, ordens parciais e timeouts observados e tratados.
- [ ] Reconexão/erros de API testados em mercado real.

**Software & risco**
- [ ] `pytest` verde, cobertura razoável, tudo mockado e determinístico.
- [ ] `ruff` e `mypy` sem erros no CI.
- [ ] **Position sizing**, **risk limits** (perda máxima diária/por trade) implementados e testados.
- [ ] **Kill-switch / emergency shutdown** testado de verdade.
- [ ] **Logging** de toda ordem, erro e evento.
- [ ] Alertas/monitoramento configurados ("don't set and forget").
- [ ] Secrets fora do código, secret-scanning ativo.

**Rollout**
- [ ] Começar com **capital pequeno**.
- [ ] **Escalar gradualmente**, nunca de paper para tamanho cheio de uma vez.

Fontes: [Alpaca quando começar live](https://alpaca.markets/learn/paper-trading-vs-live-trading-a-data-backed-guide-on-when-to-start-trading-real-money), [luxalgo bot step-by-step](https://www.luxalgo.com/blog/building-your-first-trading-bot-step-by-step-guide/), [forex.com algo bot](https://www.forex.com/en-us/trading-guides/how-to-build-a-trading-algo-bot-for-stock-trading/)

---

## 9. Erros comuns

- **Backtest com custo zero / slippage zero** → edge ilusória que some no live.
- **Look-ahead silencioso** → executar no `Close` do bar do sinal, ou normalizar com estatística do dataset inteiro.
- **Overfitting por otimização agressiva** → pico de parâmetro decorado, quebra fora da amostra.
- **Data snooping** → testar 200 variantes e ficar com a melhor sem correção estatística.
- **Survivorship bias** → universo só com sobreviventes infla retorno em 1–4% a.a.
- **Confiar na latência/liquidez do paper da Alpaca** → paper não roteia para exchange, não checa liquidez, latência diferente do live.
- **Pular o paper trading** → bugs de integração só aparecem em tempo real.
- **Testes unitários batendo na API real** → lentos, flaky, exigem chaves, podem disparar ordens.
- **Dados não ajustados (ou ajuste quebrado) na avaliação de performance** → splits/dividendos distorcem retorno; sempre valide o ajuste num símbolo conhecido.
- **Vazar chave no repo** → sem push protection / secret scanning, credencial vira pública.
- **Set and forget no live** → sem monitoramento/kill-switch, um bug pode sangrar a conta.
- **Adotar framework estagnado como base de longo prazo** (ex.: Backtrader sem release desde 2019) sem ter consciência do trade-off.

---

## Como virar skill

Para transformar este documento em uma **skill** reutilizável do Claude Code:

1. **Criar a pasta da skill:** `.claude/skills/backtesting-testes/` (ou onde o projeto agrega skills).
2. **`SKILL.md` com frontmatter de skill** (não de reference). Ex.:
   ```yaml
   ---
   name: backtesting-testes
   description: >
     Use ao escrever/revisar backtests, paper trading ou testes do bot de trading.
     Dispara em pedidos como "escreve um backtest", "monta os testes da estratégia",
     "mocka a Alpaca no pytest", "está pronto para live?", "revisa vieses do backtest".
   ---
   ```
3. **Corpo enxuto, orientado a ação:** transformar as seções em *procedimentos* ("quando o usuário pedir um backtest, faça X usando Backtesting.py com `commission` setado; valide out-of-sample…"), em vez de prosa de referência. Manter o checklist da seção 8 como gate obrigatório antes de sugerir live.
4. **Empacotar os exemplos de código** (`backtest_sma_cross.py`, `test_order_execution.py`, `ci.yml`) como `assets/` ou snippets que a skill cola e adapta.
5. **Referenciar este doc** como material de fundo (`reference/06-backtesting-testes.md`) para o Claude consultar detalhes/fontes sem inflar o `SKILL.md`.
6. **Triggers chave a cobrir:** "backtest", "walk-forward", "overfitting", "Sharpe/Sortino", "mock Alpaca", "paper trading", "pronto para live", "pytest do bot".
7. **Revalidar o status dos frameworks** (seção "Incertezas") na primeira vez que a skill rodar num projeto novo — datas de manutenção mudam.
