# Sistema de Automação de Investimentos (Alpaca, paper)

Bot de trading automatizado em Python, orquestrado por agentes (Planejador → Decisão → Executor → Feedback) e executando ordens via **API da Alpaca**. Inclui camada de risco fail-safe, loop de aprendizado, pipeline de ML (P(win)) e um harness de backtesting/validação estatística.

> **Somente paper trading.** Um *guard* no código aborta qualquer tentativa de operar com dinheiro real (ver [Segurança](#segurança)). 247 testes (`pytest`), `ruff` limpo. Conexão paper validada e execução end-to-end exercitada contra a API real.

Para o charter do projeto, restrições e roadmap, ver [BRIEFING.md](BRIEFING.md). Para a base de pesquisa que fundamenta as decisões técnicas, ver [docs/README.md](docs/README.md).

---

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.11+.

```bash
uv sync                                   # instala deps a partir do uv.lock
cp .env.example .env                      # preencha ALPACA_API_KEY / ALPACA_SECRET_KEY

uv run python main.py --check             # valida conexão/conta e sai
uv run python main.py --once              # roda um único ciclo (diagnóstico)
uv run python main.py                     # inicia o Monitor (loop agendado)
```

Outros entrypoints:

```bash
uv run python -m dashboard                # UI web de monitoramento (stdlib, porta 8787)
uv run python -m simulation.run           # sweep de backtests sobre dados reais
uv run python -m simulation.experiment_oos # experimento OOS em escala (desde 2020)
uv run python -m simulation.verdict       # tribunal estatístico de edge (PSR/DSR/PBO)
uv run python -m ml.retraining            # retreino/validação/promoção do modelo de ML

uv run pytest                             # 247 testes (offline, com FakeBroker)
uv run ruff check .
```

### Configuração

- **`.env`** (nunca versionado) — `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_ENDPOINT` (fixo no paper), `LIVE_TRADING=false`, `ORCHESTRATOR=bus`, e chaves opcionais de provedores de sinais (`QUIVER_API_KEY`, `UNUSUAL_WHALES_API_KEY`).
- **`config/watchlist.yaml`** — ativos monitorados e parâmetros por ativo (trailing %, degraus do ladder, config da wheel, classe do ativo p/ cripto 24/7). Wheel e cripto são *opt-in* e vêm comentados por padrão.

---

## Segurança (restrições inegociáveis)

1. **Paper-only.** `config/settings.py` levanta `LiveTradingBlockedError` se `LIVE_TRADING=true`; o endpoint é fixado no paper e `AlpacaBroker` instancia com `paper=True`.
2. **Segredos só em `.env`** (carregado via `python-dotenv`); `.env.example` versionado, `.env` no `.gitignore`.
3. **Kill switch central** (`core/kill_switch.py`) — variável de ambiente *ou* arquivo sentinela; rechecado imediatamente antes de cada ida à rede.
4. **Auditoria completa** — todo trade é logado em SQLite (timestamp, ativo, lado, qtd, preço, estratégia) e toda decisão/evento passa por um `audit_log` com ator.

---

## Arquitetura

O sistema é um **pipeline de agentes** sobre um message bus in-memory. Cada agente consome de um tópico, processa e publica — sem conhecer os demais. A orquestração fica atrás da interface `AgentOrchestrator`, com três implementações selecionáveis por `ORCHESTRATOR` (`bus` é o default).

```
Scheduler
  → IngestionAgent      (busca barras reais)            → market_data
  → PlannerAgent        (estratégias + RiskManager)      → intents
  → DecisionAgent       (regime + policy + track-record) → approved
  → ExecutorAgent       (kill-switch → idempotência →    → fills
                         persiste → submete c/ retry)
  → FeedbackAgent       (anexa Outcome win/loss à decisão)
  ReconcileAgent         (periódico; broker = verdade)
```

### Camadas de inteligência

| Camada | Pacote | O que faz |
|--------|--------|-----------|
| 0 — Feedback | [`feedback/`](feedback/) | Registra decisão + contexto + resultado; classifica regime (rule-based); avalia edge por estratégia×regime |
| 0.5 — Decisão | [`intelligence/`](intelligence/) | Gate que veta combos estratégia×regime com expectancy negativa comprovada; calcula convicção |
| 1 — Estratégias + Sizing | [`strategies/`](strategies/), [`sizing/`](sizing/) | Trailing stop, ladder buys, wheel; sizing dinâmico por convicção (Kelly fracionário) |
| 1.5 — Síntese | [`synthesis/`](synthesis/) | Combina visões ortogonais (técnico/smart money/ML/regime) em convicção + consenso/conflito + racional |
| 2 — ML | [`ml/`](ml/) | Classificador P(win) (logística numpy) com walk-forward, calibração e promoção champion/challenger automática |
| Integração | [`integration/`](integration/) | `DecisionEnricher` compõe FIMATHE + síntese + ML + sizing num único ponto |

### Pacotes

| Pacote | Responsabilidade |
|--------|------------------|
| [`agents/`](agents/) | Agentes do bus: `ingestion`, `planner`, `executor`, `monitor`, `feedback_agent`, `reconcile_agent`, `bus_agents`, `base` |
| [`broker/`](broker/) | Abstração `BrokerClient`; `AlpacaBroker` (paper) e `FakeBroker` (testes/sim) |
| [`config/`](config/) | `settings` (guard paper-only), `watchlist`, `risk` |
| [`core/`](core/) | Modelos Pydantic v2, `idempotency` (client_order_id determinístico), `kill_switch`, `market_clock`, `rounding` |
| [`data/`](data/) | SQLite: `db` (schema), repos de `order`/`position`/`signal`/`state`, `trade_logger`, `audit_log` |
| [`risk/`](risk/) | `RiskManager` + `PortfolioRiskGuard` (circuit breakers) + `sizing` fixed-fractional |
| [`orchestration/`](orchestration/) | `AgentOrchestrator`, `InMemoryBus`, orquestradores `local`/`bus`/`opensquad`, `factory`, `reconcile` |
| [`scheduling/`](scheduling/) | APScheduler em UTC (DST-safe) para os ticks do Monitor |
| [`fimathe/`](fimathe/) | `FimatheEngine` — canais/ZN/Fibonacci/RSI/ATR/ADX vetorizados sobre OHLC (feature table p/ ML) |
| [`simulation/`](simulation/) | Backtesting honesto (sem look-ahead), Monte Carlo, experimentos OOS, custos, tribunal estatístico |
| [`dashboard/`](dashboard/) | UI web read-only (stdlib `http.server`) — KPIs, posições, trades, curva de equity |

---

## Estratégias

- **Trailing Stop** ([`strategies/trailing_stop.py`](strategies/trailing_stop.py)) — stop nativo da Alpaca (`trail_percent`) que sobe com o preço e sobrevive a restart; protege qualquer long sem proteção.
- **Ladder Buys** ([`strategies/ladder_buys.py`](strategies/ladder_buys.py)) — compras escalonadas em quedas para reduzir preço médio; cada degrau dispara uma vez.
- **Wheel / opções** ([`strategies/wheel.py`](strategies/wheel.py)) — venda de CSP → covered call (Nível 3 de opções, *opt-in*); gate verifica nível de opções e liquidez em runtime.
- **Sinais de Smart Money** ([`strategies/signals/`](strategies/signals/)) — trades do Congresso (STOCK Act) e movimentação de fundos (13F) atrás de `SignalProvider`. **Apenas sugerem** — nunca geram ordem automaticamente.

---

## Persistência (SQLite, `data/trading.sqlite`)

Tabelas: `trade_log` (auditoria de execuções), `orders` (ciclo de vida PENDING→SUBMITTED→FILLED/PARTIAL), `positions` (snapshot reconciliado), `state` (chave/valor entre ticks), `signals` (sugestões para revisão), `decisions` (decisão + outcome — base do loop de feedback e do ML), `audit_log` (trilha imutável com ator). Modelos treinados de ML em `data/models/`, cache de barras em `data/cache/`.

---

## Status e roadmap

**Veredito honesto** (benchmark estrutural 2026-06-15): chassi de nível institucional — idempotência, reconciliação (broker = fonte de verdade), risco fail-safe, 247 testes, validação OOS anti-overfit — com **motor de alpha ainda fraco**. Os 66k backtests OOS desde 2020 mostram que o ganho *provado* vem da **camada de risco** (pior drawdown −100% → −21.5%) e o **gate de decisão** agrega em regime adverso; o alpha bruto das estratégias é modesto.

Roadmap para elevar a média estrutural a ~9/10:

- **F1 — Ligar o ML + validação honesta** *(em andamento)*: o pipeline de retreino/walk-forward/calibração/promoção está pronto ([`ml/`](ml/)); falta o **cutover** — carregar o modelo promovido no boot e wirar o `DecisionEnricher` no caminho de decisão vivo, e agendar o job de retreino.
- **F3 — Observabilidade + risco fino + validar opções/cripto no paper ao vivo.**
- **F2 — Alpha de verdade**: dados alternativos, feature engineering, model zoo, modelos por regime.
- **F4 — Runtime concorrente**: asyncio, WebSocket streaming, bus durável (Redis Streams), DLQ + supervisão.
- **F5 — LLM no caminho de decisão** (advisory, nunca gatekeeper sozinho).
- **F6 — Prontidão produção**: `LiveReadinessGate`, soak test, canary. A flag `LIVE_TRADING` segue **bloqueada**.

---

## Base de conhecimento

[`docs/research/`](docs/README.md) reúne 14 documentos de referência (Alpaca API, risco/execução, smart money, Wheel/opções, arquitetura multi-agente, backtesting, FIMATHE, e as camadas de feedback/decisão/ML/síntese/integração já implementadas), com fontes citadas e padrões de código.
