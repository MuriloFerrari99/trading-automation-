---
name: arquitetura-agentes
description: Arquitetura e orquestração de um trading bot multi-agente (Planejador/Executor/Monitor) em Python sobre Alpaca, com camada de orquestração desacoplada plugável (local agora, OpenSquad depois).
metadata:
  type: reference
---

# Arquitetura e Orquestração de um Sistema de Trading Multi-Agente

Documento de referência para o bot de trading em Python (Alpaca, paper trading primeiro). O objetivo é ter um sistema **exemplar**: responsabilidades separadas, comunicação clara entre agentes, orquestração desacoplada por uma interface `AgentOrchestrator`, agendamento ciente de horário de mercado e estado persistido com recuperação após crash.

Princípio que atravessa o documento inteiro: **o broker (Alpaca) é a fonte de verdade**. O bot mantém estado local por desempenho e auditoria, mas **reconcilia contra o broker** ao iniciar e periodicamente — nunca confia cegamente no próprio estado. ([mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure), [Medium / James Hall](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c))

---

## 1. Arquitetura geral de um trading bot de produção

A arquitetura é uma **pipeline em camadas**, cada uma com uma única responsabilidade. Isso é o que NautilusTrader, sistemas quant institucionais e bots de produção convergem a fazer: separar ingestão, sinal, decisão/risco, execução, monitoramento e persistência, com fronteiras explícitas entre elas. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/architecture/), [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure))

### Camadas e responsabilidades

1. **Ingestão de dados (Data ingestion)** — Busca preços (barras, quotes, trades) e dados de conta da Alpaca; normaliza para um formato interno. Não toma decisões. Pode rodar por *polling* (REST) no MVP ou *streaming* (WebSocket) depois.
2. **Geração de sinal (Signal / Strategy)** — Recebe dados normalizados e produz sinais ("comprar AAPL", "neutralizar"). É onde vivem as estratégias. Pura função de dados → sinal, idealmente sem efeitos colaterais.
3. **Decisão / Risco (Decision & Risk)** — Converte sinal em **intenção de ordem** já dimensionada (quantidade, tipo, stop) e aplica regras de risco: tamanho máximo de posição, exposição total, perda diária máxima, *symbol whitelist*. Pode vetar um sinal.
4. **Execução (Execution)** — Valida a intenção uma última vez (idempotência, fundos, mercado aberto) e **envia a ordem à Alpaca**. Lida com rejeições, *partial fills*, retry e *client order id* para deduplicação.
5. **Monitoramento (Monitoring)** — Loop temporizado: observa posições/ordens abertas, ajusta *trailing stops*, dispara *schedules* (ex.: reavaliar a cada 5–15 min), detecta divergências e emite alertas.
6. **Persistência de estado (State / Persistence)** — Guarda posições, ordens, stops, P&L e logs de auditoria. Permite reconciliação e recuperação após crash. Regra: **persistir só o que não pode ser recomputado** de fontes externas. ([mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure))

### Diagrama textual

```
                        ┌──────────────────────────────────────────────────┐
                        │                  ALPACA (broker)                  │
                        │   REST/WebSocket · fonte de verdade · clock/cal    │
                        └───────▲───────────────────────────────┬──────────┘
                                │ ordens / fills                 │ dados de mercado + conta
                                │                                │
   ┌────────────┐   intenção  ┌┴───────────┐   sinal   ┌────────▼─────────┐
   │  MONITOR   │◀───────────▶│  EXECUTOR   │◀──────────│   PLANEJADOR     │
   │ (loop)     │  ajusta     │ (Executor)  │  valida   │   (Planner)      │
   │ stops,     │  stops      │ + envia     │  + envia  │ estratégia+risco │
   │ schedules, │             │ à Alpaca    │           │ → intenção ordem │
   │ alertas    │             └──────┬──────┘           └────────▲─────────┘
   └─────┬──────┘                    │                           │
         │                           │  todos publicam/consomem  │ dados normalizados
         │            ┌──────────────▼───────────────────────────┴───────┐
         │            │            MESSAGE BUS / EVENT QUEUE              │
         │            │   (in-memory queue agora → Redis/event bus depois) │
         │            └──────────────┬───────────────────────────────────┘
         │                           │
         ▼                           ▼
   ┌──────────────┐          ┌───────────────────┐        ┌────────────────────┐
   │  SCHEDULER   │          │  INGESTÃO DE DADOS │        │   PERSISTÊNCIA      │
   │ (APScheduler)│          │ (REST/WebSocket)   │        │ SQLite: posições,   │
   │ market hours │          │ normaliza p/ bus   │        │ ordens, stops, P&L, │
   │ + calendário │          └───────────────────┘        │ audit log           │
   └──────────────┘                                        └────────────────────┘
                                        ▲                            │
                                        └─── reconciliação no boot ──┘
```

Fluxo de uma decisão: `Ingestão` publica dados → `Planejador` gera **intenção de ordem** → `Executor` valida e envia à Alpaca → fill volta → `Persistência` grava → `Monitor` acompanha e ajusta stops. O `Scheduler` é quem "acorda" o ciclo dentro do horário de mercado.

---

## 2. Os 3 agentes: Planejador, Executor, Monitor

Cada agente tem **uma responsabilidade**, uma **fronteira clara** e um **contrato de entrada/saída**. Eles não se chamam diretamente (ver §3): comunicam-se por mensagens.

### 2.1 Planejador (Planner)

- **Responsabilidade:** decidir *o que* fazer. Roda as estratégias sobre os dados de mercado, aplica gestão de risco e dimensionamento, e produz **intenções de ordem** (`OrderIntent`). É o único que conhece a lógica de estratégia.
- **Consome:** barras/quotes normalizadas, posições atuais, parâmetros de risco e config de estratégia.
- **Produz:** `OrderIntent` (symbol, side, qty, order_type, limit/stop, motivo, strategy_id). **Não fala com a Alpaca.** Não sabe enviar ordens.
- **Fronteira:** decisão e risco. Se o risco veta, nem chega a gerar intenção. Função (quase) pura sobre estado → fácil de testar.

### 2.2 Executor (Executor)

- **Responsabilidade:** decidir *se e como* enviar. Recebe `OrderIntent`, faz a **última validação** (mercado aberto via clock, fundos suficientes, idempotência por `client_order_id`, *whitelist*) e **envia a ordem à Alpaca**. Trata rejeições, partial fills e retries.
- **Consome:** `OrderIntent` do bus + estado da conta da Alpaca.
- **Produz:** `OrderResult` / eventos de fill; grava ordem na persistência **antes** da chamada de rede (a intenção sobrevive se a chamada falhar). ([Medium / James Hall](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c))
- **Fronteira:** é o **único** componente que escreve no broker. Centralizar isso evita ordens duplicadas e facilita auditoria.

### 2.3 Monitor (Monitor)

- **Responsabilidade:** vigiar *o que está aberto*. Roda em **loop temporizado** (5–15 min): acompanha posições e ordens abertas, **ajusta stops** (ex.: trailing stop), dispara *schedules* (pedir ao Planejador para reavaliar), detecta divergências entre estado local e broker, e emite **alertas**.
- **Consome:** posições/ordens abertas (persistência + broker), preços atuais, regras de stop.
- **Produz:** ajustes de stop (que viram novas `OrderIntent`/modificações via Executor), gatilhos de reavaliação, alertas/logs.
- **Fronteira:** observa e dispara, mas **roteia mudanças de ordem pelo Executor** — não envia ordens por conta própria. Mantém a regra "só o Executor escreve no broker".

> Regra de ouro das fronteiras: **Planejador decide, Executor escreve, Monitor vigia.** Nenhum invade a competência do outro.

---

## 3. Comunicação entre agentes

Opções, do mais simples ao mais robusto:

| Abordagem | Como | Prós | Contras |
|---|---|---|---|
| **Chamadas diretas** | `executor.handle(planner.plan())` | trivial, zero infra | acoplamento forte, difícil testar/escalar, sem buffer |
| **Fila in-memory** (`queue.Queue` / `asyncio.Queue`) | produtor/consumidor num processo | desacopla, buffer, fácil de mockar, sem infra externa | um processo só; perde mensagens se o processo morre |
| **Redis (lista/stream/pub-sub)** | fila/stream entre processos | sobrevive a restart (streams), multi-processo, observável | infra a operar, serialização, mais complexidade |
| **Event bus** (ex.: pub/sub interno do NautilusTrader) | eventos tipados num barramento | escalável, múltiplos consumidores | maior peso conceitual |

### Recomendação: comece com **fila in-memory + padrão produtor/consumidor**

No MVP (um processo, paper trading), use uma **`queue.Queue` (ou `asyncio.Queue`) por tópico** (`signals`, `intents`, `fills`, `alerts`). Cada agente é um consumidor de uma fila e produtor de outra. Justificativa:

- **Desacopla** os agentes sem nenhuma infra externa — o Planejador não importa o Executor.
- A interface fica idêntica à de uma fila Redis: trocar `queue.Queue` por Redis Streams depois é **mudar a implementação, não os agentes**.
- A própria literatura recomenda manter o framework simples e investir em eval/observabilidade/recuperação, que é onde está a diferença real entre um bom e um mau sistema. ([augmentcode.com](https://www.augmentcode.com/tools/open-source-agent-orchestrators))

Migrar para **Redis Streams** quando precisar de: múltiplos processos, durabilidade da fila entre restarts, ou consumidores concorrentes.

```python
# message_bus.py — fila in-memory plugável, trocável por Redis depois
from __future__ import annotations
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Protocol, Any


@dataclass
class Message:
    topic: str
    payload: Any
    meta: dict = field(default_factory=dict)


class MessageBus(Protocol):
    def publish(self, topic: str, payload: Any, **meta: Any) -> None: ...
    def consume(self, topic: str, timeout: float | None = None) -> Message | None: ...


class InMemoryBus:
    """Implementação MVP: um Queue por tópico. Mesma interface de um bus Redis."""
    def __init__(self) -> None:
        self._topics: dict[str, Queue[Message]] = {}

    def _q(self, topic: str) -> Queue[Message]:
        return self._topics.setdefault(topic, Queue())

    def publish(self, topic: str, payload: Any, **meta: Any) -> None:
        self._q(topic).put(Message(topic=topic, payload=payload, meta=meta))

    def consume(self, topic: str, timeout: float | None = 1.0) -> Message | None:
        try:
            return self._q(topic).get(timeout=timeout)
        except Empty:
            return None
```

---

## 4. Orquestração desacoplada: a interface `AgentOrchestrator`

Queremos que **hoje** uma orquestração local simples rode os 3 agentes, e **amanhã** possamos plugar OpenSquad (ou LangGraph/CrewAI) **sem tocar nos agentes nem na lógica de trading**. A solução é uma **interface estável** (`AgentOrchestrator`) que o resto do sistema usa, com implementações concretas plugáveis.

### Princípio

O núcleo de trading depende da **abstração**, não da implementação (Dependency Inversion). A orquestração concreta é um detalhe injetado na inicialização. Isso é exatamente o que justifica a camada desacoplada que você pediu: a escolha de framework "não deve atrapalhar" — o que importa é eval, observabilidade e recuperação, e esses ficam no seu código, não no orquestrador. ([augmentcode.com](https://www.augmentcode.com/tools/open-source-agent-orchestrators))

### Esboço da interface e de um agente base

```python
# orchestrator.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


# --- contrato de um agente ---------------------------------------------------
class Agent(Protocol):
    name: str
    def step(self, bus: "MessageBus") -> None:
        """Executa um ciclo: consome do bus, processa, publica resultados."""
        ...


class BaseAgent(ABC):
    """Agente base: dá nome, logging e o esqueleto do ciclo. Subclasses só
    implementam `handle`. Não conhece a orquestração concreta."""
    def __init__(self, name: str, inbox: str, bus: "MessageBus") -> None:
        self.name = name
        self.inbox = inbox
        self.bus = bus

    def step(self, bus: "MessageBus") -> None:
        msg = bus.consume(self.inbox, timeout=1.0)
        if msg is not None:
            self.handle(msg)

    @abstractmethod
    def handle(self, msg: "Message") -> None: ...


# --- a interface de orquestração que o resto do sistema usa ------------------
@dataclass
class OrchestratorConfig:
    tick_seconds: float = 1.0


class AgentOrchestrator(ABC):
    """Abstração estável. Implementações: LocalOrchestrator (agora),
    OpenSquadOrchestrator (depois). O núcleo de trading só conhece ISTO."""
    @abstractmethod
    def register(self, agent: Agent) -> None: ...

    @abstractmethod
    def start(self) -> None:
        """Inicia o loop de orquestração (bloqueante ou em background)."""

    @abstractmethod
    def stop(self) -> None: ...


# --- implementação concreta LOCAL (MVP) --------------------------------------
class LocalOrchestrator(AgentOrchestrator):
    """Round-robin simples sobre os agentes registrados, dentro de um processo.
    Sem dependência externa. Plugável por outra impl depois."""
    def __init__(self, bus: "MessageBus", config: OrchestratorConfig | None = None) -> None:
        self.bus = bus
        self.config = config or OrchestratorConfig()
        self._agents: list[Agent] = []
        self._running = False

    def register(self, agent: Agent) -> None:
        self._agents.append(agent)

    def start(self) -> None:
        import time
        self._running = True
        while self._running:
            for agent in self._agents:
                agent.step(self.bus)
            time.sleep(self.config.tick_seconds)

    def stop(self) -> None:
        self._running = False
```

### Implementação futura (OpenSquad ou outro)

```python
class OpenSquadOrchestrator(AgentOrchestrator):
    """ESBOÇO/STUB. NÃO implementado: a API pública/SDK do OpenSquad para
    embutir agentes Python ainda precisa ser confirmada (ver §5). Esta classe
    existe só para mostrar que a troca é local — register/start/stop continuam
    iguais; o que muda é como os agentes são despachados por baixo."""
    def __init__(self, bus, config=None):
        raise NotImplementedError(
            "Aguardando documentação/SDK do OpenSquad. Mapear os agentes "
            "Planejador/Executor/Monitor para 'roles' de um squad e despachar "
            "cada step() pela ferramenta. Até lá, usar LocalOrchestrator."
        )
```

Ponto-chave: **trocar de orquestrador é trocar uma linha na composição** (`orchestrator = LocalOrchestrator(bus)` → `OpenSquadOrchestrator(bus)`). Agentes, estratégias e broker não mudam.

---

## 5. OpenSquad — o que existe publicamente (e o que falta)

**Honestidade primeiro.** Pesquisei e **OpenSquad existe como projeto open-source público**, mas as informações são limitadas e **não há documentação de SDK suficiente para embutir agentes Python de trading de forma confiável**. Não vou inventar API.

O que foi possível **confirmar** do repositório principal ([github.com/brunomcps/opensquad](https://github.com/brunomcps/opensquad)):

- É um **framework de orquestração multi-agente** com pitch "descreva em linguagem natural e ele cria um time de agentes especializados".
- **Stack:** TypeScript/Node.js (Node 20+), **não é uma biblioteca Python**. (≈72% TS, ≈2% Python no repo.)
- **Uso é via CLI / slash-commands**, não via API de biblioteca: `npx opensquad init`, e comandos como `/opensquad create`, `/opensquad run <name>`, `/opensquad list`, `/opensquad dashboard`.
- Conceito central: um **"squad"** é um time de agentes com **papéis** (ex.: Researcher, Strategist, Writer, Reviewer) rodando em **pipeline com checkpoints** de aprovação humana. Tem um "Architect" que desenha o squad e uma "Virtual Office" (interface 2D em tempo real).
- Arquivos de config no repo: `.env.example`, `.mcp.json` (sugere integração via **MCP**), licença **MIT**.
- Funciona com stacks como Google Antigravity (Gemini free tier) ou OpenCode com LLMs locais (Ollama, LM Studio).

> ⚠️ Cuidado: há **vários projetos com nomes parecidos** ("agent-squad" da 2FastLabs, "agents-squads", o **AWS Agent Squad/Multi-Agent Orchestrator**, e o "Squad" do GitHub Copilot). **Não são o mesmo projeto.** Confirme com o usuário **qual** "OpenSquad" é o pretendido antes de integrar. ([github.com/2FastLabs/agent-squad](https://github.com/2FastLabs/agent-squad), [github blog](https://github.blog/ai-and-ml/github-copilot/how-squad-runs-coordinated-ai-agents-inside-your-repository/))

**O que falta para integrar de verdade:** como o OpenSquad é orientado a CLI/MCP em Node, **não há (publicamente documentado) um SDK Python para registrar/despachar agentes programaticamente** como a interface `AgentOrchestrator` precisa. Para uma integração real, o usuário precisará fornecer: (a) a doc do SDK/MCP do OpenSquad, ou (b) o contrato de como acionar um squad e receber resultados de volta a partir de Python. **Até lá, use `LocalOrchestrator`** e mantenha o `OpenSquadOrchestrator` como stub.

### Alternativas conhecidas (contexto) — orquestração multi-agente

Frameworks maduros e bem documentados, caso OpenSquad não atenda:

- **LangGraph** — grafo dirigido com arestas condicionais e **estado explícito**; melhor nota em orquestração multi-agente e *production-readiness*; tradeoff é verbosidade. ([tensoria.fr](https://tensoria.fr/en/blog/multi-agent-orchestration-comparison), [qubittool.com](https://qubittool.com/blog/ai-agent-framework-comparison-2026))
- **CrewAI** — "crews" baseados em **papéis** e *process types*; chega rápido a um protótipo, mas tem comunicação agente-a-agente limitada e sem checkpointing nativo. ([humaineeti.ai](https://www.humaineeti.ai/resources/multi-agent-orchestration-frameworks))
- **AutoGen / AG2** — padrão **conversacional** (GroupChat); flexível, porém caro (20+ chamadas LLM por tarefa). ([pecollective.com](https://pecollective.com/blog/ai-agent-frameworks-compared/))
- **OpenAI Agents SDK** — *handoffs* explícitos entre agentes; leve e direto. ([openai.github.io](https://openai.github.io/openai-agents-python/multi_agent/))
- **Google ADK / Claude Agent SDK / Strands** — outros entrantes citados nas comparações 2026. ([qubittool.com](https://qubittool.com/blog/ai-agent-framework-comparison-2026))

> Caveat das comparações: *"o debate de framework é em grande parte uma distração. A diferença entre um bom e um mau sistema de agentes quase nunca é o framework — é o pipeline de eval, a observabilidade e a lógica de recuperação de falhas."* Por isso a interface `AgentOrchestrator` desacoplada é a decisão certa: ela isola o framework e te deixa investir onde importa. ([augmentcode.com](https://www.augmentcode.com/tools/open-source-agent-orchestrators))

---

## 6. Agendamento (scheduling)

Use **APScheduler**. Ele oferece `CronTrigger` e `IntervalTrigger`, aceita **timezone** por job (string ou `tzinfo`) e é o caminho padrão para cron em Python. ([apscheduler.readthedocs.io](https://apscheduler.readthedocs.io/en/stable/modules/triggers/cron.html), [betterstack.com](https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/))

Regras para trading US:

- **Só rodar em horário de mercado.** Antes de cada ciclo, **consulte o `clock` da Alpaca** (`is_open`, `next_open`, `next_close`) em vez de assumir 9:30–16:00 fixo. ([Alpaca Clock](https://alpaca.markets/sdks/python/api_reference/trading/clock.html))
- **Feriados e fechamentos antecipados:** use o **`calendar` da Alpaca**, que serve a lista de pregões (1970–2029) com horários de abertura/fechamento já considerando *early closures*. Não mantenha sua própria lista de feriados. ([Alpaca Calendar](https://alpaca.markets/sdks/python/api_reference/trading/calendar.html))
- **Fuso (DST):** opere o scheduler internamente em **UTC** e converta para `America/New_York` só na borda, ou agende contra o clock/calendar da Alpaca — assim você evita os buracos/duplicações do horário de verão. APScheduler recomenda explicitamente evitar horários no momento da virada de DST. ([apscheduler.readthedocs.io](https://apscheduler.readthedocs.io/en/latest/modules/triggers/cron.html))
- **Verificações periódicas:** `IntervalTrigger` de **5–15 min** para o Monitor reavaliar; o ciclo verifica o clock e **no-op se o mercado estiver fechado**.

```python
# scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

def make_scheduler(monitor_tick, planner_tick) -> BlockingScheduler:
    # tudo em UTC; a checagem real de "mercado aberto" usa o clock da Alpaca
    sched = BlockingScheduler(timezone="UTC")
    # Monitor: a cada 10 min, mas só age se a Alpaca disser que o mercado está aberto
    sched.add_job(monitor_tick, IntervalTrigger(minutes=10), id="monitor")
    sched.add_job(planner_tick, IntervalTrigger(minutes=15), id="planner")
    return sched

def market_is_open(trading_client) -> bool:
    # fonte de verdade: clock da Alpaca (considera feriados/early close via calendar)
    return trading_client.get_clock().is_open
```

---

## 7. Persistência de estado

**Comece com SQLite.** É suficiente para um processo, dá transações ACID, é um arquivo só e facilita backup/auditoria. Migre para Postgres só se precisar de concorrência multi-processo.

**O que guardar e por quê** (regra: persistir só o que **não dá para recomputar** do broker): ([mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure))

| Tabela | Conteúdo | Por quê |
|---|---|---|
| `positions` | symbol, qty, avg_price, strategy_id, updated_at | "ideia local" da posição, reconciliada contra o broker |
| `orders` | client_order_id, broker_order_id, status, side, qty, type, timestamps | saber o que foi submetido **antes** de um crash; idempotência |
| `stops` | symbol, stop_price, trailing_pct, order_id | stops são lógica do bot — o broker pode não ter todos |
| `pnl` | realized/unrealized por dia e por estratégia | métricas e gate de risco (perda diária máxima) |
| `audit_log` | evento, ator (Planner/Executor/Monitor), payload, ts | trilha de auditoria do "porquê" de cada decisão e ordem |

**Padrão de escrita seguro:** grave a **intenção/ordem no DB antes da chamada de rede**, não depois — se a chamada falhar, a intenção sobrevive e a recuperação sabe que existia uma ordem pendente. ([Medium / James Hall](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c))

### Recuperação após crash: reconciliar com o broker no boot

Ao iniciar, **antes de qualquer decisão nova**, o bot deve reconciliar: ([mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure), [Medium / James Hall](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c))

1. Buscar **posições reais** e **ordens abertas** na Alpaca.
2. Comparar com o estado local (`positions`, `orders`).
3. Resolver divergências tratando o **broker como verdade**: ordens que o DB achava pendentes mas a Alpaca já preencheu/cancelou são atualizadas; posições local≠broker são corrigidas para o broker; stops órfãos são recriados.
4. Só então liberar o ciclo de trading.

```python
def reconcile_on_boot(db, trading_client) -> None:
    broker_positions = {p.symbol: p for p in trading_client.get_all_positions()}
    broker_orders = {o.client_order_id: o for o in trading_client.get_orders(status="open")}

    # broker é a fonte de verdade: estado local é corrigido para bater com ele
    db.upsert_positions_from_broker(broker_positions)
    db.reconcile_orders(broker_orders)        # fecha/atualiza ordens que sumiram
    db.recreate_missing_stops(broker_positions)
    db.write_audit("reconcile_on_boot", actor="system", payload={"ok": True})
```

---

## 8. Estrutura de pastas recomendada

```
trading-automation/
├── pyproject.toml
├── config/
│   ├── settings.py            # carrega env, modo paper/live, parâmetros de risco
│   └── strategies.yaml        # config declarativa das estratégias
├── trading/
│   ├── agents/                # os 3 agentes + base
│   │   ├── base.py            # BaseAgent
│   │   ├── planner.py         # Planejador
│   │   ├── executor.py        # Executor
│   │   └── monitor.py         # Monitor
│   ├── orchestration/         # camada DESACOPLADA
│   │   ├── orchestrator.py    # AgentOrchestrator (interface) + LocalOrchestrator
│   │   ├── opensquad.py       # OpenSquadOrchestrator (stub até ter SDK)
│   │   └── bus.py             # MessageBus: InMemoryBus (→ Redis depois)
│   ├── strategies/            # lógica pura de sinal (testável isolada)
│   │   └── sma_crossover.py
│   ├── broker/                # adaptador Alpaca (única porta p/ o broker)
│   │   ├── alpaca_client.py   # ordens, posições, clock, calendar
│   │   └── interfaces.py      # BrokerPort (Protocol) p/ permitir mock/outro broker
│   ├── data/                  # ingestão e normalização de dados de mercado
│   │   └── ingestion.py
│   ├── risk/                  # regras de risco e sizing
│   │   └── rules.py
│   ├── persistence/           # SQLite + repositórios + reconciliação
│   │   ├── db.py
│   │   ├── models.py
│   │   └── reconcile.py
│   ├── scheduling/
│   │   └── scheduler.py       # APScheduler + checagem de market_is_open
│   └── app.py                 # composição: monta bus, agentes, orchestrator, scheduler
├── tests/
│   ├── test_planner.py
│   ├── test_executor.py
│   ├── test_reconcile.py
│   └── test_orchestrator.py
└── docs/
    └── research/05-arquitetura-agentes.md   # este documento
```

Pontos da estrutura:
- **`broker/` é a única porta para a Alpaca** — um `BrokerPort` (Protocol) permite mockar nos testes e, em teoria, trocar de broker.
- **`strategies/` é lógica pura** — sem I/O, fácil de testar com dados sintéticos.
- **`orchestration/` isola o framework** — `agents/` e `strategies/` não importam OpenSquad/LangGraph/etc.

---

## 9. Erros comuns em arquitetura de trading bots

1. **Confiar no estado local em vez do broker.** O broker é a verdade; reconcilie no boot e periodicamente. Não reconciliar é a causa nº1 de posições fantasma. ([mbrenndoerfer.com](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure))
2. **Gravar depois da chamada de rede.** Se grava o fill só após a Alpaca responder e o processo morre no meio, você perde a ordem. Grave a intenção **antes**. ([Medium / James Hall](https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c))
3. **Sem idempotência.** Reenviar uma ordem após timeout sem `client_order_id` duplica posição. Sempre use id de cliente determinístico.
4. **Hardcodar horário de mercado / feriados.** Use `clock` e `calendar` da Alpaca; feriados e *early close* mudam. ([Alpaca Calendar](https://alpaca.markets/sdks/python/api_reference/trading/calendar.html))
5. **Ignorar DST.** Agendar em horário local sem cuidado quebra duas vezes por ano. Opere em UTC / contra o clock do broker. ([apscheduler.readthedocs.io](https://apscheduler.readthedocs.io/en/latest/modules/triggers/cron.html))
6. **Misturar decisão e execução.** Se a estratégia envia ordem direto, fica impossível auditar, testar risco ou trocar de broker. Mantenha Planejador↛broker.
7. **Acoplar agentes por chamada direta.** Vira monólito impossível de evoluir; use o bus.
8. **Casar com um framework de orquestração cedo demais.** O framework não é o gargalo — eval, observabilidade e recuperação são. Isole-o atrás de `AgentOrchestrator`. ([augmentcode.com](https://www.augmentcode.com/tools/open-source-agent-orchestrators))
9. **Sem trilha de auditoria.** Sem `audit_log`, você não consegue explicar *por que* o bot fez um trade — fatal para debugging e compliance.
10. **Começar em live.** Paper trading primeiro; a API de paper da Alpaca é idêntica à de live, então não há desculpa. ([Alpaca Clock](https://alpaca.markets/sdks/python/api_reference/trading/clock.html))
11. **Tratar partial fills como tudo-ou-nada.** Ordens preenchem em partes; o Executor e a persistência precisam acumular `filled_qty`.
12. **Sem *kill switch* / limite de perda diária.** Uma regra de risco que pausa o bot ao atingir perda máxima diária evita catástrofe.

---

## Fontes

- Alpaca — Calendar API: https://alpaca.markets/sdks/python/api_reference/trading/calendar.html
- Alpaca — Clock API: https://alpaca.markets/sdks/python/api_reference/trading/clock.html
- NautilusTrader — Architecture: https://nautilustrader.io/docs/latest/concepts/architecture/
- Quant Trading System Architecture & Infrastructure (M. Brenndoerfer): https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure
- Concurrency, State Management & Fault Tolerance in Trading Bots (J. Hall): https://medium.com/@halljames9963/concurrency-state-management-and-fault-tolerance-in-stock-trading-bots-da774736c58c
- OpenSquad (repo principal): https://github.com/brunomcps/opensquad
- agent-squad (2FastLabs, projeto distinto): https://github.com/2FastLabs/agent-squad
- GitHub Blog — Squad (Copilot, projeto distinto): https://github.blog/ai-and-ml/github-copilot/how-squad-runs-coordinated-ai-agents-inside-your-repository/
- Open-Source Agent Orchestrators 2026 (Augment Code): https://www.augmentcode.com/tools/open-source-agent-orchestrators
- LangGraph vs CrewAI vs AutoGen — Benchmark 2026 (Tensoria): https://tensoria.fr/en/blog/multi-agent-orchestration-comparison
- AI Agent Framework Showdown 2026 (QubitTool): https://qubittool.com/blog/ai-agent-framework-comparison-2026
- Multi-Agent Orchestration Frameworks 2026 (Humaineeti): https://www.humaineeti.ai/resources/multi-agent-orchestration-frameworks
- AI Agent Frameworks Compared (PE Collective): https://pecollective.com/blog/ai-agent-frameworks-compared/
- OpenAI Agents SDK — Multi-agent: https://openai.github.io/openai-agents-python/multi_agent/
- APScheduler — CronTrigger docs: https://apscheduler.readthedocs.io/en/stable/modules/triggers/cron.html
- APScheduler — guia (Better Stack): https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/

---

## Como virar skill

Para transformar este documento de referência em uma **skill** reutilizável do Claude Code:

1. **Criar o diretório da skill:** `~/.claude/skills/arquitetura-agentes/` (ou no projeto, `.claude/skills/`).
2. **Mover o frontmatter para `SKILL.md`:** o bloco YAML no topo deste arquivo (`name`, `description`) já é compatível. Ajustar `description` para conter **gatilhos** claros — ex.: "Use ao projetar, revisar ou estender a arquitetura do trading bot multi-agente (Planejador/Executor/Monitor), a interface AgentOrchestrator, comunicação por bus, scheduling com APScheduler ou persistência/reconciliação SQLite com a Alpaca."
3. **Corpo da skill:** colar §§1–9 como o conteúdo do `SKILL.md` (a skill é instrução + referência). Manter os esboços de código — eles viram o "scaffold" que o agente copia ao criar `orchestrator.py`, `bus.py`, `scheduler.py`, etc.
4. **Anexar arquivos de apoio (opcional):** extrair os blocos de código para `assets/` (ex.: `assets/orchestrator.py`, `assets/bus.py`) e referenciá-los no `SKILL.md`, para que a skill instale o esqueleto diretamente.
5. **Atualizar quando o OpenSquad for confirmado:** assim que o usuário fornecer o SDK/contrato do OpenSquad (§5), substituir o stub `OpenSquadOrchestrator` pela implementação real e atualizar a §5/§4 da skill.
6. **Registrar no índice de memória** do projeto, se desejado, apontando para a skill como referência canônica de arquitetura do bot.
