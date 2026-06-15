# Briefing — Sistema de Automação de Investimentos

> Ponto de partida para uma sessão neste projeto. Dá o contexto, as restrições inegociáveis e o estado atual. Para o passo a passo de como rodar e o mapa de pacotes, ver [README.md](README.md); para a base de pesquisa técnica, ver [`docs/`](docs/README.md).

Sistema de trading automatizado em **Python**, orquestrado por agentes (Planejador → Decisão → Executor → Feedback) e executando ordens via **API da Alpaca**. As fases 0–5 do plano original estão concluídas (chassi completo, 247 testes); a evolução em curso é elevar o motor de alpha e ligar o ML (roadmap F1–F6, ver [README](README.md#status-e-roadmap)).

> 📚 **Base de pesquisa.** [`docs/research/`](docs/README.md) contém 14 documentos de referência aprofundados (Alpaca API, risco/execução, smart money, Wheel/opções, arquitetura multi-agente, backtesting, FIMATHE, e as camadas de feedback/decisão/ML/síntese/integração já implementadas), com fontes citadas. **São a fonte de verdade técnica** — consulte ao planejar e ao codar.

## Restrições inegociáveis (segurança) — sempre em vigor

1. **Somente Paper Trading.** O código aponta exclusivamente para o endpoint de paper da Alpaca. Operar com dinheiro real exige `LIVE_TRADING=true`, que hoje é **bloqueado** por um guard em `config/settings.py` (`LiveTradingBlockedError`).
2. **Segredos nunca no código nem no git.** Use `.env` (via `python-dotenv`) com `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` e `ALPACA_ENDPOINT`. Há `.env.example` versionado e `.env` no `.gitignore`.
3. **Kill switch** (`core/kill_switch.py`): variável de ambiente ou arquivo sentinela; rechecado imediatamente antes de cada ida à rede.
4. **Auditoria:** todo trade é logado em SQLite (timestamp, ativo, lado, qtd, preço, estratégia) e toda decisão/evento passa por `audit_log` com ator.

## Stack

- Python 3.11+, gerenciado com **uv** (lockfile reprodutível); `alpaca-py` (SDK oficial), `pydantic` v2 + `pydantic-settings`, `python-dotenv`, `apscheduler` (agendamento UTC), `pandas`/`numpy`, `pyyaml`. Persistência em **SQLite**.
- Estrutura modular flat na raiz: `agents/`, `broker/`, `config/`, `core/`, `data/`, `risk/`, `strategies/`, `orchestration/`, `scheduling/`, mais as camadas de inteligência (`feedback/`, `intelligence/`, `synthesis/`, `ml/`, `sizing/`, `integration/`, `fimathe/`), `simulation/` (backtesting) e `dashboard/` (UI).
- Testes com `pytest` (247) usando `FakeBroker` — nenhum teste bate na API real. `ruff` para lint.

## Estratégias (módulos independentes e testáveis)

### Nível 1 — Gestão de risco e preço médio
- **Trailing Stop** — stop nativo da Alpaca (`trail_percent`) que sobe com o preço e sobrevive a restart; protege qualquer long sem proteção.
- **Ladder Buys** — compras escalonadas em quedas para reduzir preço médio, parametrizável por ativo na watchlist.

### Nível 2 — Sinais de "Smart Money"
- Providers para trades do Congresso (STOCK Act) e movimentação de fundos (13F) atrás da interface `SignalProvider` (fontes swappable). **Apenas geram sugestões** para o Planejador — nunca executam. Hoje rodam com fontes estáticas vazias, prontas para plugar uma fonte real.

### Nível 3 — Wheel Strategy (opções)
- Venda de **Puts** ~10% abaixo (coleta de prêmio/entrada); se exercido, **Covered Calls** ~10% acima do custo. Um gate verifica nível de opções e liquidez em runtime antes de habilitar (a conta paper está em options level 3, verificado). *Opt-in* na watchlist.

## Agentes e orquestração

Pipeline sobre um message bus in-memory, atrás da interface `AgentOrchestrator` (selecionável por `ORCHESTRATOR`; `bus` é o default):

```
Scheduler → Ingestion[market_data] → Planner(estratégias+RiskManager)[intents]
  → Decision(regime+policy+track-record)[approved] → Executor[fills]
  → Feedback(anexa Outcome) ; Reconcile periódico (broker = verdade)
```

- **Planejador:** consome dados + sinais, aplica estratégias e o RiskManager, emite intenções de ordem.
- **Decisão:** classifica regime e veta combos estratégia×regime com edge negativo comprovado.
- **Executor:** único que escreve no broker — kill-switch → idempotência (client_order_id determinístico) → persiste antes da rede → submete com retry → audita.
- **Monitor:** roda em intervalos durante o horário de mercado (24/7 quando há cripto), ajusta trailing stops e dispara os ciclos.
- **Feedback / Reconcile:** fecham o loop (anexam resultado às decisões) e reconciliam estado local com o broker.

## Sobre o OpenSquad

O OpenSquad público é um framework **TypeScript/Node.js via CLI/MCP**, não uma lib Python com SDK de runtime — e há projetos homônimos (ver [doc 05](docs/research/05-arquitetura-agentes.md)). Por isso a orquestração é desacoplada: `LocalOrchestrator`/`BusOrchestrator` em Python rodam o sistema hoje, e há a costura (`OrchestratorBridge`/`OpenSquadOrchestrator`) pronta para integrar o OpenSquad quando você confirmar qual é e fornecer a doc. Não há API inventada no código.

---

## Notas de risco (não-técnicas)

- **Wheel Strategy / opções** exige nível de aprovação de opções na corretora e tem risco real de exercício.
- **Copy trading de congressistas** é legal (dados públicos via STOCK Act), mas o atraso de divulgação (até ~45 dias) reduz muito o "edge". Tratar como um sinal entre outros, não como tese principal.
