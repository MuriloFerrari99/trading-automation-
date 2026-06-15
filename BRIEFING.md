# Briefing — Sistema de Automação de Investimentos

> Cole/abra este arquivo no início da sessão do Claude Code. Ele é o ponto de partida do projeto.

Você é um engenheiro de software sênior especializado em FinTech e sistemas de trading automatizado. Vamos construir, do zero, um sistema de automação de investimentos. **Antes de escrever qualquer código, leia este briefing inteiro E a base de conhecimento em [`docs/`](docs/README.md), e me apresente um plano de arquitetura e um roadmap em fases. Não comece a implementar até eu aprovar o plano.**

> 📚 **Base de pesquisa já existe.** A pasta [`docs/research/`](docs/README.md) contém 6 documentos de referência aprofundados (Alpaca API, risco/execução, smart money, Wheel/opções, arquitetura multi-agente, backtesting/testes), com fontes citadas e padrões de código. **Consulte-os ao planejar e ao codar** — eles são a fonte de verdade técnica deste projeto.

## Objetivo
Sistema de trading automatizado em **Python**, orquestrado por múltiplos agentes (papéis: Planejador, Executor, Monitor), executando ordens via **API da Alpaca**.

## Restrições inegociáveis (segurança)
1. **Somente Paper Trading** nesta fase. O código deve apontar exclusivamente para o endpoint de paper da Alpaca. Operar com dinheiro real exige uma flag explícita (`LIVE_TRADING=true`) que, por ora, deve estar bloqueada por um *guard* no código que aborta a execução real.
2. **Segredos nunca no código nem no git.** Use um arquivo `.env` (carregado via `python-dotenv`) com `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` e `ALPACA_ENDPOINT`. Já existem `.env.example` versionado e `.env` no `.gitignore`.
3. **Kill switch:** um mecanismo central que pausa todas as ordens (ex: flag em arquivo/DB ou variável de ambiente).
4. Todo trade executado deve ser **logado** (timestamp, ativo, lado, qtd, preço, estratégia que originou) em arquivo/SQLite para auditoria.

## Stack sugerida (proponha alternativas se fizer sentido)
- Python 3.11+, `alpaca-py` (SDK oficial), `python-dotenv`, `apscheduler` para o agendamento, `SQLite` para persistência de estado/posições/logs.
- Estrutura de pastas modular: `agents/`, `strategies/`, `broker/`, `data/`, `config/`, `tests/`.
- Testes com `pytest` e uso intensivo de *mocks* da API antes de tocar a Alpaca real.

## Estratégias (implementar como módulos independentes e testáveis)

### Nível 1 — Gestão de risco e preço médio
- **Trailing Stop:** stop-loss que sobe junto com o preço (ex: trava 10% abaixo da máxima), mas nunca desce.
- **Ladder Buys:** ordens de compra escalonadas em quedas (ex: +X ações a −20%, +Y ações a −30%) para reduzir preço médio. Parametrizável por ativo.

### Nível 2 — Sinais de "Smart Money"
- Módulo para ingerir dados de movimentação de grandes fundos e de congressistas dos EUA (ex: Capital Trades / dados públicos de disclosure). **Importante:** este módulo deve apenas **gerar sinais/sugestões** para o Agente Planejador — não executar automaticamente. Isole a fonte de dados atrás de uma interface (`SignalProvider`) para eu poder trocar de provedor.

### Nível 3 — Wheel Strategy (opções)
- Venda de **Puts** ~10% abaixo do preço (coleta de prêmio / entrada).
- Se exercido, venda de **Covered Calls** ~10% acima do preço de custo.
- Trate explicitamente a verificação de que a conta tem permissão/nível de opções e liquidez antes de habilitar este nível.

## Agentes e orquestração
- **Planejador:** consome dados de mercado + sinais, decide qual estratégia aplicar e gera *intenções de ordem*.
- **Executor:** traduz intenções em ordens na Alpaca, com validação e tratamento de erro/retry.
- **Monitor:** roda em intervalos (ex: a cada 5–15 min) **apenas durante o horário de mercado**, ajusta trailing stops e dispara os agendamentos.
- Defina a interface de comunicação entre os agentes (fila, eventos ou chamadas diretas) e justifique a escolha.

## Sobre o OpenSquad
Pretendo usar a arquitetura de agentes do **OpenSquad** para orquestrar isso. **Atenção (descoberto na pesquisa — ver [doc 05](docs/research/05-arquitetura-agentes.md)):** o OpenSquad público é um framework **TypeScript/Node.js via CLI/MCP**, não uma lib Python com SDK de runtime — e há projetos homônimos. Por isso: **deixe a camada de orquestração desacoplada** (interface `AgentOrchestrator`), implemente um `LocalOrchestrator` em Python para o MVP, e só integre o OpenSquad quando eu confirmar qual é e fornecer a doc. Não invente API do OpenSquad.

## Entrega esperada agora
1. Plano de arquitetura (diagrama textual dos módulos e do fluxo entre agentes).
2. Roadmap em fases — sugira começar por um **MVP**: conexão Alpaca paper + 1 estratégia (Trailing Stop) + log + 1 agente Monitor. Os demais níveis vêm depois.
3. Lista de decisões/perguntas em aberto que você precisa que eu responda.

**Comece pelo plano.**

---

## Notas de risco (não-técnicas)
- **Wheel Strategy / opções** exige nível de aprovação de opções na corretora e tem risco real de exercício.
- **Copy trading de congressistas** é legal (dados públicos via STOCK Act), mas o atraso de divulgação (até 45 dias) reduz muito o "edge". Tratar como um sinal entre outros, não como tese principal.
