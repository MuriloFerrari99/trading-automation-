# Base de Conhecimento — Trading Bot

Documentação de pesquisa que fundamenta a construção deste bot de trading. Cada arquivo em `research/` é uma referência aprofundada, com fontes citadas e padrões de código, **estruturada para virar uma skill do Claude Code depois** (todos começam com frontmatter `name`/`description` e terminam com uma seção "Como virar skill").

> Pesquisa conduzida em jun/2026, com informação atual de 2025/2026. Pontos marcados como "verificar" devem ser confirmados contra a doc oficial / no paper account antes de virar código de produção.

## Índice

| # | Doc | Cobre |
|---|-----|-------|
| 01 | [research/01-alpaca-api.md](research/01-alpaca-api.md) | API Alpaca + SDK `alpaca-py`: auth, ordens, posições, market data, opções, rate limits, pitfalls |
| 02 | [research/02-risco-execucao.md](research/02-risco-execucao.md) | Trailing stop, ladder buys, position sizing (1-2%/ATR/Kelly), kill switch, slippage, reconciliação |
| 03 | [research/03-smart-money-signals.md](research/03-smart-money-signals.md) | Copy trading: trades do Congresso (STOCK Act), 13F via SEC EDGAR, fontes grátis vs pagas, interface `SignalProvider` |
| 04 | [research/04-wheel-opcoes.md](research/04-wheel-opcoes.md) | Wheel Strategy: CSP → covered call, seleção por delta/DTE, greeks, suporte a opções na Alpaca |
| 05 | [research/05-arquitetura-agentes.md](research/05-arquitetura-agentes.md) | Arquitetura multi-agente (Planejador/Executor/Monitor), comunicação, `AgentOrchestrator`, scheduling, persistência |
| 06 | [research/06-backtesting-testes.md](research/06-backtesting-testes.md) | Backtesting (frameworks), vieses, métricas, paper trading, testes pytest com mock da Alpaca |
| 07 | [research/07-fimathe-forex.md](research/07-fimathe-forex.md) | Metodologia FIMATHE (forex/ouro): canais, zona neutra, virada de mão, linhas do Equador, barra elefante, fatiamento + adaptação ouro→pares de moedas |
| 08 | [research/08-fimathe-engine.md](research/08-fimathe-engine.md) | **`FimatheEngine` (implementada, `fimathe/`):** canais/ZN/PCM/Fibonacci/RSI/ATR/ADX, sinais e stops sobre OHLC; feature table p/ ML (`get_features_for_ml`) |
| 09 | [research/09-loop-de-feedback.md](research/09-loop-de-feedback.md) | **Camada 0 (implementada):** loop de feedback — registro decisão+contexto+resultado, classificador de regime e avaliação por estratégia/regime (pacote `feedback/`) |
| 10 | [research/10-camada-decisao.md](research/10-camada-decisao.md) | **Camada de decisão (implementada):** gate por regime que veta combos estratégia×regime com edge negativo + score por sinais (pacote `intelligence/`), fiado no `LocalOrchestrator`/`main` |
| 11 | [research/11-camada-ml.md](research/11-camada-ml.md) | **Camada 2 — ML (implementada, `ml/`):** classificador P(win) de setup (logística numpy) em champion/challenger; promove só com skill comprovado (IC99 AUC) + expectancy; cold-start seguro |
| 12 | [research/12-sizing-dinamico.md](research/12-sizing-dinamico.md) | **Sizing dinâmico (implementado, `sizing/`):** tamanho por convicção via Kelly fracionário (¼), teto/piso, qty 0 sem edge; compõe `risk/sizing` |
| 13 | [research/13-camada-sintese.md](research/13-camada-sintese.md) | **Camada 3 — síntese (implementada, `synthesis/`):** combina visões ortogonais (técnico/smart money/ML/regime) em convicção + consenso/conflito + racional; hook LLM opcional |
| 14 | [research/14-integracao.md](research/14-integracao.md) | **Integração (implementada, `integration/`):** `DecisionEnricher` compõe FimatheEngine+síntese+ML+sizing num ponto; degrada com graça; sem leak no ML; snippet de wiring |

**Fontes brutas:** [research/fontes/](research/fontes/) — transcrições de áudio que embasam docs (ex.: `fimathe-ouro-transcricao.txt`).

## Descobertas que mudam decisões do projeto

1. **SDK Alpaca:** usar `alpaca-py` (oficial atual). O `alpaca-trade-api` é legacy. (doc 01)
2. **Opções na Alpaca:** a Wheel completa (vender CSP + covered call) exige apenas **Level 1**; ✅ a conta paper deste projeto está em **options level 3** (verificado jun/2026), então Wheel e multi-leg estão liberados. A options chain básica **não retorna delta**: é preciso calcular via Black-Scholes no cliente. (doc 04)
3. **OpenSquad ≠ Python.** O OpenSquad público ([github.com/brunomcps/opensquad](https://github.com/brunomcps/opensquad)) é um framework **TypeScript/Node.js orquestrado por CLI/MCP**, não uma biblioteca Python com SDK para despachar agentes em runtime. Existem vários projetos homônimos (agent-squad, AWS Agent Squad, Squad do Copilot). **Decisão:** começar com um `LocalOrchestrator` em Python por trás da interface `AgentOrchestrator`; integrar o OpenSquad depois, quando você confirmar qual é e fornecer a doc. Não há API inventada no código. (doc 05)
4. **Smart money tem latência estrutural.** Disclosure do Congresso atrasa até ~45 dias e 13F é foto trimestral só de posições long — tratar como **um sinal entre vários, nunca execução automática cega**. SEC EDGAR é a fonte gratuita confiável (exige header `User-Agent`); Capitol Trades não tem API pública oficial; Quiver/Unusual Whales têm tiers pagos. (doc 03)
5. **Backtest antes de paper, paper antes de live.** Backtrader está estagnado (último release 2019, mas funcional); avaliar `Backtesting.py`/`VectorBT`/`NautilusTrader` — confirmar manutenção no GitHub antes de fixar. (doc 06)
6. **FIMATHE é forex/ouro, fora do escopo Alpaca.** A metodologia (doc 07) opera **pares de moedas e XAU/USD em MetaTrader** — a **Alpaca não negocia forex**. É conhecimento de metodologia (e base p/ backtest das partes determinísticas num feed forex/MT5), **não** um módulo do bot Alpaca atual. Boa parte do método é discricionária; só um subconjunto (canal/ciclos/stop/take/disjuntor) é determinístico. (doc 07)

## Caminho para virar skills

Quando o bot estiver maduro, cada doc vira uma skill em `.claude/skills/<nome>/SKILL.md`:

- O frontmatter (`name`, `description`) de cada doc já está no formato de skill.
- O corpo do doc vira o material de referência da skill; os blocos de código viram snippets reutilizáveis.
- Sugestão de skills a extrair: `alpaca-api`, `risco-execucao`, `wheel-opcoes`, `smart-money-signals`, `arquitetura-agentes`, `backtesting-testes`.
- Use a skill `skill-creator` do Claude Code para empacotar cada uma.

## Ordem de leitura sugerida (para construir o MVP)

1. **05** (arquitetura) → entende o esqueleto e as fronteiras dos agentes.
2. **01** (Alpaca) → conexão e ordens.
3. **02** (risco/execução) → o trailing stop do MVP.
4. **06** (testes) → mock da Alpaca para não bater na API real nos testes.
5. **03** e **04** entram nas fases seguintes (sinais e opções).
