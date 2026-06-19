# PLANO DE EXECUÇÃO — BETA DISCIPLINADO → TRACK RECORD AUDITÁVEL → AUM

**Autor:** O Planner (não-conselho; plano de operacionalização)
**Repo:** `/Users/muriloferrari/Projetos/trading-automation`
**Decisão do usuário (fixa):** Beta disciplinado — carteira diversificada com gestão de risco. Objetivo = **retorno ~de mercado com drawdown menor** + **track record limpo e auditável** → rampa para gerir capital de terceiros (AUM).
**Data:** 2026-06-15

> Este documento NÃO escreve código de produção. Ele especifica tarefas → entregável → dependência → executor → critério de aceite → gate. Decisões que exigem o usuário estão marcadas com **[DECISÃO]**. Fatos regulatórios que o Pesquisador confirmou ou ainda precisa confirmar estão marcados com **[CONFIRMADO]/[PROVÁVEL]/[INCERTO — verificar com advogado]**.

---

## 0. LEITURA DA SITUAÇÃO (o que já existe vs. o que falta)

### Já existe e será REUSADO (não reescrever)
| Asset | Arquivo | Papel no plano |
|---|---|---|
| Estratégia beta disciplinado (backtest) | `simulation/beta_portfolio.py` | Fonte da **lógica de pesos** (`weights_disciplined`): vol-target + portão de regime + teto cripto, sem look-ahead. Será portada para um construtor de pesos vivo. |
| Orquestração viva | `agents/monitor.py`, `orchestration/bus_orchestrator.py`, `orchestration/factory.py` | Loop agendado (APScheduler, UTC/DST-safe) Planner→Executor→Feedback com reconciliação periódica. O rebalance entra como uma estratégia/etapa nova nesse pipeline. |
| Execução idempotente | `agents/executor.py` | Único componente que escreve no broker; idempotência por `client_order_id`, retry/backoff, anti-rajada, kill switch. **Reusar como está.** |
| Broker (PAPER) | `broker/alpaca_broker.py`, `broker/base.py` | `get_account()` (equity/cash/buying_power), `get_positions()`, `get_last_price()`, `get_bars()`. **Suficiente para NAV e rebalance.** |
| Gestão de risco | `risk/portfolio_guard.py`, `risk/manager.py`, `config/risk.py` | Circuit breakers (daily loss, max DD pico-a-vale, per-symbol, heat) com high-water **persistido**. Aplicar limites próprios do beta. |
| Portão de regime | `feedback/regime.py` (`classify_regime`), usado em `beta_portfolio._regime_derisk_flags` | Des-arrisca ativo (peso→0) em TREND_DOWN/HIGH_VOL ou preço < SMA200. **Mesma função** roda no backtest e poderá rodar vivo. |
| Persistência/auditoria | `data/db.py` (SQLite WAL), `data/trade_logger.py` (`trade_log`), `data/audit_log.py` (`audit_log`), `feedback/decision_log.py` (`decisions`) | Trilha imutável de trades + "porquê" de cada decisão. Base do track record. |
| Kill switch | `core/kill_switch.py` | Env var `KILL_SWITCH` OU arquivo sentinela `KILL_SWITCH`. Para tudo. **Reusar.** |
| Reconciliação | `orchestration/reconcile.py` | Broker = fonte de verdade no boot e a cada N ciclos. **Reusar.** |
| Dashboard | `dashboard/queries.py`, `dashboard/server.py` | Hoje só agrega P&L realizado das `decisions` fechadas. Precisa ganhar curva de equity/NAV e métricas since-inception. |

### Lacunas (o que FALTA implementar) — fundamento de todas as fases
1. **NAV/equity time-series não é persistido.** O dashboard só tem `_pnl_curve` (P&L realizado de trades fechados), não a **curva de equity mark-to-market diária**. **Sem isso não há track record.** → Tabela nova `nav_history` (Fase 1).
2. **Não há rebalancer de carteira-alvo.** Todo o sistema atual é "estratégia por símbolo emite `OrderIntent`". O beta disciplinado é **carteira de pesos-alvo rebalanceada mensalmente**. Falta um componente "diferença entre pesos-alvo e posições atuais → intents". → `strategies/beta_rebalancer.py` (Fase 2).
3. **Métricas since-inception e benchmark vivo não são calculados/reportados.** O `beta_portfolio.py` calcula Sharpe/Sortino/MaxDD/Calmar **no backtest**; falta o mesmo **sobre o NAV real**, vs. benchmark (SPY e 60/40) capturados em paralelo. → `reporting/track_record.py` (Fase 1).
4. **Pesos-alvo do backtest dependem de yfinance/pandas;** o caminho vivo deve derivar pesos das **barras do broker** (`get_bars`) ou de um snapshot diário, com a MESMA lógica, sem look-ahead. → adaptador em Fase 2.
5. **Limites de risco hoje são calibrados p/ trading de sinais** (max DD 20%, per-symbol 20%). Beta disciplinado quer DD-alvo menor e teto por nome coerente com vol-target. → Fase 3.

---

## FASE 1 — SISTEMA DE TRACK RECORD (o ativo central p/ AUM)

**Objetivo:** registrar e reportar, de forma **auditável e à prova de cherry-picking**, a curva de equity desde o início, retornos mensais, Sharpe/Sortino/MaxDD/Calmar vs benchmark, períodos de drawdown e since-inception — **sem reescrever histórico**. Esta fase é PRÉ-REQUISITO de tudo: começa a rodar ANTES mesmo do rebalancer (captura o NAV de qualquer atividade na conta paper, inclusive período de "conta parada"/baseline).

### Princípios anti-cherry-picking (não-negociáveis, viram critérios de aceite)
- **Since-inception obrigatório:** todo relatório mostra a janela COMPLETA desde o 1º dia de NAV. Janelas customizadas só são permitidas como *adendo*, nunca substituindo a since-inception.
- **Append-only / imutável:** `nav_history` e `trade_log` nunca sofrem UPDATE/DELETE de linhas passadas (só INSERT). Backfill proibido após o fato (uma linha por dia, gravada naquele dia).
- **Benchmark capturado em paralelo, mesmo dia:** o benchmark (SPY total-return e 60/40) é snapshotado no MESMO tick do NAV, para não haver "escolha" retroativa de benchmark.
- **Marca d'água de drawdown persistida** (já existe `risk_peak_equity` em `state`): o MaxDD reportado vem do pico real vivido, não recomputado de forma conveniente.
- **Hash/encadeamento opcional** (ver T1.6): cada snapshot guarda o hash do anterior → adulteração detectável.

### Tarefas

**T1.1 — Tabela `nav_history` (snapshot diário de equity mark-to-market)**
- **Entregável:** nova tabela criada por um módulo `reporting/nav_repo.py` (padrão `feedback/decision_log.py`: `CREATE TABLE IF NOT EXISTS` no mesmo `data/trading.sqlite`, WAL, conexão compartilhada). Colunas mínimas: `ts` (ISO UTC), `date` (YYYY-MM-DD, **UNIQUE** — um snapshot oficial por dia), `equity`, `cash`, `long_market_value`, `gross_exposure`, `net_exposure`, `regime` (regime agregado do dia), `bench_spy` (preço SPY total-return), `bench_6040` (NAV sintético 60/40), `source` ("eod"/"intraday"), `prev_hash`, `row_hash`.
- **Dependência:** `data/db.py`, `broker.get_account()`, `broker.get_positions()`, `broker.get_last_price("SPY"/"IEF")`.
- **Executor:** Construtor (dev).
- **Critério de aceite:** ao chamar `record_snapshot()` duas vezes no mesmo dia, a 2ª faz UPSERT do snapshot do dia (sem criar 2 linhas) e **nunca** altera linhas de dias anteriores; teste unitário cobre idempotência diária e a unicidade de `date`.
- **Gate:** —

**T1.2 — Captura de NAV no fim de cada dia de pregão (EOD)**
- **Entregável:** um passo no pipeline do bus (`orchestration/bus_orchestrator.py`) OU um job APScheduler dedicado em `agents/monitor.py` que, no **fechamento** (detectado via `broker.get_clock().next_close`/`is_open` transição, ou um tick agendado pós-close), grava UM snapshot oficial em `nav_history`. Snapshots intradiários adicionais permitidos com `source="intraday"` mas só o EOD conta para o track record.
- **Dependência:** T1.1; `MarketClock`; agendador existente.
- **Executor:** Construtor (dev).
- **Critério de aceite:** após rodar `main.py` por 1 dia de pregão (paper), existe exatamente 1 linha EOD em `nav_history` para o dia; o `equity` bate com `broker.get_account().equity` no fechamento (tolerância de arredondamento). Sobrevive a restart (não duplica).
- **Gate:** —

**T1.3 — Benchmark vivo (SPY total-return + 60/40 sintético)**
- **Entregável:** no mesmo snapshot diário, gravar `bench_spy` (fechamento ajustado de SPY) e `bench_6040` (NAV de uma carteira sintética 60% SPY / 40% IEF rebalanceada mensalmente — o mesmo benchmark B do backtest). Função `reporting/benchmarks.py` que mantém o NAV sintético 60/40 incremental (sem look-ahead: usa o preço do próprio dia para marcar, rebalance no 1º pregão do mês).
- **Dependência:** T1.1, T1.2; `broker.get_last_price`/`get_bars`.
- **Executor:** Construtor (dev). Lógica de pesos 60/40 já existe em `beta_portfolio.SIXTY_FORTY` / `weights_sixty_forty` — **reusar a convenção**.
- **Critério de aceite:** a série `bench_6040` reconstruída offline a partir dos snapshots reproduz, dentro de tolerância, um 60/40 mensal; teste com preços sintéticos.
- **Gate:** —

**T1.4 — Calculadora de métricas since-inception sobre o NAV REAL**
- **Entregável:** `reporting/track_record.py` que lê `nav_history` e produz: CAGR, Sharpe, Sortino, MaxDD (pico-a-vale, datas), Calmar, vol anualizada, retornos **mensais** (tabela mês×ano), pior mês/pior ano, %positivo de meses, beta e correlação vs SPY, **excess return e tracking error vs benchmark**, e **drawdown table** (top-5 quedas: início, vale, recuperação, duração). **Reusar as fórmulas já testadas** de `simulation/beta_portfolio.py` (`_annualized_sharpe`, `_annualized_sortino`) e `simulation/metrics.max_drawdown` — extrair para `reporting` ou importar, evitando reimplementar.
- **Dependência:** T1.1–T1.3.
- **Executor:** Construtor (dev). Funções estatísticas: reuso de `simulation/`.
- **Critério de aceite:** rodando sobre uma série de NAV sintética conhecida, as métricas batem com os valores fechados calculados à mão (teste unitário com tolerância). Since-inception SEMPRE presente no output.
- **Gate:** **G-TR1:** as métricas são reprodutíveis a partir do `nav_history` por um terceiro rodando o mesmo script (determinístico, sem estado oculto).

**T1.5 — Relatório/Dashboard de track record**
- **Entregável (a):** relatório de texto/markdown `reporting/track_record.py --report` que imprime a tabela since-inception + mensal + drawdown + vs-benchmark, salvando em `data/track_record.txt` (mesmo padrão dos `*_verdict.txt`). **Entregável (b):** estender `dashboard/queries.py` com `_nav_curve()` e `_track_record_kpis()` (lendo `nav_history` em modo SOMENTE-LEITURA, como o resto do dashboard) e `dashboard/server.py` para exibir: curva de equity vs benchmark, heatmap de retornos mensais, drawdown underwater plot, e o bloco de KPIs since-inception.
- **Dependência:** T1.4.
- **Executor:** Construtor (dev).
- **Critério de aceite:** o dashboard mostra a curva de equity desde o 1º snapshot e a since-inception bate com o relatório de texto (mesma fonte, `nav_history`). Disclaimer fixo no rodapé (ver T4.x): *"Rentabilidade passada não representa garantia de resultados futuros. Track record de simulação/paper trading — não houve gestão de recursos de terceiros."*
- **Gate:** —

**T1.6 — Imutabilidade e prova de integridade (anti-adulteração)**
- **Entregável:** encadeamento por hash em `nav_history` (`row_hash = sha256(date|equity|...|prev_hash)`), mais um verificador `reporting/verify_chain.py` que detecta qualquer linha alterada/removida. Snapshot diário do banco para WORM/append-only externo **[DECISÃO]** (ver abaixo).
- **Dependência:** T1.1.
- **Executor:** Construtor (dev).
- **Critério de aceite:** alterar uma linha antiga do `nav_history` faz `verify_chain.py` falhar apontando a 1ª linha quebrada.
- **Gate:** **G-TR2 (gate de credibilidade do track record):** cadeia de hash íntegra + backup externo configurado.
- **[DECISÃO do usuário]:** onde guardar o backup append-only imutável do `nav_history`/banco (ex.: commit diário em repositório git privado dedicado; bucket S3 com Object Lock; planilha exportada e versionada). Recomendação: **commit git diário automatizado de um export CSV de `nav_history`** — barato, à prova de reescrita (histórico git), e auditável por terceiros. *(O Pesquisador deve confirmar se, para fins de captação futura, há expectativa de auditoria independente — ver Fase 4.)*

### Entregável macro da Fase 1
Curva de equity viva, auditável, append-only, com benchmark, métricas since-inception e relatório/dashboard. **A partir daqui, todo dia que o sistema roda em paper conta para o track record.**

---

## FASE 2 — DEPLOY EM PAPER (ligar a carteira de beta disciplinado)

**Objetivo:** ligar a carteira de beta disciplinado no Monitor/orchestration sobre Alpaca **paper**, reusando a infra: rebalance mensal, vol-target, portão de regime.

> Nota de arquitetura: o sistema atual roda estratégias **por símbolo** que emitem `OrderIntent`. Beta disciplinado é uma **carteira de pesos-alvo**. A peça nova é um *rebalancer* que traduz `pesos-alvo × equity` em ordens de diferença. Ele é registrado como mais uma estratégia no Planner (a infra de execução/risco/auditoria fica intacta).

### Tarefas

**T2.1 — Construtor de pesos-alvo VIVO (porta da lógica do backtest)**
- **Entregável:** `strategies/beta_weights.py` que, dado o histórico de fechamentos por ativo (via `broker.get_bars(symbol, limit=≥250, timeframe="1Day")` para cobrir SMA200 + vol60), produz o **vetor de pesos-alvo do dia** com a MESMA lógica de `beta_portfolio.weights_disciplined`: inv-vol → teto cripto → portão de regime (SMA200 + `classify_regime`) → vol-target 10% a.a. → gross ≤ 1 (resto caixa). **Reuso direto** de `feedback.regime.classify_regime` e dos parâmetros padrão (`SMA_WINDOW`, `VOL_LOOKBACK`, `VOL_TARGET_ANNUAL`, `CRYPTO_MAX_WEIGHT`).
- **Dependência:** `broker.get_bars` (confirmar que retorna ≥250 barras diárias por ativo). Universo: `SPY, QQQ, TLT, IEF, GLD, SLV, BTC-USD, ETH-USD` (= `beta_portfolio.UNIVERSE`). **[DECISÃO]:** confirmar tickers de cripto disponíveis na Alpaca paper (ex.: `BTC/USD`, `ETH/USD`) e mapear ↔ tickers yfinance do backtest.
- **Executor:** Construtor (dev).
- **Critério de aceite:** dado o MESMO histórico de preços, `beta_weights.target_weights(hist)` reproduz (dentro de tolerância numérica) `weights_disciplined(...).iloc[-1]` do backtest para uma data fixa. **Teste de paridade backtest↔vivo é o critério central** (garante que o que foi validado é o que roda).
- **Gate:** **G-PAPER1:** paridade backtest↔vivo provada por teste.

**T2.2 — Rebalancer: pesos-alvo → `OrderIntent`s de diferença**
- **Entregável:** `strategies/beta_rebalancer.py` (implementa a interface `strategies/base.py`). Entrada: pesos-alvo (T2.1) + posições atuais (`broker.get_positions`) + equity (`broker.get_account`). Saída: lista de `OrderIntent` (buy/sell) para mover cada nome do peso atual ao peso-alvo. Regras: (a) **banda de no-trade** (só rebalanceia se |Δpeso| > banda, ex. 1 ponto percentual — corta turnover/custo, igual ao espírito do backtest); (b) `strategy="beta_disciplinado"` em todo intent (rastreabilidade no `trade_log`/`decisions`); (c) ordens de **venda primeiro, compra depois** (libera buying power); (d) respeita `MAX_GROSS=1` (sem alavancagem).
- **Dependência:** T2.1; `core/models.OrderIntent`; Planner/Executor existentes.
- **Executor:** Construtor (dev).
- **Critério de aceite:** com posições mockadas e pesos-alvo dados, gera o conjunto correto de intents (testes: do zero monta a carteira; de uma carteira já alinhada gera **zero** intents dentro da banda; des-arriscar um ativo gera venda total dele).
- **Gate:** —

**T2.3 — Gatilho de rebalance mensal (sem look-ahead vivo)**
- **Entregável:** lógica de cadência que dispara o rebalancer **uma vez por mês** (1º pregão do mês — `beta_portfolio._rebalance_mask` define a convenção) e, fora disso, dispara apenas **rebalance de des-risco** quando o portão de regime vira para um ativo já posicionado (proteção de crash entre rebalances) **[DECISÃO]**. Estado do "último rebalance" persistido em `state` (sobrevive a restart).
- **Dependência:** T2.2; `data/state_repo.py`.
- **Executor:** Construtor (dev).
- **Critério de aceite:** em 2 ticks no mesmo mês, só o 1º (após virada de mês) rebalanceia; teste cobre a persistência da marca de mês.
- **[DECISÃO do usuário]:** permitir **des-risco intra-mês** (vender quando regime/SMA viram) mas **re-risco só no rebalance mensal**? (Recomendado: sim — assimetria que protege o drawdown sem aumentar turnover, coerente com "drawdown menor".) Ou rebalance estritamente mensal puro?
- **Gate:** —

**T2.4 — Registro no Planner + decisão logada**
- **Entregável:** registrar `BetaRebalancerStrategy` na lista `strategies` em `main.py` (`build_app`), atrás de uma flag de config (`config/settings.py` / `.env`: `BETA_PORTFOLIO_ENABLED`). Cada intent de rebalance vira uma `Decision` no `DecisionLog` (com `regime`, `reference_price`, `context`={pesos-alvo, pesos-atuais, motivo: rebalance mensal vs. des-risco}) — reusa o fluxo de feedback que já existe.
- **Dependência:** T2.2, T2.3; `main.py`, `feedback/decision_log.py`.
- **Executor:** Construtor (dev).
- **Critério de aceite:** `main.py --once` em paper com a flag ligada produz intents de rebalance, executados pelo Executor, com linhas em `trade_log`, `orders`, `decisions` e `audit_log` rastreáveis por `client_order_id`.
- **Gate:** **G-PAPER2:** um ciclo `--once` completo em paper executa o 1º rebalance, com auditoria íntegra e reconciliação OK.

**T2.5 — Universo, custos e fracionários**
- **Entregável:** decidir execução fracionária (Alpaca paper suporta fractional shares para ETFs/ações — necessário para pesos finos) **[DECISÃO/verificação]**; cripto via Alpaca crypto. Documentar custos reais assumidos (equities ~0 comissão, cripto ~spread) e conferir vs. os custos do backtest (`EQUITY_BASE`/`CRYPTO_BASE`).
- **Dependência:** `broker/alpaca_broker.py`.
- **Executor:** Construtor (dev) + **[DECISÃO]** do usuário sobre habilitar cripto no paper desde o início ou começar só com ETFs (cripto adiciona ruído e custo).
- **Critério de aceite:** ordens fracionárias aceitas pelo broker paper; se não suportado para algum ativo, o rebalancer arredonda para baixo e registra o resíduo em caixa (sem alavancar).
- **Gate:** —

### Entregável macro da Fase 2
Carteira de beta disciplinado **rodando em paper** dentro do Monitor, rebalanceando mensalmente, des-arriscando por regime, com cada ordem auditada e alimentando o `nav_history` da Fase 1.

---

## FASE 3 — CADÊNCIA E DISCIPLINA (risco, kill switch, reconciliação)

**Objetivo:** garantir que a operação seja disciplinada e segura: limites do `portfolio_guard` aplicados ao perfil beta, kill switch, reconciliação periódica.

### Tarefas

**T3.1 — Calibrar limites de risco para o perfil BETA**
- **Entregável:** ajustar `config/risk.py` para o beta disciplinado: `max_drawdown_pct` (halt de risco novo) coerente com DD-alvo do backtest + folga (ex.: se o backtest mostra MaxDD da estratégia ~X%, setar o halt em ~1.3–1.5×X); `max_per_symbol_pct` coerente com o teto efetivo de vol-target (ex.: ETF até ~35–40%, cripto travada no `CRYPTO_MAX_WEIGHT=5%`); `daily_loss_limit_pct` apropriado a uma carteira de baixa vol. **[DECISÃO do usuário]:** os números finais (DD-halt, per-symbol) — devem sair do **resultado do backtest** que está rodando em `beta_portfolio.py`. *Aguardar o veredito do backtest (`data/beta_verdict.txt`) para fixar.*
- **Dependência:** veredito de `simulation/beta_portfolio.py`; `risk/portfolio_guard.py`.
- **Executor:** **[DECISÃO]** usuário define os alvos; Construtor implementa.
- **Critério de aceite:** os limites estão em `config/risk.py`, versionados, e o `PortfolioRiskGuard` é instanciado com eles em `main.py`; teste cobre que ultrapassar o DD-halt bloqueia **abertura de risco novo** (mas não as vendas de des-risco — coerente com `can_open` que só barra abertura).
- **Gate:** **G-RISK1:** limites de risco do beta documentados e ligados; halt testado.

**T3.2 — Distinguir "des-risco" de "abrir risco" no guard**
- **Entregável:** garantir que, com o guard em halt (DD/daily-loss), o rebalancer **ainda pode VENDER** para des-arriscar (o `portfolio_guard.can_open` já só barra abertura; confirmar que vendas de rebalance não passam por `can_open`). Documentar o caminho.
- **Dependência:** T2.2, T3.1.
- **Executor:** Construtor (dev).
- **Critério de aceite:** teste: guard halted → rebalancer de des-risco gera/admite vendas; tentativas de compra são barradas com motivo logado em `audit_log`.
- **Gate:** —

**T3.3 — Kill switch operacional + runbook**
- **Entregável:** runbook curto (no `docs/` ou README) de como engajar/desengajar o kill switch (`touch KILL_SWITCH` / `KILL_SWITCH=1`), o que acontece (Executor para toda submissão; posições ficam como estão; reconciliação no próximo boot), e checklist de incidente. Reusa `core/kill_switch.py` (sem código novo de produção).
- **Dependência:** `core/kill_switch.py`.
- **Executor:** Construtor (doc).
- **Critério de aceite:** seguindo o runbook, `touch KILL_SWITCH` faz o próximo ciclo não enviar nenhuma ordem (verificável em `audit_log`: `order_blocked_killswitch`).
- **Gate:** —

**T3.4 — Reconciliação periódica + alerta de divergência**
- **Entregável:** confirmar que o bus reconcilia a cada `reconcile_every` ciclos (`orchestration/factory.py`) e que divergências (`ORPHAN_POSITION`, `STALE_LOCAL_POSITION`, `UNKNOWN_LOCAL_ORDER`) viram alerta visível (log CRÍTICO + linha em `audit_log` já existem; adicionar, se desejado, notificação **[DECISÃO]** — e-mail/desktop). Reconciliação no boot já roda em `main.build_app`.
- **Dependência:** `orchestration/reconcile.py`.
- **Executor:** Construtor (dev/doc).
- **Critério de aceite:** injetar divergência (posição local que o broker não tem) → reconciliação corrige a favor do broker e loga a issue.
- **Gate:** —

**T3.5 — Disciplina de "não tocar": congelar parâmetros durante a janela de track record**
- **Entregável:** política escrita: durante a janela de track record limpo (Gate G5), os parâmetros do beta (SMA200, vol-alvo 10%, lookback 60, teto cripto 5%, banda de no-trade) **não mudam** — qualquer mudança reinicia o relógio do track record (anti-overfitting/anti-curva-de-equity-cosmética). Mudanças exigem registro em `audit_log` com justificativa e nova data de início de "track v2".
- **Dependência:** —
- **Executor:** **[DECISÃO]** usuário acorda a disciplina; Construtor documenta.
- **Critério de aceite:** documento de política existe e está linkado no README; uma mudança de parâmetro gera entrada de auditoria e nota no relatório de track record.
- **Gate:** **G-RISK2:** política de congelamento aceita pelo usuário.

### Entregável macro da Fase 3
Operação disciplinada: limites de risco do beta ligados e testados, kill switch + runbook, reconciliação com alerta, e disciplina de parâmetros congelados.

---

## FASE 4 — RAMPA DE AUM (Brasil) — pré-requisitos legais

> **Conteúdo confirmado pelo Pesquisador (web habilitada, 2026).** Rótulos: **[CONFIRMADO]** (texto legal/fonte autoritativa lida), **[PROVÁVEL]** (fonte secundária confiável), **[INCERTO — advogado]**. **Isto não é aconselhamento jurídico; validar com advogado/contador especializado em mercado de capitais.**

### O caminho de menor atrito (sequência recomendada)
**Clube de Investimento (você como cotista-gestor não remunerado) → tirar CGA → Gestora de Recursos (PJ) registrada na CVM.** Raciocínio: o clube é o único veículo que permite, **hoje e legalmente**, gerir dinheiro real de terceiros sem credencial da CVM (desde que **sem remuneração** pela gestão) e produzir **cota oficial auditável registrada na B3** — exatamente a ponte para sair do paper. A CGA destrava a remuneração e a escala; a gestora é o destino que monetiza.

### Fato-âncora (gotcha central) — **[CONFIRMADO]**
**Gerir carteira de terceiros sem autorização da CVM é crime, AINDA QUE GRATUITO.** Lei 6.385/76, Art. 27-E: *"Exercer, ainda que a título gratuito, no mercado de valores mobiliários, a atividade de administrador de carteira... sem estar autorizado ou registrado — Pena: detenção de 6 meses a 2 anos, e multa."* Art. 23 exige autorização prévia. Sanção administrativa: multa que pode chegar a 30% do valor da operação irregular (Lei 6.385, Art. 11); a CVM aplica de fato (ex.: multa de R$ 340 mil por administração irregular em 2024).
- **Implicação para o plano:** **NÃO** gerir, nem informalmente, dinheiro de terceiros antes do veículo/credencial. A única forma legal sem credencial é o **clube, como cotista-gestor não remunerado**.

### DEGRAU 1 — Clube de Investimento

**Norma — [CONFIRMADO]:** **Resolução CVM nº 11/2020** (alterada pela Res. CVM 179/23), que revogou a Instrução CVM 494/2011. Demonstrações: Res. CVM 12/2020.

| Item | Fato | Confiança |
|---|---|---|
| Cotistas | **mínimo 3, máximo 50**, e **somente pessoas naturais** (PJ não pode ser cotista) — Art. 2º | [CONFIRMADO] |
| Concentração | nenhum cotista pode ter **> 40%** das cotas — Art. 7º | [CONFIRMADO] |
| Administrador | **deve ser** corretora, distribuidora, banco de investimento ou banco múltiplo com carteira de investimento — Art. 19. **Você não pode ser o administrador.** | [CONFIRMADO] |
| Gestor | três caminhos (Art. 20): (I) o próprio administrador autorizado; (II) PN/PJ contratada **autorizada pela CVM**; **(III) um ou mais cotistas eleitos — SEM exigência de credenciamento na CVM** | [CONFIRMADO] |
| **Remuneração do cotista-gestor** | Art. 20, §2º: o cotista-gestor **(i)** só pode ter 1 clube e **(ii) é VEDADO receber qualquer remuneração** pela gestão. Vedado também gestor que seja assessor de investimento | [CONFIRMADO] |
| Taxa de adm e **de performance** | permitidas no estatuto (Art. 23, V e VI) — **mas** se o gestor for cotista não credenciado, ele **não pode** receber (interação com Art. 20 §2º) | [CONFIRMADO] |
| Registro | constituído por ato do administrador; funcionamento depende de **registro em mercado organizado (B3)**, não diretamente na CVM — Art. 4º | [CONFIRMADO] |
| Custódia | ativos em conta do clube; custodiante qualificado; só ativos em mercados autorizados — Arts. 21/30 | [CONFIRMADO] |
| Custos | patrimônio inicial usual **~R$ 500 mil** (prática comercial dos administradores, **não** exigência da Res. 11); taxa de adm + corretagem + PIS/COFINS/ISS sobre taxas; IR 15% no resgate sobre o ganho | [PROVÁVEL] |

**Leitura prática:** você pode, **legalmente e barato**, montar um clube com um administrador parceiro (corretora/DTVM), ser o **cotista-gestor (Art. 20, III)** e gerar **track record real auditável** sob o CNPJ do clube — **sem credencial e sem capital próprio mínimo**, mas **sem ser remunerado** pela gestão nesse formato. Até 50 cotistas PF, teto de 40% por cotista.

#### Tarefas — Degrau 1
- **T4.1** Cotar 2–3 administradores (corretoras/DTVMs que administram clubes) — patrimônio mínimo comercial, taxa de adm, setup, prazos. **Executor:** usuário. **Aceite:** 3 cotações comparadas. **[CONFIRMAR com administrador]** o PL mínimo e taxas.
- **T4.2** Validar com advogado/contador: tributação do clube; redação do estatuto (taxas, política de investimento compatível com o beta disciplinado), e a vedação de remuneração ao cotista-gestor (Art. 20 §2º). **Executor:** usuário + advogado. **Aceite:** parecer escrito.
- **T4.3** Definir cotistas iniciais (≥3 PF, respeitando 40%) e patrimônio. **[DECISÃO do usuário].**
- **T4.4** Política de investimento do clube espelhando o beta disciplinado (universo, vol-target, portão de regime, sem alavancagem) — alinhar o que o sistema executa com o que o estatuto permite. **Aceite:** política do estatuto ⊇ o que o bot faz.
- **Gate:** **G-AUM1:** parecer jurídico + administrador escolhido + estatuto coerente com a estratégia, ANTES de captar qualquer cotista.

### DEGRAU 2 — Certificação CGA (destrava remuneração e a gestora)

**[CONFIRMADO]:** para ser **gestor de recursos autorizado** (pessoa natural), a Res. CVM 21/2021 exige curso superior + aprovação em **certificação do Anexo A**: **CGA (ANBIMA)** ou **CFA Level III** ou **CAIA (Exams 1+2 do Final Level)**. A **CFG (ANBIMA)** é pré-requisito da CGA mas **sozinha não habilita**. **Atenção [CONFIRMADO]:** a dispensa por experiência (Art. 3º §1º) exige 7 anos em gestão profissional, e **"atuar como investidor" NÃO conta** (§2º) — ou seja, **seu track record pessoal não dá direito à dispensa**; planeje **tirar a CGA**.

#### Tarefas — Degrau 2
- **T4.5** Plano de estudo CFG→CGA (cronograma, custos das provas ANBIMA). **Executor:** usuário. **[DECISÃO]:** começar a CGA em paralelo ao clube (recomendado — leva meses).
- **Gate:** **G-AUM2:** CGA aprovada (pré-requisito da gestora e de remuneração pela gestão).

### DEGRAU 3 — Gestora de Recursos (PJ) registrada na CVM

**Norma — [CONFIRMADO]:** **Resolução CVM nº 21/2021**. Atividade **privativa de autorizado pela CVM** (Art. 2º). Duas categorias: **gestor de recursos** (decide investimentos) e **administrador fiduciário** (controladoria/escrituração/supervisão). Modelo de mercado para gestora independente: registrar-se como **gestor de recursos** e **contratar administrador fiduciário terceirizado** (corretora/DTVM/banco).

| Item | Fato | Confiança |
|---|---|---|
| Requisitos PJ | sede no Brasil; objeto social de gestão + CNPJ; **diretor de gestão** (autorizado, com CGA); **diretor de risco/compliance segregado** (pode ser 1 pessoa, mas **não** o gestor) — Art. 4º. Mínimo realista: **2 pessoas** | [CONFIRMADO] |
| **Capital mínimo do gestor** | a Res. 21 **NÃO impõe capital prudencial ao gestor de recursos**. O PL de 0,20%/R$ 550 mil é do **administrador fiduciário**. O "0,02%/R$ 300 mil" do Anexo E é **declaração informativa**, não exigência | [PROVÁVEL→CONFIRMADO; confirmar com advogado] |
| Prazo | pedido à **SIN** com 60 dias de análise; **deferimento automático** se a SIN silenciar (Art. 7º §10). Na prática, **4–6 meses** com a montagem | [CONFIRMADO]/[PROVÁVEL] |
| Autorregulação | **Código ANBIMA de Administração e Gestão** — na prática indispensável ("selo ANBIMA") para captar; exige estrutura de compliance/PLD e certificação da equipe | [CONFIRMADO]/[PROVÁVEL] |
| PLD-FT | estrutura de prevenção à lavagem (Res. CVM 50/2021) | [CONFIRMADO] |

#### Tarefas — Degrau 3
- **T4.6** Estruturar a PJ (CNAE/objeto social, 2 diretores com segregação gestão × risco/compliance), com advogado. **Aceite:** minuta de constituição.
- **T4.7** Selecionar administrador fiduciário terceirizado (cotação). **Aceite:** 2 cotações.
- **T4.8** Montar pacote de registro SIN (Anexo C/PJ, Formulário de Referência Anexo E) + adesão ANBIMA. **Aceite:** protocolo na CVM.
- **Gate:** **G-AUM3:** registro de gestor de recursos deferido pela CVM + adesão ANBIMA + administrador fiduciário contratado, ANTES de cobrar taxa de adm/performance ou captar de forma escalável.

### TRACK RECORD para captação — o que o mercado espera
- **[PROVÁVEL]** Alocadores valorizam **track record de dinheiro real, sob veículo regulado (clube/fundo), com administrador e auditor independentes**. **Paper trading é fortemente descontado** (sem custódia/execução real/auditoria) → reforça por que o **clube é a ponte**: gera cota oficial calculada pelo administrador e divulgada na B3.
- **GIPS [INCERTO]:** não é exigência legal no Brasil; padrão internacional voluntário (relevante p/ institucional/estrangeiro). ANBIMA tem regras próprias de apresentação de rentabilidade (padrão de fato local). **Verificar com a área de distribuição.**
- **Divulgação de rentabilidade [CONFIRMADO em parte]:** disclaimer obrigatório *"A rentabilidade obtida no passado não representa garantia de resultados futuros"*; vedado assegurar/sugerir garantia de resultado. **Período mínimo** de divulgação (6 vs 12 meses) e material consistente com a lâmina (Res. 175 p/ fundos) — **[INCERTO — advogado]** a redação/timing exatos por tipo de veículo. **Solicitar/ofertar cotas publicamente é atividade de distribuição regulada**, separada da gestão — não sair captando publicamente sem o arcabouço.

### Lista "verificar com advogado/contador" (do Pesquisador)
1. Inexistência de capital mínimo para **gestor** de recursos PJ (só fiduciário tem 0,20%/R$ 550k).
2. Tributação do clube (IR 15% no resgate; PIS/COFINS/ISS sobre taxas) e estrutura de remuneração respeitando Art. 20 §2º.
3. Redação e timing exatos do disclaimer e do período mínimo (6 vs 12 meses) por veículo (Res. 175 / regras ANBIMA).
4. Confirmar que a dispensa de graduação/certificação (Art. 3º §1º) **não** se aplica a você (investidor não conta) → tirar CGA.
5. Regras da B3 para registro/funcionamento do clube e exigências comerciais do administrador parceiro.
6. Estrutura mínima de PLD/compliance (Res. CVM 50/2021) para a gestora.
7. Crime do Art. 27-E — nenhuma gestão remunerada de terceiros (nem "informal/gratuita") antes da autorização.

### Fontes (Pesquisador)
- Res. CVM 11/2020 (clubes): https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/resolucoes/anexos/001/resol011consolid.pdf
- Res. CVM 21/2021 (gestão de carteiras): https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/resolucoes/anexos/001/resol021consolid.pdf
- Res. CVM 175/2022, Anexo I (fundos/divulgação): https://conteudo.cvm.gov.br/export/sites/cvm/legislacao/resolucoes/anexos/100/resol175consolid_Anexo01.pdf
- Lei 6.385/76 (Arts. 23 e 27-E): https://www.planalto.gov.br/ccivil_03/leis/l6385.htm
- CVM — multa R$ 340 mil por administração irregular (2024): https://www.gov.br/cvm/pt-br/assuntos/noticias/2024/cvm-multa-em-r-340-mil-acusado-por-exercicio-irregular-de-administracao-de-carteira-de-valores-mobiliarios
- ANBIMA — clubes: https://comoinvestir.anbima.com.br/noticia/o-que-sao-clubes-de-investimento/
- ANBIMA — Código de Administração e Gestão (ART): https://www.anbima.com.br/pt_br/autorregular/codigos/administracao-de-recursos-de-terceiros.htm
- ANBIMA Edu — CGA: https://anbimaedu.com.br/certificacao/cga
- B3 — clubes de investimento: https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-a-vista/clubes-de-investimento/sobre-clubes-de-investimento/

---

## FASE 5 — GATES (track record mínimo antes de dinheiro real e de terceiros)

> Gates são **portas de ir/não-ir**, mecânicos e auditáveis. Não há captação nem dinheiro real sem passar o gate correspondente.

### GATE G4 — De paper para **dinheiro real PRÓPRIO** (você arrisca seu capital)
Critérios (todos obrigatórios):
- **≥ 6 meses corridos** de paper **limpo** com a versão CONGELADA do beta (G-RISK2). *Mínimo absoluto; ver recomendação abaixo.*
- `nav_history` com cadeia de hash íntegra (**G-TR2**) e relatório since-inception reprodutível (**G-TR1**).
- **Zero** incidentes não resolvidos: nenhuma divergência de reconciliação aberta; nenhum bug de execução/idempotência sem correção.
- Comportamento **coerente com o backtest**: o drawdown real não excede materialmente o MaxDD do backtest na mesma janela de mercado; o portão de regime de fato des-arriscou quando deveria (verificável no `audit_log`).
- Kill switch testado (G-RISK runbook).
- **[DECISÃO do usuário]:** o número exato de meses (recomendação: **6 meses como mínimo, 12 meses como alvo** — captura ao menos um trimestre de stress e dá amostra mensal mínima para Sharpe ter sentido). Quanto mais limpo o histórico, mais valioso o ativo de captação.

### GATE G5 — De dinheiro próprio para **capital de terceiros** (clube)
Critérios (todos obrigatórios, ALÉM de G4):
- **≥ 12 meses** (recomendado **18–24**) de track record contínuo, **incluindo dinheiro real próprio** (G4 cumprido e mantido), sem reescrita de histórico.
- **G-AUM1** cumprido (parecer jurídico + administrador + estatuto coerente) — clube é o **primeiro** veículo de terceiros.
- Track record apresentável com disclaimer correto (Fase 4) e métricas since-inception vs benchmark (Fase 1).
- Política de congelamento de parâmetros respeitada durante toda a janela (mudança = reinicia relógio).
- **Nenhuma** captação/gestão remunerada de terceiros antes de G-AUM1 (risco penal Art. 27-E).
- **[DECISÃO do usuário]:** meses mínimos para captar — **12 como piso, 18–24 como alvo** (alocadores descontam históricos curtos e de paper; um histórico real de clube de ≥1 ano é o mínimo crível).

### GATE G6 — De clube para **gestora CVM** (escala/remuneração)
- **G-AUM2** (CGA) + **G-AUM3** (registro CVM + ANBIMA + fiduciário) cumpridos.
- Track record do clube (cota oficial B3) com **≥ 12 meses** de dinheiro real sob veículo regulado.
- **[DECISÃO do usuário]:** quando migrar (custo da gestora só se justifica com AUM/captação que pague a estrutura de 2 diretores + administrador fiduciário + ANBIMA).

### Tabela-resumo dos gates
| Gate | Porta | Pré-requisitos-chave |
|---|---|---|
| G-TR1/G-TR2 | Track record crível | Métricas reprodutíveis + cadeia de hash + backup externo |
| G-PAPER1/2 | Beta vivo em paper | Paridade backtest↔vivo + ciclo `--once` auditado |
| G-RISK1/2 | Operação disciplinada | Limites do beta ligados/testados + política de congelamento |
| **G4** | Paper → dinheiro **próprio** | ≥6m (alvo 12m) paper limpo + integridade + coerência c/ backtest |
| **G5** | Próprio → **terceiros (clube)** | ≥12m (alvo 18–24m) real + parecer jurídico + estatuto + administrador |
| **G6** | Clube → **gestora CVM** | CGA + registro CVM/ANBIMA + ≥12m de cota oficial do clube |

---

## SEQUENCIAMENTO E PARALELISMO (visão de execução)

```
AGORA  ── Fase 1 (track record) ──┐  começa JÁ; captura NAV mesmo antes do rebalancer
                                  ├─ Fase 2 (deploy paper) ── Fase 3 (disciplina) ──► G4 (≥6–12m paper limpo)
EM PARALELO (humano):              │
  Aguardar veredito do backtest beta_portfolio.py (fixa limites de risco T3.1 e decisão G4)
  Iniciar CFG→CGA (Degrau 2)  ─────┘                                   │
                                                                       ▼
                                       Fase 4 Degrau 1 (clube: jurídico + administrador) ── G-AUM1
                                                                       │
                                                                       ▼ (após G4 + ≥12–24m)
                                                                  G5 → captar no clube
                                                                       │
                                                                       ▼ (após CGA + registro)
                                                                  G6 → gestora CVM
```

**Caminho crítico técnico:** T1.1→T1.4 (NAV + métricas) e T2.1 (paridade backtest↔vivo) são os dois nós que destravam todo o resto.
**Caminho crítico humano/legal:** CGA (meses de estudo) e o parecer jurídico do clube — **começar em paralelo desde já**, pois são os mais lentos.

---

## DECISÕES PENDENTES DO USUÁRIO (consolidadas)
1. **[Track record]** Onde guardar o backup append-only imutável do `nav_history` (recomendado: commit git diário de export CSV).
2. **[Paper]** Habilitar cripto no paper desde o início, ou começar só com ETFs? (cripto = mais ruído/custo).
3. **[Paper]** Tickers de cripto na Alpaca paper e mapeamento ↔ backtest.
4. **[Cadência]** Permitir des-risco intra-mês com re-risco só no rebalance mensal (recomendado), ou rebalance estritamente mensal?
5. **[Risco]** Números finais de `max_drawdown_pct` (halt) e `max_per_symbol_pct` — derivar do veredito do backtest (`data/beta_verdict.txt`).
6. **[Disciplina]** Aceitar a política de congelamento de parâmetros durante a janela de track record.
7. **[Gates]** Meses mínimos: G4 (próprio) e G5 (terceiros) — recomendações: 6/12 e 12/18–24.
8. **[AUM]** Cotistas iniciais e patrimônio do clube; quando iniciar a CGA; quando migrar para gestora.
9. **[Reconciliação]** Adicionar notificação externa (e-mail/desktop) para divergências?

## NÃO-METAS / GUARDRAILS (do briefing e da memória)
- **NÃO** vender retorno acima do mercado — o objetivo é **risco-ajustado/drawdown menor** + track record. Se o backtest do beta **falhar** a barra honesta (`_verdict_text` em `beta_portfolio.py`: tombo materialmente melhor com Sharpe ≥ buy&hold), **reavaliar a estratégia antes de qualquer deploy** (não forçar um deploy que o próprio tribunal reprovou).
- **NÃO** gerir dinheiro de terceiros (nem informalmente, nem grátis) antes de G-AUM1 — **crime** (Art. 27-E).
- **NÃO** reescrever histórico de NAV/trades — quebra o ativo central (track record).
- **PAPER apenas** até G4. O guard de segurança de `config/settings.py` (`LiveTradingBlockedError`) permanece como trava.
```
