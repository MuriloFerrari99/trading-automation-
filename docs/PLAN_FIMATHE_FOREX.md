# PLANO DE EXECUÇÃO — Pivô para FOREX + FIMATHE

> **Documento do PLANNER** (não-membro do conselho; só executa). Traduz a decisão do conselho num plano granular, sequenciado e com gates. **Não contém código de produção** — só o mapa de execução.
>
> **Decisão do conselho (o norte):** pivotar 100% para **FOREX** com a técnica **FIMATHE** (Marcelo Ferreira), operando majors conhecidos e **menos voláteis** (EUR/USD, EUR/GBP, USD/CHF). Esquecer ações e, por ora, cripto. **Provar no tribunal** (DSR/PBO, custo real incl. SWAP, vs buy&hold) **ANTES de qualquer dinheiro real**. Execução futura via **Hantec** (MT5).

---

## 0. TL;DR (leia isto primeiro)

- **Caminho crítico** = `Dados forex → Custos+SWAP → Tribunal forex (vs buy&hold) → VEREDITO`. Tudo o que toca execução (MT5/Hantec) fica **bloqueado atrás do GATE 1** (tribunal PASSA).
- **Maior risco do projeto é o SWAP**, não o spread. EUR/USD long paga ~−10 pips/noite (triplo na quarta). Uma técnica que segura posição por dias/semana (FIMATHE) pode ter o edge inteiro comido pelo carry negativo. **O tribunal tem que medir isso com swap real.**
- **Correção estatística obrigatória** (lição já paga, `edge-alpha-vs-beta`): o tribunal atual compara Sharpe contra **zero**. Forex (sem o beta de bull das ações) é o lugar certo pra isso importar menos, mas a barra continua sendo **alpha vs buy&hold**, não retorno absoluto. O `forex_tribunal` nasce já com o benchmark correto.
- **Dependência aberta que pode reescrever a spec:** O PESQUISADOR está extraindo a essência pura da FIMATHE. Pode revelar que (a) a entrada certa é **pullback**, não rompimento; (b) o **timeframe de marcação** é o canal de abertura M15 da semana, não swing rolante. O plano isola isso num ponto de reconciliação (T-A4) **antes** de fechar a Fase B.
- **O Mac não roda MT5.** Execução exige **VPS Windows**. Isso é decisão/custo do usuário e só morde na Fase D — não bloqueia o tribunal.

**Sequência de fases e seus gates:**

```
FASE A  Dados + Custos forex     ──┐
FASE B  Tribunal forex (vs B&H)    │  GATE 1 (veredito) ── decide se o projeto continua
                                  ──┘
        ▼ (só se GATE 1 = PASSA)
FASE C  Risco + estado (sleeve forex, disjuntores FIMATHE)
        ▼
FASE D  Execução MT5/Hantec (paper/demo)  ── GATE 2 (paridade backtest×demo) ── GATE 3 (go-live)
```

---

## 1. O que reaproveitar (NÃO refazer)

| Asset existente | Papel no pivô | Mudança necessária |
|---|---|---|
| `fimathe/engine.py` (`FimatheEngine`) | Motor analítico OHLC: canais por swing, ZN, Fibonacci, PCM, RSI/ADX, sinais, stops. **Já em pips** (`pip_size`). | **Nenhuma na v1** (já é forex-ready). Possíveis ajustes só vêm da reconciliação do PESQUISADOR (T-B2). |
| `simulation/statistics.py` | Juiz: PSR, **DSR ≥ 0.95**, PBO via CSCV. Pronto. | **Adicionar `FOREX_PERIODS`** e usar `evaluate_edge` com **benchmark = retorno do buy&hold** (não 0). |
| `simulation/metrics.py` | Sharpe/Sortino/MDD/PF/CAGR. | Reusar como está. |
| `simulation/costs.py` (`CostModel`, bps) | Modelo de custo por lado em bps + cenário estressado (2x). | **Estender p/ forex**: spread em **pips→bps**, e um modelo de **SWAP** separado (custo por noite, não por trade). |
| `simulation/breakout_tribunal.py` | Já monta carteira diária de setups FIMATHE com fills gap-aware, equal-dólar vs equal-risco, gate de regime, n_trials honesto. | **Forte base do `forex_tribunal`** — mas hoje roda cripto/ações vs **Sharpe-contra-zero**. Adaptar p/ forex + **benchmark buy&hold** + **swap**. |
| `simulation/crypto_intraday.py` | Padrão de loader (download→cache→backtest), CLI `--download`. | **Template** para `data/forex_data.py` (estrutura de cache + CLI). |
| `risk/crypto_sleeve.py` + `config/crypto_sleeve.py` | Trava isolada: capital isolado, **ceiling rígido de alavancagem**, kills (perda diária/DD), posições concorrentes, respeita kill global. Strategy-agnostic. | **Clonar p/ `forex_sleeve`**: ceiling adequado a forex, **disjuntor FIMATHE de 2–3 stops/semana**, sizing por **pip/lote**. |
| `broker/base.py` (`BrokerClient`, `MarketClockInfo`, `BrokerOrder`) | Interface agnóstica. Toda a lógica depende dela, não da Alpaca. | **Implementar `MT5Broker`** cumprindo o contrato (+ `get_ohlc_bars`). |
| `core/kill_switch.py` | Kill global, sentinela em arquivo + env. | Reusar intacto (forex sleeve respeita). |
| `feedback/regime.py` (`classify_regime`) | Classificador de regime usado pelo tribunal. | Reusar; avaliar se faz sentido em forex (sessões, não bull/bear). |
| `risk/sizing.py` (`fixed_fractional_qty`) | Sizing fracionário. | Reusar; o sleeve forex converte risco→lote via valor do pip. |

**Convenções do repo a respeitar** (não-negociáveis):
- Cache de dados e relatórios gerados vão em `data/` e entram no **`.gitignore`** (já é o padrão: `data/cache/`, `data/*_verdict.txt`).
- Testes **offline** (sem rede) — fetch/download é passo separado com flag `--download`.
- `uv` p/ ambiente; `ruff` limpo; Pydantic v2 p/ settings.
- Endpoint **paper/demo** travado; flag `LIVE_TRADING` segue bloqueada por guard (igual ao que já existe).
- `numpy` é usado em toda a simulação mas **só vem transitivamente via `pandas`** — se algum módulo novo usar numpy direto (e usará), **adicionar `numpy` às deps** explicitamente (T-A0).

---

## FASE A — Dados de forex + modelo de custo honesto

**Objetivo da fase:** ter barras OHLC reais de EUR/USD, EUR/GBP, USD/CHF (em vários timeframes) cacheadas offline, e um modelo de custo que capture **spread + SWAP** (o sangrador). Sem isto o tribunal não roda.

> **Paralelismo:** A0, A1 e A2 podem rodar em paralelo. A3 depende de A1+A2.

### T-A0 — Saneamento de dependências e esqueleto de pacote
- **Objetivo:** garantir que `numpy` seja dep explícita e criar os stubs vazios dos módulos forex (com docstring + assinatura) para destravar trabalho paralelo.
- **Entregáveis:** edição em `pyproject.toml` (`numpy>=1.26`); stubs de `data/forex_data.py`, `simulation/forex_costs.py`, `simulation/forex_swap.py`, `simulation/forex_tribunal.py`.
- **Dependências:** nenhuma.
- **Executor:** `general-purpose` (edição simples) ou direto.
- **Critério de aceite:** `uv sync` ok; `uv run ruff check .` limpo; `python -c "import simulation.forex_costs"` não quebra.
- **Esforço:** XS (~30 min).

### T-A1 — Loader de dados forex (`data/forex_data.py`)
- **Objetivo:** baixar e cachear OHLC dos 3 majors em **M15, H1 e D1** (M15 é o timeframe nativo da FIMATHE; H1/D1 p/ contexto e p/ "linhas do Equador"). Espelhar o padrão `crypto_intraday` (download→cache CSV/parquet→leitura offline).
- **Decisão de fonte (precisa resolver — ver §RISCOS):** Alpaca **não tem forex**. Opções: (a) **dukascopy** (tick/candle histórico grátis, bom p/ majors); (b) **HistData.com** (M1 grátis, CSV); (c) **MT5 `copy_rates`** (mas exige Windows — só na Fase D); (d) provedor pago (Polygon FX, EODHD). **Recomendação do PLANNER: HistData ou dukascopy p/ o backtest** (offline, grátis, sem depender do MT5/VPS ainda).
- **Entregáveis:** `data/forex_data.py` com `download_pair(symbol, timeframe)` + `load(symbol, timeframe)`; cache em `data/forex/<SYM>/<TF>.csv` (gitignored); CLI `uv run python -m data.forex_data --download`.
- **Dependências:** T-A0.
- **Executor:** `general-purpose` (escreve módulo + adiciona `data/forex/` ao `.gitignore`).
- **Critério de aceite:** após `--download`, `load("EURUSD","M15")` devolve DataFrame com `open/high/low/close`, índice datetime UTC monotônico, **sem NaN nos OHLC**, ≥ 2 anos de histórico (idealmente cobrindo 2020+ p/ ter regimes adversos, lição de `simulation-findings`). Teste offline lê de um fixture pequeno (não baixa).
- **Esforço:** M (meio dia, dominado por achar/limpar a fonte).

### T-A2 — Modelo de custo forex (`simulation/forex_costs.py`)
- **Objetivo:** custo de transação por lado em **pips**, convertido p/ a unidade que o tribunal usa (fração de retorno). Presets por par (EUR/USD ~0,6 pip; EUR/GBP e USD/CHF a medir). Reusar a mecânica de `CostModel.stressed(2x)`.
- **Entregáveis:** `simulation/forex_costs.py` com `ForexCostModel` (spread_pips, slippage_pips, pip_size, pip_value) + `per_side_frac` + `.stressed()` + presets `EURUSD_BASE`, `EURGBP_BASE`, `USDCHF_BASE`.
- **Dependências:** T-A0.
- **Executor:** `general-purpose`.
- **Critério de aceite:** teste unitário: spread 0,6 pip em EUR/USD a 1,1000 ≈ 0,55 bps/lado; `.stressed(2)` dobra; round-trip = 2×. **Sem custo otimista** embutido.
- **Esforço:** S (1–2 h).

### T-A3 — Modelo de SWAP overnight (`simulation/forex_swap.py`) ⚠️ peça crítica
- **Objetivo:** modelar o **carry overnight** — o principal candidato a matar a FIMATHE. Custo aplicado **por noite mantida** (não por trade), com **regra do triplo na quarta-feira** (swap de 3 dias p/ cobrir o fim de semana de settlement).
- **Entregáveis:** `simulation/forex_swap.py` com `SwapModel(long_pips_per_night, short_pips_per_night, triple_weekday=2)` + função que, dada a série de datas que a posição ficou aberta e o lado, devolve o **total de swap em pips** (aplicando o triplo na quarta). Presets pessimista/base por par (default conservador até medir na demo).
- **Dependências:** T-A0. (Os números reais vêm da demo Hantec — **decisão aberta**; até lá, usar valores conservadores/documentados.)
- **Executor:** `general-purpose`.
- **Critério de aceite:** teste unitário: posição long EUR/USD aberta seg→sex (4 noites, incl. quarta) cobra `3×qua + 3 outras noites` = 6 "diárias" de swap; short usa a perna short; finais de semana sem barra não duplicam. Documentar a fonte de cada número e marcá-lo como "a confirmar na demo".
- **Esforço:** M (meio dia — a regra do triplo + alinhamento com o calendário de barras é o detalhe sutil).

**🚧 GATE A→B:** `data/forex_data.py` carrega os 3 pares offline (OHLC limpo, ≥2 anos) **E** `forex_costs`/`forex_swap` têm testes verdes. Sem dados limpos e custo+swap, o tribunal não tem o que julgar.

---

## FASE B — Tribunal forex (o juiz que decide o projeto)

**Objetivo da fase:** submeter a FIMATHE em forex ao mesmo rigor estatístico já usado (DSR/PSR/PBO), com **custo real + swap**, comparada a **buy&hold** (medir ALPHA, não retorno absoluto). É aqui que o conselho recebe o veredito.

> **Caminho crítico.** Nada de execução começa antes do GATE 1.

### T-B0 — Corrigir o benchmark do juiz (`simulation/statistics.py`)
- **Objetivo:** implementar a lição de `edge-alpha-vs-beta`: o tribunal hoje testa Sharpe **vs zero** (`sr_benchmark=0`). Adicionar suporte a **benchmark = Sharpe do buy&hold** no mesmo span, e `FOREX_PERIODS` (252 dias úteis; ou 260 — decidir e documentar).
- **Entregáveis:** em `simulation/statistics.py`: constante `FOREX_PERIODS`; e um caminho em `evaluate_edge` (ou wrapper no tribunal) que receba `sr_benchmark_annual` ≠ 0 e meça **alpha** (excesso sobre o buy&hold), mantendo retrocompat com os tribunais existentes.
- **Dependências:** nenhuma (pode rodar em paralelo com a Fase A).
- **Executor:** `general-purpose`. **Cuidado:** não quebrar `test_statistics.py` nem os tribunais de cripto/ações que já importam daqui.
- **Critério de aceite:** `test_statistics` continua verde; novo teste mostra que, dado um retorno idêntico ao buy&hold, o **alpha-DSR ≈ não-significativo** (não passa por ser só beta).
- **Esforço:** S–M (2–4 h; o cuidado é a retrocompat).

### T-B1 — Adaptador de série de retornos forex p/ o tribunal
- **Objetivo:** gerar, a partir dos dados forex + `FimatheEngine`, a **série diária de retornos da carteira** de setups FIMATHE — análogo a `breakout_tribunal.generate_trades`/`portfolio_series`, mas: (a) **long E short** (forex é simétrico, sem viés de bull); (b) descontando **spread (entrada+saída)** e **swap (por noite mantida)**; (c) com fills **gap-aware** (reusar a lógica existente).
- **Entregáveis:** funções de geração de trades/carteira dentro de `simulation/forex_tribunal.py` (ou um `simulation/forex_backtest.py` separado se ficar grande), reusando ao máximo o código de `breakout_tribunal.py`.
- **Dependências:** T-A1, T-A2, T-A3. Spec FIMATHE provisória (rompimento, como já está na engine) — **a reconciliar em T-B2**.
- **Executor:** `general-purpose` ou `Plan`+`general-purpose` (é o coração técnico). Recomendo um **agente dedicado** com o `breakout_tribunal.py` como referência.
- **Critério de aceite:** roda nos 3 pares sem erro; produz série diária; **o swap aparece como dedução** (ligar/desligar swap muda o resultado de forma material — sanity check de que está conectado); sem look-ahead (entrada no open de t+1).
- **Esforço:** L (1–2 dias — é o módulo mais denso).

### T-B2 — Reconciliação com a essência FIMATHE do PESQUISADOR ⚠️ dependência externa
- **Objetivo:** quando o PESQUISADOR entregar a essência pura, **reconciliar a spec** com o que está codificado. Itens prováveis de mudança: **entrada por pullback** (não rompimento puro); **canal de abertura M15 das 4 primeiras velas da semana** como referência (vs swing rolante atual); **zona neutra** como o canal imediatamente anterior; **stop fora da zona neutra** (a engine já faz "stop fora da caixinha", validar se é a mesma coisa).
- **Entregáveis:** um **diff de spec** (o que muda no `FimatheEngine`/no adaptador) + ajustes pontuais. Se a mudança for grande, vira sub-tarefas próprias.
- **Dependências:** **entrega do PESQUISADOR** (externa — pode atrasar a Fase B). T-B1 (pra saber o que ajustar).
- **Executor:** humano (usuário) decide a spec final; `general-purpose` implementa o diff.
- **Critério de aceite:** a lógica de entrada/zona-neutra/stop no código bate com a essência aprovada pelo usuário; documentado em `docs/research/07/08` (atualizar os docs existentes).
- **Esforço:** indeterminado (S a L conforme o achado). **Risco de retrabalho** se rodarmos o tribunal antes e a spec mudar — ver Decisão D-4.

### T-B3 — Tribunal forex completo (`simulation/forex_tribunal.py`)
- **Objetivo:** o CLI que produz o **VEREDITO**. Matriz de cenários: 3 pares × {peso dólar, peso risco} × {custo base, custo estressado 2x} × {com swap, sem swap (diagnóstico)} × {M15 vs H1 — qual timeframe a FIMATHE vive}. Saída: DSR/PSR/PBO + **alpha vs buy&hold** + n_trials honesto.
- **Entregáveis:** `simulation/forex_tribunal.py` com CLI (`--report-file data/forex_verdict.txt`, `--n-trials`); relatório textual no padrão do `breakout_tribunal`.
- **Dependências:** T-B0, T-B1, (idealmente T-B2 reconciliada).
- **Executor:** `general-purpose`.
- **Critério de aceite:** `uv run python -m simulation.forex_tribunal` gera `data/forex_verdict.txt` (gitignored) com a tabela; **n_trials reflete a busca real** (não espantalho); o bloco "sem swap vs com swap" deixa explícito quanto o carry custa.
- **Esforço:** M (meio dia a 1 dia sobre T-B1).

### T-B4 — Veredito + memória
- **Objetivo:** ler o relatório, escrever o **veredito honesto** (passou/falhou, e por quê — provavelmente o swap), e gravar na memória do projeto (como `momentum-oos-verdict`, `edge-alpha-vs-beta`).
- **Entregáveis:** nota de memória `fimathe-forex-verdict.md`; resumo pro conselho.
- **Dependências:** T-B3.
- **Executor:** `general-purpose` + usuário.
- **Critério de aceite:** veredito registrado com números; decisão de GATE 1 tomada.
- **Esforço:** XS.

**🚧 GATE 1 (o gate que decide o projeto):**
> A FIMATHE em forex **PASSA** se, **líquida de spread + SWAP** (cenário **estressado 2x** p/ ser honesto), em pelo menos um par/timeframe:
> 1. **DSR ≥ 0.95** (significativo a p<0.05 após n_trials honesto), **E**
> 2. **Sharpe anual líquido ≥ 0.8**, **E**
> 3. **Alpha positivo vs buy&hold** no mesmo span (não é só beta), **E**
> 4. **PBO < 0.5** (a seleção pega skill, não sorte).
>
> **Se FALHAR** (cenário esperado, dada a base de price-based reprovados: ML AUC~0.48, momentum DSR 0.04): **NÃO prosseguir p/ execução.** O sistema é um framework de gestão de risco, não um gerador de alpha. Opções: (a) aceitar e parar; (b) reconciliar a spec FIMATHE de novo (se T-B2 mudou algo); (c) último lever de `momentum-oos-verdict` = **informação nova além do preço** (não mais variação técnica). **Não gastar VPS/Hantec sem passar este gate.**

---

## FASE C — Risco + estado (só se GATE 1 = PASSA)

**Objetivo da fase:** envelopar a FIMATHE numa trava de risco isolada (igual ao sleeve cripto) e implementar os **disjuntores próprios da FIMATHE** (parar a semana após 2–3 stops). Ainda **sem tocar a corretora**.

> **Paralelismo:** C1 e C2 são independentes entre si.

### T-C1 — Sleeve de risco forex (`risk/forex_sleeve.py` + `config/forex_sleeve.py`)
- **Objetivo:** clonar `CryptoSleeveGuard`/`CryptoSleeveSettings` p/ forex: capital isolado, **ceiling rígido de alavancagem** (forex de varejo chega a 30–500x; o ceiling existe pra impedir o suicídio — definir, ex. 5–10x), risco por trade, kills de perda diária/DD, posições concorrentes, respeita kill global. **Sizing por pip/lote** (a engine já calcula `position_size` em lotes via valor do pip).
- **Entregáveis:** `risk/forex_sleeve.py`, `config/forex_sleeve.py`, testes (espelhar `test_crypto_sleeve.py`).
- **Dependências:** GATE 1.
- **Executor:** `general-purpose`.
- **Critério de aceite:** testes verdes; ceiling de alavancagem **não ultrapassável** nem por env absurdo (dupla trava, como no cripto); sizing converte risco% → lote corretamente p/ um par dado.
- **Esforço:** M (meio dia — é majoritariamente adaptação).

### T-C2 — Disjuntores FIMATHE + estado (`strategies/` + `data/state_repo`)
- **Objetivo:** implementar a **contenção de danos** da FIMATHE (doc 07 §7): contador de stops consecutivos → **para a semana após 2–3 stops** (desliga novos trades até a próxima abertura semanal). Persistir estado (reusar `StateRepository`, padrão `trailing_high:SYM`).
- **Entregáveis:** lógica de disjuntor (no sleeve ou numa Strategy FIMATHE), chave de estado `fimathe_stops_week:SYM`/global; teste.
- **Dependências:** GATE 1. (Independe de T-C1.)
- **Executor:** `general-purpose`.
- **Critério de aceite:** após N stops na semana, o sistema **recusa novas entradas** até o reset semanal; estado sobrevive a restart.
- **Esforço:** S–M.

**🚧 GATE C→D:** sleeve forex + disjuntores com testes verdes; o caminho de decisão (Planner/risco) consegue produzir uma **intenção FIMATHE dimensionada em lote** sem tocar a corretora (validável com `FakeBroker`).

---

## FASE D — Execução MT5 / Hantec (paper/demo primeiro)

**Objetivo da fase:** ligar a corretora real **em demo**, validar paridade backtest×demo, e só então discutir go-live. **Bloqueada por infra (VPS Windows) e pelo GATE 1.**

> ⚠️ **Restrição dura:** o pacote `MetaTrader5` (Python) **só roda em Windows** com o terminal MT5 instalado. O Mac do usuário **não** roda nativo. **Requer VPS Windows** (decisão/custo aberto — D-1). Sem REST/FIX na Hantec. **Não comece a Fase D sem o VPS provisionado.**

### T-D1 — `MT5Broker` cumprindo `BrokerClient` (`broker/mt5_broker.py`)
- **Objetivo:** implementar a interface `broker/base.py` sobre o pacote `MetaTrader5`: `get_account`, `get_positions`/`get_position`, `get_last_price`, `get_bars` + `get_ohlc_bars` (via `copy_rates_*`), `submit_order` (idempotência via `client_order_id`→`magic`/comment), `cancel_all_orders`, `get_open_orders`, `get_order_by_client_id`, `is_market_open`/`get_clock`. **Opções não se aplicam** (forex) — levantar `NotImplementedError` claro nos métodos de opção.
- **Entregáveis:** `broker/mt5_broker.py`; testes com um **fake/stub do módulo `MetaTrader5`** (offline, sem terminal) validando o mapeamento; `MetaTrader5` como dep **opcional** (extra `mt5`, só instalável no Windows — não quebrar o `uv sync` do Mac).
- **Dependências:** GATE 1, VPS Windows (D-1), confirmação do servidor MT5 da Hantec (D-2). T-D2 (clock) pode vir junto.
- **Executor:** `general-purpose` (mapeamento) — mas a **validação real exige o ambiente Windows** (humano/VPS).
- **Critério de aceite:** testes offline (stub) verdes no Mac; a dep opcional não quebra o ambiente principal; (no VPS) conecta na conta demo Hantec e lê conta/posições.
- **Esforço:** L (1–2 dias + tempo de ambiente Windows).

### T-D2 — Relógio de sessões forex (`core/mt5_clock.py`)
- **Objetivo:** `is_market_open`/`next_open`/`next_close` para forex (24/5, abre domingo noite, fecha sexta; respeitar a "abertura de domingo" que a FIMATHE usa p/ marcar o canal). Hoje o `MarketClock` delega a Alpaca (equities).
- **Entregáveis:** `core/mt5_clock.py` (do MT5 `symbol_info`/horários, ou tabela de sessões); teste offline.
- **Dependências:** GATE 1. (Pode vir junto com T-D1.)
- **Critério de aceite:** sabe que sábado está fechado e domingo à noite abre; identifica a abertura semanal (gatilho da marcação FIMATHE).
- **Esforço:** S–M.

### T-D3 — Paper/demo ao vivo + paridade (validação)
- **Objetivo:** rodar a FIMATHE na **demo Hantec** por um período, capturar fills/spread/**swap reais**, e **comparar com o backtest** (paridade). É aqui que se **mede o swap real** (fecha a decisão D-3) e se valida que os fills batem com os assumidos no tribunal.
- **Entregáveis:** relatório de paridade backtest×demo; swap real medido alimentado de volta no `forex_swap` (re-rodar o tribunal com swap real fecha o loop).
- **Dependências:** T-D1, T-D2, VPS rodando.
- **Critério de aceite:** **GATE 2** abaixo.
- **Esforço:** tempo de mercado (semanas de demo) + análise.

**🚧 GATE 2 (paridade):** os fills, spread e **swap reais** da demo **não destroem** o edge medido no tribunal (re-rodar `forex_tribunal` com swap real **ainda passa o GATE 1**). Se o swap real for pior que o assumido e derrubar o edge → **volta pro GATE 1** (não vai a dinheiro real).

**🚧 GATE 3 (go-live):** **decisão de negócio do usuário**, fora do escopo técnico. Pré-condições: GATE 2 passou; risco regulatório BR/offshore (D-5) aceito explicitamente; `LIVE_TRADING` só destravado com `LiveReadinessGate`/soak/canary (padrão do roadmap F6). **O PLANNER não destrava isto.**

---

## 2. Mapa de paralelismo vs caminho crítico

**Caminho crítico (sequência que não pode ser comprimida):**
`T-A1 (dados) → T-A2+T-A3 (custo+swap) → T-B1 (adaptador) → T-B3 (tribunal) → GATE 1 → [Fase C] → [Fase D após VPS] → GATE 2 → GATE 3`

**Pode rodar EM PARALELO:**
- **Bloco 1 (agora, antes do tribunal):** T-A0, T-A1, T-A2, T-A3 e **T-B0** (correção do benchmark no juiz) — T-B0 não depende dos dados.
- **Reconciliação (T-B2):** trabalho do PESQUISADOR roda **em paralelo** com A/B até o ponto de merge; só vira bloqueio se decidirmos *não* rodar o tribunal com a spec provisória (D-4).
- **Fase C:** T-C1 (sleeve) e T-C2 (disjuntores) são independentes entre si.
- **Fase D:** T-D1 (broker) e T-D2 (clock) podem ser feitos juntos; a **validação real** de ambos espera o VPS.

**Recomendação de execução:** disparar T-A1/T-A2/T-A3/T-B0 **em paralelo** (4 agentes), convergir em T-B1→T-B3, e **segurar** a Fase C/D até o GATE 1. Reconciliar T-B2 assim que o PESQUISADOR entregar — idealmente **antes** de T-B3 pra não retrabalhar o veredito.

---

## 3. Decisões / Riscos ABERTOS (precisam do usuário)

| # | Decisão / Risco | Por que importa | Quando trava |
|---|---|---|---|
| **D-1** | **VPS Windows** (provisionar, custo, provedor) | `MetaTrader5` só roda em Windows; Mac não. Sem VPS não há execução automatizada. | Bloqueia **Fase D** (não o tribunal). |
| **D-2** | **Servidor MT5 da Hantec** (confirmar no portal) + tipo de conta (Global/spread 0,6 pip) | Necessário p/ conectar o `MT5Broker` e p/ saber spread/condições reais. | Bloqueia **T-D1/T-D3**. |
| **D-3** | **Medir o SWAP real na demo** | É o sangrador nº1 da FIMATHE; o tribunal usa valor conservador/documentado até medir. O número real pode reprovar a estratégia. | Refina **T-A3**; fecha no **GATE 2**. |
| **D-4** | **Rodar o tribunal com a spec FIMATHE provisória (rompimento) OU esperar o PESQUISADOR?** | Se rodarmos antes e a essência for "pullback", retrabalha-se T-B1/T-B3 e o veredito. Esperar atrasa; rodar cedo dá sinal direcional. **Recomendação PLANNER:** rodar um veredito **provisório** com a engine atual (barato, reusa o `breakout_tribunal`) e **re-rodar** após reconciliar — o provisório já diz se o swap mata tudo. | Afeta ordem **T-B2 vs T-B3**. |
| **D-5** | **Risco regulatório BR** (Hantec sem CVM; cliente via entidade offshore/Maurício) | Decisão de negócio; não bloqueia o teste, mas é **pré-condição do GATE 3** (go-live). | Bloqueia **GATE 3**. |
| **D-6** | **Fonte de dados forex p/ backtest** (HistData/dukascopy grátis vs pago vs MT5) | Define T-A1; MT5 como fonte só existe na Fase D (Windows). Grátis destrava o tribunal já. | Bloqueia **T-A1**. |
| **D-7** | **Universo e timeframe final** (EUR/USD, EUR/GBP, USD/CHF; M15 nativo da FIMATHE vs H1/D1) | O conselho fixou os pares; o **timeframe** ainda depende da essência (canal de abertura M15 da semana?). O tribunal testa M15 vs H1 e o PESQUISADOR confirma. | Refina **T-A1/T-B3** via **T-B2**. |

---

## 4. Resumo dos GATES (o que precisa ser verdade pra avançar)

| Gate | De → Para | Condição (resumo) |
|---|---|---|
| **GATE A→B** | Fase A → B | Dados dos 3 pares carregam offline (OHLC limpo, ≥2 anos) + `forex_costs`/`forex_swap` com testes verdes. |
| **GATE 1** ⭐ | Fase B → C (decide o projeto) | Líquido de spread+swap (**estressado 2x**): **DSR ≥ 0.95** E **Sharpe anual ≥ 0.8** E **alpha > 0 vs buy&hold** E **PBO < 0.5**, em ≥1 par/TF. Se falhar, **pára** (não vai a execução). |
| **GATE C→D** | Fase C → D | Sleeve forex + disjuntores FIMATHE com testes verdes; intenção FIMATHE dimensionada em lote via `FakeBroker`, sem tocar a corretora. |
| **GATE 2** | dentro da Fase D | Fills/spread/**swap reais** da demo não derrubam o edge (re-rodar tribunal com swap real **ainda passa GATE 1**). |
| **GATE 3** | Fase D → dinheiro real | **Decisão de negócio do usuário** + risco regulatório (D-5) aceito + `LiveReadinessGate`/soak/canary. `LIVE_TRADING` segue bloqueada por guard até aqui. |

---

## 5. Entregáveis novos (mapa de arquivos)

| Arquivo | Fase | Status | Reusa |
|---|---|---|---|
| `data/forex_data.py` | A | novo | padrão de `simulation/crypto_intraday.py` |
| `simulation/forex_costs.py` | A | novo | `simulation/costs.py` (`CostModel`) |
| `simulation/forex_swap.py` | A | novo | — (peça nova, crítica) |
| `simulation/statistics.py` | B | **editar** (add `FOREX_PERIODS` + benchmark buy&hold) | ele mesmo |
| `simulation/forex_tribunal.py` (+ `forex_backtest.py` se grande) | B | novo | **`simulation/breakout_tribunal.py`** (base forte) |
| `docs/research/07/08` | B | **editar** (reconciliar essência do PESQUISADOR) | docs existentes |
| `risk/forex_sleeve.py` + `config/forex_sleeve.py` | C | novo | `risk/crypto_sleeve.py` + `config/crypto_sleeve.py` |
| disjuntor FIMATHE (Strategy/estado) | C | novo | `data/state_repo.py`, `core/kill_switch.py` |
| `broker/mt5_broker.py` | D | novo | `broker/base.py` (contrato), `broker/fake_broker.py` (padrão de teste) |
| `core/mt5_clock.py` | D | novo | `core/market_clock.py` |
| testes correspondentes em `tests/` | A–D | novos | `test_costs.py`, `test_crypto_sleeve.py`, `test_fimathe_engine.py`, `test_statistics.py` |

Tudo aditivo; nada quebra o sistema Alpaca existente (que pode ser arquivado/desligado depois, fora do escopo deste plano).

---

## 6. Nota honesta do PLANNER (contexto que o conselho deve ter em mente)

Quatro candidatos de alpha **price-based** já foram reprovados em OOS rigoroso neste repo: ML por-setup (AUC ~0.48), momentum (DSR 0.04 / PBO 0.84), fading de liquidação, e carry delta-neutro (edge insuficiente). A FIMATHE é o **quinto** candidato price-based e é **não-testada em forex** — mas o padrão histórico e a física do **swap negativo** sugerem que o **GATE 1 é uma barra real, não cerimônia**. O valor já provado do sistema é a **camada de risco** (corta DD de cauda −100%→−21,5%). Este plano foi desenhado pra **descobrir a verdade barato** (reusando o tribunal pronto) **antes** de gastar VPS/Hantec/capital. Se a FIMATHE passar o GATE 1 com swap real, é o **primeiro alpha de verdade** do projeto; se não, economizamos o caminho de execução inteiro.
