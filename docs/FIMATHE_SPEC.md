# FIMATHE — Especificação Canônica (revisada com as TRANSCRIÇÕES do canal)

> **REVISÃO 2026-06-15 (O PESQUISADOR).** Esta spec foi **reescrita** após baixar e estudar
> **64 transcrições reais** do canal de Marcelo Ferreira (`https://www.youtube.com/@MARCELOFERREIRAFIMATHE`),
> incluindo os cursos oficiais **"Forex Setup" (Aulas 1-9, 2026)**, **"Forex Scalper Fimathe"
> (Aulas 1-20)**, **"Fimathe 4.0 Aula 1"**, **"Primórdios da Fimathe"** e dezenas de operações ao vivo.
> Transcrições cacheadas em `data/fimathe_videos/<id>.txt` (legendas automáticas pt via yt-dlp).
> **A versão anterior desta spec descrevia um MODELO ERRADO** (CR = último topo/fundo da perna da
> tendência; swing D1→H4; entrada nos 50%). As correções abaixo vêm das **palavras do próprio
> Marcelo**. Itens corrigidos marcados com **[CORRIGIDO]**; confirmados com **[CONFIRMADO]**.
> Citações com `id` do vídeo entre colchetes.

## Filosofia [CONFIRMADO]
FIMATHE = **Fibonacci + Matemática** — Marcelo "homenageou" Fibonacci no nome; a técnica é uma
destilação de **Fibonacci (raiz), Teoria de Dow, Elliott, fractal e rastreadores de tendência**
que ele usava antes [ypWwZ3K7wrI, adXpiMMMJHo]. Tela LIMPA, só **preço puro** (sem médias, RSI,
bandas). "Tenho um oceano de conhecimento mas para operar uso só um copo d'água" [adXpiMMMJHo].
Premissa: o movimento institucional deixa uma **geometria de ciclos repetíveis** que se **mapeia
ANTES** do preço se mover ("desenhar a escada/rodovias antes dos carros entrarem nelas").

## Instrumento, timeframes e sessão [CORRIGIDO — era "swing D1→H4, ignora calendário"]
- **MERCADO: FOREX + XAU/USD** (ouro é **99% das operações** dele), via **MetaTrader 5** na
  corretora Hantec [k7Jtalap-7Q, adXpiMMMJHo]. Também índices (HK50) e, esporadicamente, Bitcoin.
  **Alpaca não tem forex nem XAU/USD** → a técnica nativa NÃO roda no nosso broker (ver §Crypto).
- **É DAY TRADE: 99% das ordens abrem e fecham em < 24h** [k7Jtalap-7Q, sKnrcI9YN8s].
  **NÃO é swing, NÃO atravessa noites** → **swap/overnight é IRRELEVANTE** para o método canônico.
- **Timeframes REAIS: marca no MACRO (mensal/semanal/diário), executa no MICRO (M15 e M1)**.
  Pares ensinados: **semanal→M15**, **diário→M1** [q8SXvERstI8, ginFNrsiX5A]. Ele usa
  1/5/15/30/60/240/D/W/M e diz que **dislike M5**; "a técnica funciona em qualquer tempo gráfico".
- **HÁ um relógio semanal** (forex é descentralizado, sem sede, mas ele tem ritmo):
  **a análise SEMPRE começa no domingo** (abertura do mercado ~19h Brasília) — "o filtro nº1 da
  técnica, tudo começa no domingo" [ypWwZ3K7wrI]. Opera **domingo→quinta**, **evita sexta**.
  **Quarta e quinta** = o preço tende a estar nas máximas/mínimas da semana.

## Marcação — o coração [CORRIGIDO — o modelo "último topo/fundo da perna" estava ERRADO]
A unidade fundamental NÃO é um swing rolante. É o **CANAL DE ABERTURA**:
- **CANAL DE ABERTURA (CA):** as **4 PRIMEIRAS VELAS do gráfico M15** logo na abertura
  (semanal no domingo, ou diário) formam uma "caixinha" [oCCOa4_awiE: "o canal de abertura é
  formado pelas primeiras quatro velas do M15; mercado abriu domingo 19h, as quatro primeiras
  velas formam o canal de abertura"]. Termo "quatro velas" aparece em **33 transcrições**.
  - Existe **CA semanal** (início da semana) **e CA diário** ("canal de abertura todo dia") [k7Jtalap-7Q].
- **CICLOS (C1, C2, C3...):** **clones aritméticos** da altura do CA, projetados na direção do
  rompimento. "Pega o tamanho exato do canal e clona para cima → ciclo 1; clona de novo → ciclo 2"
  [oCCOa4_awiE]. **C1 = a expansão do CA para cima OU para baixo.**
- **CANAL DE REFERÊNCIA (CR) + ZONA NEUTRA (ZN):** termos usados em **27/27** das transcrições de
  mecânica. Marcelo usa **CR e CA de forma quase intercambiável** ("o primeiro canal eu chamo de
  referência" [ginFNrsiX5A]); o CR é a caixa-base atual e a **ZN é o canal IMEDIATAMENTE ATRÁS**
  (o que ficou para trás após um rompimento), região de "respiro/não-operação" [07-fimathe-forex].
  - **A spec anterior já tinha a ZN certa** (canal adjacente do mesmo tamanho, NÃO o meio do canal) — **[CONFIRMADO]**.
- **Tamanho do CR é DISCRICIONÁRIO** [oy96SQ_L_wY: "não tem regra fixa de pontos; tem que encaixar
  perto do preço, proporcional, nem largo nem estreito... é uma linha de raciocínio, individual;
  ninguém ficou famoso copiando o desenho dos outros"]. Marcar por **corpo ou sombra é "pessoal e
  individual"** [QB2yIFRdd_w].
- **LINHA DO EQUADOR:** o **50% (e 50% do 50%, recursivo) de um canal macro** [v8rNtABQGhw,
  EveXzkrhINo]. Marcada no semanal/mensal. Dupla função: (a) **alvo** quando o preço rompe da
  abertura; (b) **ponto de reversão** quando o preço chega nela após um movimento grande ("voltou
  dois níveis"). Em **24 transcrições**.
- **FATIAMENTO ("feite"):** quando um canal macro é gigante (ex. barra elefante), corta-se ao
  meio (50%), e a metade ao meio de novo, criando subcanais operáveis [07-fimathe-forex, mmlFQbkE27Q].
- **KID BENGALA / BARRA ELEFANTE:** vela única de amplitude/volume brutal que cria novo paradigma
  (ex. XAU **1888/1762**, citado por ele [ginFNrsiX5A]); ditou ~3 anos. Em **10 transcrições**.

## Entrada [CORRIGIDO — a entrada PADRÃO é o ROMPIMENTO, não os "50%"]
**RESOLVE A INCERTEZA ABERTA Q1.** A entrada **canônica ensinada** é o **ROMPIMENTO** da
estrutura, **antecipado no M1**:
- **Regra textbook:** "**a regra da Fimathe: entrada NO ROMPIMENTO. Canal de referência, zona
  neutra, topzinho fora da caixinha, e quando rompeu, sobe o subciclo**" [sKnrcI9YN8s].
- **Sequência da semana** [oCCOa4_awiE, ginFNrsiX5A]: mercado abre → forma o **CA** → forma **C1**
  (não opera o C1, "mercado acabou de abrir") → **quando C1 rompe, executa** a favor do rompimento
  → busca base de C2/C3.
- **Antecipação no M1:** como **1 vela M15 = 15 velas M1**, ele entra no M1 **antes** do fechamento
  da vela M15 que vai romper [ginFNrsiX5A, sKnrcI9YN8s].
- **Os "50%" NÃO são a entrada padrão** — são (a) a **entrada no 50% do 2º ciclo** (variação que
  ele faz quando o C1 já rompeu: "entrei no 50% do segundo ciclo porque rompeu o primeiro"
  [ginFNrsiX5A]); e (b) a régua do **trailing/alvo** ("deixa pro santo"). A spec anterior tratava
  "pullback aos 50%" como a Regra A padrão — **incorreto**.
- **Filtro de tendência macro: ENFRAQUECIDO** [CORRIGIDO — era "só opera a favor do macro"].
  Em 2026 ele diz: "**se o macro está em alta mas no micro dá venda para mim, eu vendo, porque
  sigo o setup**" [k7Jtalap-7Q]. O macro orienta o *sentimento*, mas o **setup do micro decide**.

## Stop e Alvo [CONFIRMADO + detalhado]
- **STOP "fora da caixinha":** SEMPRE fora da ZN (atrás de um canal inteiro de margem), nunca
  colado. Em **20 transcrições**. Protege contra o "violino"/caça de liquidez. **Regra binária:**
  enquanto o preço não TOCAR fisicamente a linha de stop, a tese continua viva — não zerar no dedo
  [v8rNtABQGhw, 07-fimathe-forex]. Alta: stop abaixo da ZN/C1. Baixa: acima.
- **ALVO = projeção de ciclos (clones do canal), que SÃO as expansões de Fibonacci** [CONFIRMADO].
  **RESOLVE A INCERTEZA ABERTA Q2.** [QB2yIFRdd_w: "**os meus canais foram totalmente inspirados
  em expansão e retração de Fibonacci**"]. Marcelo abandonou o Fib tradicional (entrar no 23,6%,
  alvo 50%/61,8%) porque o **CR+ZN dá entrada antecipada** e é "mais objetivo".
  - **Take de 1 nível** = medir o(s) canal(is) e projetar **1 canal** a partir da entrada (= ~100%
    de expansão). **Take de 2 níveis** = **2 canais** (= ~200%). [J9x9J9Vue8k, ginFNrsiX5A].
  - **A escolha 1 vs 2 níveis é DISCRICIONÁRIA / de gestão**, NÃO um sinal de gráfico: "**no
    domingo eu NÃO sei se o mercado vai buscar 1 ou 2 níveis**; coloco 2 níveis quando tenho margem
    (venho acertando, posso 'abusar')" [J9x9J9Vue8k]. Take=1 para iniciante (bate mais, consistência).
  - **"Deixar pro santo":** colocar o alvo **um pouco AQUÉM** do nível (100%/200%), porque o preço
    tende a vir no meio do caminho e voltar [ginFNrsiX5A, J9x9J9Vue8k].
  - Expansões usadas como parciais: **100% / 161,8% / 200%** (Fib como RÉGUA de projeção).

## Sub Ciclo de Proteção (trailing) [CONFIRMADO + corrigido no detalhe]
Após a entrada, projeta-se um subciclo a partir da linha de entrada e **trava-se no 0a0
(breakeven)** quando o subciclo rompe [sKnrcI9YN8s, ddXTlW_fFug]. Detalhes reais:
- O subciclo usa **DOIS canais (CR+ZN), não um** ("um canal é muito curto") [sKnrcI9YN8s].
- O 0a0 é confirmado por **confluência** (preço bate no 50, quebra a linha do Equador, rompe os
  dois níveis do M1) [ginFNrsiX5A].
- Execução **fractal**: executa no M1, **trava o 0a0 no M15** [sKnrcI9YN8s, ddXTlW_fFug].
- Pode ser **por fechamento de vela (manual) ou por preço (Robô Fimathe)** [sKnrcI9YN8s].

## Virada de Mão [NOVO — não estava na spec]
[oCCOa4_awiE, oCCOa4_awiE] Só na **PRIMEIRA posição da semana**: se o trade do rompimento de C1
toma stop (fechamento além do CA/C1 contra você), **abre IMEDIATAMENTE a posição contrária**
(a força institucional inverteu). Ele **dobra/triplica o lote** na virada (mas diz para iniciante
NÃO fazer). **NÃO faz virada de mão no 4º nível** (tarde no movimento, qua/qui na máxima). Em 6 transcrições.

## PCM — Padrão de Candle [NOVO — corrige a divergência #8]
PCM é um **setup de leitura de candle/comportamento** nomeado por ele, parte do leque ("setup de
abertura, setup diário, PCM, setup das 3h/4h") [k7Jtalap-7Q, T-0eBp_JgQQ, _BaLT-9zzwU]. Em 7
transcrições da série de curso. NÃO é "score de força de candle" do nosso engine, nem é a filosofia
dos ciclos — é um **setup discricionário-visual específico** que merece estudo próprio se for usado.

## Disjuntores psicológicos [CONFIRMADO]
- **Parar a semana após 2-3 stops consecutivos** (protege o "capital mental").
- **Afastamento físico** após o clique (sair da frente do gráfico).
- **Síndrome do dedo nervoso** elimina 99%; "a frieza está em executar o plano apesar da dor".

---

## VEREDITO: MECÂNICO vs DISCRICIONÁRIO [a entrega central]
**O próprio Marcelo afirma (2026) que o NÚCLEO é MECÂNICO, não discricionário** [k7Jtalap-7Q,
adXpiMMMJHo]: *"setup é mais lucrativo que discricionário, e muito mais... hoje eu não uso o
discricionário praticamente para nada... descobri esses setups quando comecei a programar e
automatizar a técnica"*. Ele vende um **Robô Fimathe** que executa a técnica e ensina a programá-lo.

Estimativa de mecanizabilidade por componente (peso no edge):
- **MECÂNICO (~70%):** canal de abertura (4 velas M15), projeção de ciclos (clones), stop fora da
  ZN, take 1/2 níveis (clones), trava 0a0 do subciclo, antecipação M1 (fractal), disjuntores
  2-3 stops, relógio da semana (domingo→quinta). **Tudo isto é determinístico e backtestável.**
- **DISCRICIONÁRIO (~30%):** (a) **tamanho/encaixe do CR** ("proporcional, perto do preço" — julgamento);
  (b) **escolha de 1 vs 2 níveis** (gestão por win-streak, não sinal); (c) **leitura de barra
  elefante e da linha do Equador** como reversão vs continuação; (d) **decisão da virada de mão**;
  (e) **PCM e padrões de lateralização** (leitura visual de candle).

**Conclusão:** o edge **NÃO depende criticamente do discricionário** — Marcelo trata o discricionário
como algo que ele *abandonou*. Um bot fiel ao **canal de abertura + ciclos + stop fora da ZN +
take por clones + trava 0a0** captura a maior parte. As partes discricionárias afetam *seleção e
sizing*, não a geometria. Um backtest honesto das partes mecânicas é legítimo — **desde que no
instrumento e timeframe certos (forex/XAU intraday M15/M1), não cripto/ações intraday rolante.**

## DIVERGÊNCIAS — reavaliadas contra os VÍDEOS (não só vs engine.py)
| # | Spec/engine dizia | Vídeos dizem | Status |
|---|---|---|---|
| 1 | ZN = meio do canal (engine) | ZN = canal adjacente atrás | engine ERRADO; spec/`canonical.py` certo **[OK]** |
| 2 | entrada no rompimento do topo (engine) | entrada no rompimento do **CA/C1** (M1-antecipado) | engine "rompimento" estava **mais perto** do certo que a Regra A da spec **[CORRIGIDO]** |
| 3 | só opera a favor do macro | em 2026 o **micro/setup decide**; macro = sentimento | spec **forte demais** **[CORRIGIDO]** |
| 4 | falta multi-TF | **semanal→M15, diário→M1** (não D1→H4) | TF da spec **errado** **[CORRIGIDO]** |
| 5 | stop canal±ATR (engine) | stop **fora da ZN** | spec/`canonical.py` certo **[OK]** |
| 6 | alvo R:R fixo + retrações (engine) | alvo = **clones de canal = expansões Fib** (1/2 níveis) | spec direção certa; detalhe = clones **[OK/refinado]** |
| 7 | falta subciclo | subciclo trava **0a0**, fractal M1→M15 | **[OK, detalhado]** |
| 8 | PCM = score de candle (engine) | PCM = **setup de leitura de candle** nomeado | engine ERRADO; PCM é setup próprio **[CORRIGIDO]** |
| 9 | canal por rolling/pivots ≠ topo/fundo da perna | **CR = canal de abertura (4 velas), não swing rolante NEM "perna da tendência"** | **AMBOS** (engine E spec/`canonical.py`) ERRADOS **[CORRIGIDO — crítico]** |
| 10 | swap ignorado | **é DAY TRADE, swap irrelevante** | a preocupação com swap era **infundada** **[CORRIGIDO]** |

## IMPACTO NO BACKTEST / O QUE REIMPLEMENTAR (não escrever código agora)
1. **`fimathe/canonical.py` modela o CR ERRADO** (linhas 57-87, 174-186: `_causal_pivots` =
   swing-high/low de "perna da tendência", `swing_lookback=8`). O CR FIMATHE é o **canal de
   abertura das 4 primeiras velas M15 ancorado à abertura (semana/dia)** + **ciclos = clones
   aritméticos** — NÃO pivôs rolantes. Um motor fiel precisa: detectar a abertura, medir a caixa
   de 4 velas, clonar C1/C2/C3, e operar o rompimento de C1 com antecipação intrabar.
2. **O instrumento está errado para a técnica nativa.** `simulation/breakout_tribunal.py` julga
   cripto; `fimathe_forex*.py` é o caminho certo. Marcelo é explícito: **intraday M15/M1 é para
   FOREX/XAU**; para **cripto** ele usa **swing macro D/W/M com fatiamento 50%** (refém =
   referência+tendência), **nunca intradiário** [mmlFQbkE27Q]. → O veredito "beta de bull" do
   tribunal de cripto pode refletir **mismatch instrumento×método**, não ausência de edge.
3. **Remover swap do custo** do backtest canônico (é day trade) — mas **manter spread**.
4. **Re-testar** o motor canônico CORRIGIDO (canal de abertura + ciclos) em **dados forex/XAU
   M15+M1** (já há `data/forex_cache/` e `simulation/fimathe_forex_canonical.py`), medindo **alpha
   vs buy&hold** (não Sharpe vs zero), com a variante de entrada = **rompimento de C1** (não 50%).
