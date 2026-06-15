---
name: fimathe-forex
description: Metodologia FIMATHE (forex/ouro) — geometria de canais a partir da abertura, zona neutra, virada de mão, linhas do Equador, barra elefante e fatiamento, com as regras de risco e psicológicas; e como adaptá-la de ouro para pares de moedas.
metadata:
  type: reference
---

# FIMATHE — Geometria de Canais para Forex / Ouro

> **Proveniência.** Este documento foi extraído da transcrição do áudio
> **"Geometria da Fimathe no ouro"** (`docs/research/fontes/fimathe-ouro-transcricao.txt`),
> um material em **formato podcast (2 narradores)** que sintetiza aulas e sessões ao vivo da
> metodologia. O nome aparece distorcido na transcrição ("FIMAT", "FIMATIF", "FIMA-FACE"); o correto é
> **FIMATHE** (metodologia discricionária de trading do brasileiro Marcelo Ferreira). Os números citados
> (ex.: "~1000 pontos por nível no ouro", drawdown de "-40k → +60k") vêm do áudio como **ilustração
> anedótica**, não como parâmetros calibrados — trate-os como exemplo, não como regra.

> ⚠️ **Nota de escopo — leia antes de usar.** A FIMATHE é uma metodologia **discricionária** de
> **forex e ouro (XAU/USD)**, operada em **MetaTrader (MT4/MT5)** sobre pares de moedas. O bot deste
> repositório opera via **Alpaca**, que **não oferece pares de moedas nem XAU/USD** (só ações/ETFs/opções
> e cripto dos EUA). Portanto: este doc é **conhecimento de metodologia**, não um módulo plugável no bot
> atual. Para automatizar FIMATHE de verdade seria preciso **outro broker/feed** (corretora forex com API,
> ou ponte para MT5). Ver §9. A maior parte do método é **discricionária/visual** — só um subconjunto é
> determinístico o suficiente para virar código.

---

## 1. Filosofia central

A FIMATHE rejeita o "painel de Boeing" (médias móveis, RSI, bandas) e adota um **minimalismo quase
monástico**: tela limpa, **apenas o preço puro**. A premissa é que o mercado é caótico na superfície, mas
o movimento institucional deixa uma **estrutura geométrica** que pode ser **mapeada antes** do preço se
mover — não para "adivinhar o futuro", mas para **projetar a escada** que o preço terá que subir/descer
caso entre em tendência.

> Frase-síntese do material: *"construir uma prisão geométrica em torno das possibilidades"* + *"o macro
> sempre dita as regras do micro"* + *"o verdadeiro trunfo não é dobrar o mercado, é dobrar a si mesmo"*.

Três pilares: **(1)** geometria de canais, **(2)** gestão de risco rígida, **(3)** disciplina psicológica
(o pilar que, segundo o material, elimina 99% das pessoas).

---

## 2. Canal de Abertura (o ponto de partida)

- No **gráfico de 15 minutos (M15)**, olha-se as **4 primeiras velas** logo na **abertura do mercado no
  domingo à noite**. Elas formam a **"caixinha" inicial** (o *canal de abertura*).
- **Analogia do material:** é como pintar as linhas do campo de futebol *antes* do jogo. Quando a bola
  rolar solta (terça/quarta), o terreno já está demarcado — não se corre atrás do preço.
- **Ressalva do próprio material:** o campo de futebol é estático; o mercado é um **campo vivo e
  predatório**, cujas linhas "tentam enganar o jogador". Por isso as regras de stop (§4) são tão rígidas.

### Por que medir no domingo (volume baixo)?
Mecânica estatística: **em >90% das vezes, o fechamento de sexta fica muito mais próximo da máxima ou da
mínima da semana do que do preço de abertura de domingo.** Ou seja, o mercado **raramente fica "zero a
zero"** — ele precisa **buscar liquidez, respirar e se expandir**. Quarta e quinta costumam ser os dias em
que o preço testa os limites dessa expansão. Projetar os canais no domingo = **desenhar as rodovias antes
dos carros entrarem nelas**.

---

## 3. Projeção dos ciclos (replicação geométrica)

1. Mede-se o **tamanho exato** do canal de abertura.
2. **Clona** essa medida para cima → **Ciclo 1**.
3. Clona de novo → **Ciclo 2**. (E assim por diante, simetricamente para cima e para baixo.)

É **pura replicação geométrica**: a altura da caixinha inicial vira a unidade de medida de toda a "escada"
de níveis projetados para a semana. Os alvos e stops (§4) são definidos em função desses níveis.

---

## 4. Gestão de risco (a "matemática da sobrevivência")

### 4.1. Take (alvo de lucro): 1 nível vs 2 níveis
- **Take de 1 nível** — alvo = o tamanho de **uma** caixinha/canal projetado. É a "espinha dorsal" do
  iniciante. Parece pouco, mas mantém a conta positiva no longo prazo. *(No ouro, o material diz que 1
  nível de expansão ≈ "1000 pontos" — exemplo, não regra.)*
- **Take de 2 níveis** — alvo mais distante (2 canais), para quem já tem leitura/consistência.

### 4.2. Stop "fora da caixinha" e a **Zona Neutra**
Regra inegociável: **a trava de segurança (stop loss) NUNCA fica dentro do canal de negociação atual, nem
na "zona neutra".**
- **Zona neutra:** quando o preço **rompe** a caixinha e entra num **novo canal**, o canal que ficou
  **imediatamente para trás** vira a *zona neutra* — o espaço onde o preço pode **flutuar/respirar sem que
  isso signifique reversão**.
- O stop fica **atrás de um canal inteiro de margem** (fora da zona neutra). Isso **aumenta o risco
  financeiro por operação**, mas **blinda a ordem contra a "caça"**.

### 4.3. O "violino" (stop hunt / caça de liquidez)
O **violino** é a oscilação que **bate no stop mal posicionado e logo depois o preço sobe lindamente para
o alvo**. Segundo o material, **não é azar — é mecânica**: instituições e **robôs de alta frequência sabem
onde a massa de amadores esconde os stops** e empurram o preço para essas zonas de propósito, acionando os
stops para **gerar a liquidez** que elas precisam para montar posição na direção oposta. Stop fora da zona
neutra = não virar "estatística de liquidez".

---

## 5. A "Virada de Mão" (o gatilho polêmico)

**Regra:** se o mercado invadir a zona neutra, **fechar fora do canal** e acionar o stop técnico, o trader
deve **imediatamente abrir uma nova operação no sentido oposto** — partindo da leitura de que **a força
institucional da semana inverteu**.

**Virada de mão ≠ trading de vingança** (distinção central do material):

| | Trading de vingança (destrói contas) | Virada de mão (FIMATHE) |
|---|---|---|
| Origem | Reativo, emocional, ego não aceita a perda | **Pré-planejado** ("contrato consigo mesmo" antes do trade) |
| Estrutura | Caótico; **dobra a posição no mesmo lugar** | Obedece à **geometria fria** traçada no domingo |
| Gatilho | "rezar pro preço voltar" | Preço **varreu a zona neutra e rompeu a fronteira do stop** |

> O ponto-chave: *"a frieza não está em não sentir dor; está em executar o plano apesar da dor."*

---

## 6. Macro dita o micro: Linhas do Equador, Barra Elefante e Fatiamento

### 6.1. Linhas do Equador
Linhas desenhadas nos **gráficos grandes (semanal/diário)**, marcando **fronteiras "tectônicas" de preço
onde o dinheiro institucional está estacionado**. Quando se desce ao **M1** (cheio de ruído), essas linhas
funcionam como **barreiras magnéticas**: avisam "se o preço chegar aqui, uma força grande vai intervir".
São o que impede operar "às cegas".

### 6.2. Barra Elefante (ou "de bengala")
Uma **única vela** de **amplitude brutal + volume obsceno** que "engole o histórico recente e cria um novo
paradigma de preço".
- **Estudo de caso do ouro (XAU/USD):** uma única **barra elefante semanal** entre **1878 e 1762** ditou
  **~3 anos** de operações — o preço subia por meses, mas ao testar aquelas fronteiras era repelido para
  dentro. (O material cita até traders **tatuando "1878 / 1762"** no corpo — virou dogma na comunidade.)

### 6.3. Fatiamento ("feite") e correção de 50%
Quando um canal (ex.: de uma barra elefante) é **gigante demais**, aplicar a regra de "stop fora da
caixinha" exigiria margem impraticável e alvos distantes demais. Solução:
1. **Passa a faca no meio do canal** → o **pullback/correção de 50%** (expansões violentas tendem a recuar
   até a metade para "ganhar fôlego" antes de continuar).
2. **Fatia essa metade em mais 50%**, criando **subcanais menores e operáveis**, com stops mais baratos e
   alvos acessíveis — **sem perder de vista a anomalia macro** que comanda o tabuleiro.

### 6.4. Fractalidade
O movimento se repete em todas as escalas. **Zoom macro (semanal) → direção; zoom micro (M1) → precisão
cirúrgica de entrada/saída.** "Mapa do continente → estado → cidade → a rua onde o dinheiro troca de mãos."

---

## 7. O pilar psicológico (onde 99% falham)

- **Síndrome do dedo nervoso:** a incapacidade de tolerar o desconforto de uma operação aberta. Fechar no
  "dedo" por pânico de uma vela vermelha = **jogar todo o gerenciamento de risco no lixo**.
- **Regra binária do stop:** enquanto o preço **não encostar fisicamente na linha de stop**, a tese
  original **continua viva**. (Caso "homem de gelo" no ouro: drawdown de **-40k** sem tocar o mouse → o
  preço achou liquidez, reverteu e fechou em **+60k**. Anedota ilustrativa.)
- **Analogia da dieta:** ter a estratégia perfeita ≠ ter disciplina para executar. "O melhor nutricionista
  te dá a dieta perfeita; ela não serve de nada se 3h depois você enche o rabo de coxinha e pizza."

### Contenção de danos (disjuntores que protegem o trader dele mesmo)
- **Parar a semana após 2–3 stops consecutivos:** encerra tudo e **desliga a plataforma pelo resto da
  semana**, não importa quão "bonito" o gráfico fique depois. O que se protege aí **não é o capital
  financeiro, é o capital mental** — um cérebro com 3 perdas seguidas opera em "modo vingança".
- **Afastamento físico:** depois do clique de abertura, **levantar da cadeira** (caminhar, dormir,
  qualquer coisa menos olhar o gráfico). O controle real **só existe antes do clique** (tamanho do risco +
  posição do stop); depois disso, o controle passa para o fluxo institucional.

---

## 8. Adaptação de OURO → PARES DE MOEDAS (forex)

A geometria da FIMATHE é, em tese, **agnóstica de ativo** (canais, ciclos, zona neutra, virada de mão
valem para qualquer série de preço). O que **muda** ao migrar de XAU/USD para pares de moedas (EUR/USD,
GBP/USD, USD/JPY etc.):

| Dimensão | Ouro (XAU/USD) | Pares de moedas | Implicação |
|---|---|---|---|
| **Unidade de preço** | "pontos" do ouro (muito grandes) | **pip** (4ª casa decimal; JPY: 2ª) | Recalibrar o tamanho da caixinha e do "1 nível" em pips, não em "pontos do ouro" |
| **Valor por pip / lote** | alto, volátil | depende do par e do lote (micro/mini/standard) | Position sizing precisa ser refeito por par |
| **Volatilidade** | extrema (barras elefante frequentes) | menor e mais "comportada" na maioria dos majors | Canais menores; "1 nível ≈ 1000 pontos" **não** transfere — medir empiricamente por par |
| **Sessões** | segue ouro/risco global | **Tóquio / Londres / Nova York** | A "abertura de domingo" e os horários de expansão (Londres/NY overlap) mudam o timing das 4 velas M15 |
| **Spread/custos** | maior | majors têm spread baixo | Stops "fora da caixinha" ficam relativamente mais baratos nos majors |
| **Drivers macro** | juros reais, risco, dólar | **diferencial de juros, bancos centrais, dados macro** | As "linhas do Equador" devem considerar níveis macro do par (ex.: defesa de banco central) |

**Regras práticas da adaptação:**
1. **Medir, não importar números.** Reconstrua o tamanho típico de caixinha e o "valor de 1 nível" **por
   par**, a partir do histórico — não reaproveite os "1000 pontos" do ouro.
2. **Ajustar o relógio.** As 4 velas M15 de referência e os dias de "expansão" devem respeitar as
   **sessões forex** (a maior liquidez é no overlap **Londres+NY**), não o calendário do ouro.
3. **Position sizing por pip.** Risco por operação em % da conta → converter para lote via valor do pip do
   par (ver doc [02-risco-execucao](02-risco-execucao.md), §position sizing).
4. **Manter as regras psicológicas intactas** — elas são agnósticas de ativo (dedo nervoso, 2–3 stops,
   afastamento físico).

---

## 9. Como (e se) automatizar — encaixe no nosso sistema

**Verdade desconfortável:** a FIMATHE é majoritariamente **discricionária/visual**. Boa parte ("ler" a
barra elefante, traçar linhas do Equador, julgar a virada de mão) depende de julgamento humano. Mesmo
assim, há um **subconjunto determinístico** que dá para codificar:

| Componente | Automatizável? | Como |
|---|---|---|
| Canal de abertura (4 velas M15) | **Sim** | Cálculo direto: high/low das 4 primeiras velas M15 da sessão |
| Projeção de ciclos | **Sim** | Múltiplos aritméticos da altura do canal |
| Stop fora da zona neutra | **Sim** | Regra geométrica determinística |
| Take 1/2 níveis | **Sim** | Múltiplos do canal |
| Contenção de danos (2–3 stops → para a semana) | **Sim** | Estado/contador no `data/` + kill switch (ver [02](02-risco-execucao.md)) |
| Virada de mão | **Parcial** | Gatilho é regra (rompeu fronteira do stop), mas exige confiança na leitura institucional |
| Linhas do Equador / barra elefante | **Difícil** | Detecção heurística (volume/amplitude anômalos), mas o "valor" é discricionário |

**Pré-requisitos de infraestrutura (que o bot Alpaca atual NÃO atende):**
- **Broker/feed de forex** com API (Alpaca não tem pares de moedas nem XAU/USD). Opções: corretora forex
  com REST/FIX, ou **ponte para MetaTrader 5** (a FIMATHE nasceu em MT4/MT5; há libs como `MetaTrader5`
  para Python).
- Dados **M15 e M1** intradiários do par, com sessões/horários corretos.

**Se um dia for automatizar**, o encaixe na arquitetura multi-agente (doc [05](05-arquitetura-agentes.md)) seria:
- **Planejador** — no "domingo" (abertura da semana), calcula canal de abertura, projeta ciclos, marca zona
  neutra e níveis de take/stop → emite *intenções* condicionais.
- **Executor** — envia ordens ao broker forex/MT5; gerencia a virada de mão como ordem pré-programada.
- **Monitor** — vigia o preço vs níveis, aplica a regra binária do stop e o disjuntor de 2–3 stops/semana.

> **Recomendação honesta:** trate este doc como **referência conceitual e, no máximo, base para um
> _backtest_ das partes determinísticas** (canal/ciclos/stop/take) num feed de forex — **não** como um
> módulo do bot Alpaca. Antes de qualquer capital real, validar em backtest + paper (doc [06](06-backtesting-testes.md)).

---

## Como virar skill

Extrair uma skill `fimathe-forex` com: **(1)** as regras determinísticas (canal de abertura, projeção de
ciclos, stop fora da zona neutra, takes de 1/2 níveis, disjuntor de 2–3 stops) como um *checklist/algoritmo*
parametrizável por par; **(2)** a tabela de adaptação ouro→pares (§8) como guia de calibração; **(3)** o
checklist psicológico (§7) como "guardrails". Deixar explícito na skill que a parte discricionária (linhas
do Equador, leitura de barra elefante, decisão da virada de mão) **não** é totalmente automatizável e que o
broker precisa suportar forex/MT5.
