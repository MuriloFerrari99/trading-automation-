---
name: camada-decisao
description: Camada de decisão inteligente (pacote intelligence/) — fecha o loop de feedback: classifica regime, veta combos estratégia×regime com edge negativo comprovado e modula confiança com sinais de smart money, registrando toda decisão. Wiring opcional no LocalOrchestrator/main.
metadata:
  type: reference
---

# Camada de Decisão — gate por regime + uso do track record

> **O que torna o bot "mais inteligente" aqui:** ele passa a **decidir com base no próprio
> histórico**. Entre o Planejador (gera intents) e o Executor (executa), esta camada classifica o
> **regime**, consulta o `DecisionLog` (Camada 0, doc [09](09-loop-de-feedback.md)) e **veta combinações
> estratégia×regime que vêm perdendo**. É o passo que fecha o loop: registrar → medir → **agir sobre a
> medição**.

> ✅ **Implementado** no pacote [`intelligence/`](../../intelligence), com testes
> (`tests/test_intelligence_*.py`, `tests/test_feedback_combo.py`). Wiring **opcional e retrocompatível**:
> sem ele, o comportamento é exatamente o anterior.

## 1. Componentes

| Módulo | Papel |
|---|---|
| `intelligence/decision_policy.py` | `DecisionPolicy` — regra **pura** de veto/permissão a partir das estatísticas do combo |
| `intelligence/engine.py` | `DecisionIntelligence` — orquestra regime + política + registro a cada ciclo |
| `feedback/evaluation.py` (estendido) | `Report.by_strategy_regime` + `combo(strategy, regime)` — granularidade que o gate usa |

## 2. Fluxo, por intent

```
intent ─► classifica regime do ativo (buffer de precos rolante + classify_regime)
       ─► mede sinal de smart money ALINHADO (mesmo ativo+lado) → signal_strength
       ─► consulta combo estrategia@regime no DecisionLog (Report.combo)
       ─► DecisionPolicy.evaluate_combo(...) → allow / block
       ─► registra Decision (allow→open buy/sell | block→skip) com contexto
       ─► allow ? mantem intent : descarta
```

## 3. A política (conservadora de propósito)

- **Sem amostra suficiente** (combo com poucos trades fechados, default `< 12`) → **permite**
  (inocente até prova em contrário). Nada de bloquear por ruído de 1-2 trades.
- **Com amostra suficiente E expectancy < limiar** (default `< 0`) → **veta** e registra `skip`. O sistema
  aprende a **não repetir o que perde**.
- O `signal_strength` (sinal de smart money alinhado) entra **só no score** (prioriza/loga), **nunca
  cria** um trade — a regra do projeto ("sinais não executam") é preservada: eles apenas ponderam intents
  que uma estratégia já produziu.

## 4. Regime sem API de barras

O broker expõe `get_last_price()`, não barras históricas. Solução pragmática: o engine mantém um **buffer
rolante de preços por símbolo**, amostrado a cada ciclo do Monitor; o regime **emerge** das amostras
acumuladas. Enquanto não há série suficiente, o regime é `UNKNOWN` (e o gate opera só na dimensão
histórica). Quando um feed de barras for plugado, basta passar um `price_provider`/série melhor — a
interface não muda. *(Limitação: amostragem ~1 ponto/tick; é um regime de baixa frequência, adequado ao
MVP.)*

## 5. Wiring

- **`LocalOrchestrator(planner, executor, *, intelligence=None)`** — quando `intelligence` é passado, os
  intents do Planejador passam por `process()` antes do Executor.
- **`orchestration/factory.build_orchestrator(..., intelligence=None)`** — repassa ao LocalOrchestrator.
- **`main.build_app()`** — constrói `DecisionLog(connection=db.conn)` (mesma tabela `decisions`, mesmo
  SQLite) + `DecisionPolicy()` + `DecisionIntelligence(price_provider=broker.get_last_price)` e injeta no
  orquestrador. **Já ativo no entrypoint** (modo paper).

## 6. Efeito hoje vs. com dados

- **Hoje (sem histórico fechado):** o gate **permite tudo** (conservador) e **registra toda decisão com
  regime** — começa a montar o dataset imediatamente. Nenhuma regressão.
- **Com o tempo (outcomes acumulados):** combos perdedores são **suprimidos automaticamente**. Ex.: se
  `trailing_stop@high_vol` provar expectancy negativa em ≥12 trades, o bot para de operar essa estratégia
  nesse regime — sozinho.

## 7. Próximo passo que destrava o aprendizado

O elo que falta é **anexar outcomes**: quando uma posição fecha (stop/alvo), gravar o P&L realizado via
`DecisionLog.attach_outcome_by_client_order_id(coid, ...)`. Isso vive no Monitor/reconciliação e depende do
P&L de posição fechada do broker (área em evolução pela outra frente do projeto). Sem outcomes, o gate
nunca bloqueia (seguro); **com** outcomes, ele fica afiado. Ver doc [09](09-loop-de-feedback.md) §5.

## Como virar skill

Extrair `camada-decisao` com: o contrato de `DecisionPolicy` (regra de veto), o fluxo do
`DecisionIntelligence`, a regra de ouro ("sem dados → permite; dados ruins comprovados → veta; sinais só
pontuam") e o ponto de wiring. Combinar com [09-loop-de-feedback](09-loop-de-feedback.md) (dados) e
[05-arquitetura-agentes](05-arquitetura-agentes.md) (onde encaixa no ciclo).
