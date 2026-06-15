---
name: integracao
description: Integração das camadas (pacote integration/) — o DecisionEnricher compõe FimatheEngine (features), síntese (convicção/conflito), ML (P(win) se promovido) e sizing dinâmico num único ponto. Degrada com graça. Inclui o snippet de wiring no agente de decisão.
metadata:
  type: reference
---

# Integração — o "cérebro" que liga as camadas

> **O que muda:** as camadas deixam de ser ilhas. O `DecisionEnricher` recebe um intent + o contexto
> disponível (barras, sinais, regime, modelo promovido) e devolve, num só lugar: **features ricas**
> (FimatheEngine), uma **convicção** com **conflito** (síntese), um **P(win)** (ML, se promovido) e a
> **quantidade** dimensionada por essa confiança (sizing). É a Camada 1+2+3+sizing operando juntas.

> ✅ **Implementado** em [`integration/enricher.py`](../../integration/enricher.py), testado
> (`tests/test_integration_enricher.py`). Pacote **isolado** (não importa o `intelligence/engine.py`),
> dependendo só das camadas estáveis.

## 1. Fluxo do `DecisionEnricher.enrich(...)`

```
barras OHLC ──FimatheEngine──► features (pcm_score, dist_to_zn, rsi, adx…) + sinal técnico
sinais smart money ───────────► Views
regime ───────────────────────► View (prior fraca)
features+regime ──ML(promovido)► P(win) ──► View         (só se champion/challenger promoveu)
        todas as Views ──síntese──► direção + convicção + CONFLITO + racional
        convicção(alinhada ao lado) ⊕ P(win) ──► confiança [0..1]
        confiança ──DynamicSizer──► quantidade
```

Saída (`EnrichedDecision`): `confidence`, `qty`, `context` (p/ o DecisionLog), `synthesis`, `p_win`,
`fimathe_signal`.

## 2. Decisões de design (importantes)

- **Sem vazamento no ML.** As features do ML são só o que se conhece **antes** da decisão
  (`signal_strength` + features da FimatheEngine + regime). `score`/`conviction` são **saídas** — usá-las
  como entrada seria leaky e circular (o ML alimenta a síntese que gera a convicção). O **mesmo
  vetorizador** (`ml.dataset.context_to_features`) serve treino e predição → **sem train/predict skew**.
- **Degradação graciosa.** Sem barras → sem features FimatheEngine; sem modelo promovido → sem P(win);
  sem sinais → só regime. Sempre produz uma decisão coerente.
- **Confiança = P(win) estimado.** É o que o sizing trata como probabilidade no Kelly. Síntese neutra →
  0.5; a favor → sobe; contra → desce (e o sizing zera abaixo do floor).
- **Nunca cria trades.** Só enriquece/score/sizing. A regra "sinais não viram trade" é preservada.

## 3. Wiring no agente de decisão (o hook)

O agente de decisão (hoje no pipeline `bus`) chama o enricher por intent e usa `qty`/`context`:

```python
from integration.enricher import DecisionEnricher
enricher = DecisionEnricher(sizer=..., classifier=promoted_model_or_None)

# por intent, no ciclo de decisao (apos classificar o regime):
enr = enricher.enrich(
    symbol=symbol, side=side, strategy=strategy, regime=regime,
    equity=account.equity, entry_price=entry, stop_price=stop,
    signals=signals, ohlc=bars_df, signal_strength=strength,
)
intent.qty = enr.qty                      # sizing dinamico
# registra a decisao com o contexto enriquecido (features + sintese):
decision.context.update(enr.context)
if enr.qty <= 0 or enr.synthesis.conflict:
    ...                                    # skip / reduzir, conforme politica
```

> **Estado deste wiring:** o `DecisionEnricher` está pronto e testado. O hook acima vive no agente de
> decisão (`agents/bus_agents.py` / `intelligence/engine.py`), que está sob edição ativa da outra frente —
> aplicar a chamada lá é a última costura, feita quando aquele arquivo estabilizar (escrever concorrente
> corromperia o trabalho em andamento).

## 4. Próximos passos

- Aplicar o hook no agente de decisão (acima).
- Passar `ohlc` real (barras OHLC do IngestionAgent) ao enricher para ligar as features da FimatheEngine ao
  vivo.
- Quando o champion/challenger (doc 11) recomendar promoção de forma estável, injetar o `SetupClassifier`
  promovido no enricher (começando como canary).

## Como virar skill

Extrair `integracao` com: o contrato do `DecisionEnricher`, o fluxo de composição, as decisões de design
(sem leak no ML, degradação graciosa, confiança=P(win)) e o snippet de wiring. Combina
[08](08-fimathe-engine.md)+[10](10-camada-decisao.md)+[11](11-camada-ml.md)+[12](12-sizing-dinamico.md)+[13](13-camada-sintese.md).
