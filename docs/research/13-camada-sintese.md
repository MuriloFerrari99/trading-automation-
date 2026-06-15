---
name: camada-sintese
description: Camada 3 — síntese (pacote synthesis/). Combina visões ortogonais (técnico, smart money, ML, regime, notícias) numa convicção unificada com medida de consenso/conflito e racional legível; hook opcional de LLM (Reasoner) sem acoplar o loop a nenhuma API.
metadata:
  type: reference
---

# Camada 3 — Síntese de visões (ensemble + reasoning)

> **O que fica mais inteligente:** em vez de cada sinal agir isolado, o sistema passa a ter uma **visão
> unificada por ativo** — combinando fontes **ortogonais** (técnico, smart money, ML, regime, e no futuro
> notícias/sentimento), medindo **consenso vs conflito**, e produzindo um **racional legível**. É a camada
> qualitativa no topo da pilha.

> ✅ **Implementado** em [`synthesis/`](../../synthesis), testado (`tests/test_synthesis.py`). Núcleo
> **determinístico** (sem dependência de LLM); o LLM é um **hook opcional injetado**.

## 1. Componentes

| Módulo | Papel |
|---|---|
| `synthesis/views.py` | `View` (fonte, direção, confiança, peso, racional) + `Direction` + adaptadores (`view_from_signal`, `view_from_proba`) |
| `synthesis/synthesizer.py` | `synthesize()` / `Synthesizer` — combina views → `SynthesisResult` |
| `synthesis/reasoner.py` | `Reasoner` (interface) · `TemplateReasoner` (padrão, determinístico) · `LLMReasoner` (hook opcional) |

## 2. Como combina

Cada `View` contribui com **peso × confiança × sinal(direção)**. O resultado:
- **direção agregada** (long/short/neutral), com uma **banda neutra** (`neutral_band`) que evita ruído;
- **convicção** 0..1 = força do consenso líquido;
- **agreement** = fração da "massa" (peso×confiança) que aponta na direção final → **detecta conflito**
  (`conflict=True` quando o agreement fica abaixo de `conflict_threshold`, mesmo com direção definida);
- **contributions** = quanto cada fonte puxou (auditoria);
- **rationale** = frase explicativa.

Pesos por fonte são configuráveis (`Synthesizer(source_weights=...)`) — uma fonte confiável pode até
**inverter** a direção agregada.

## 3. O hook de LLM (sem acoplar o loop)

`Reasoner` é uma interface. O padrão (`TemplateReasoner`) gera o racional **deterministicamente** (testável,
offline). Para narrativa qualitativa (interpretar notícias, earnings, contexto macro), injete um
`LLMReasoner(call_fn)` onde `call_fn(prompt) -> str` é **qualquer LLM que você plugar** (ex.: Claude API).
Se a chamada falhar, **cai no template** — o ciclo de trading **nunca** depende de um LLM disponível.

```python
from synthesis import synthesize, view_from_signal, LLMReasoner

views = [view_from_signal(sig, weight=1.0), View("fimathe", Direction.LONG, 0.7)]
res = synthesize("AAPL", views, reasoner=LLMReasoner(meu_llm))   # ou sem reasoner (template)
print(res.direction, res.conviction, res.conflict, res.rationale)
```

## 4. Integração (plano)

A síntese é uma camada de **score/visão**, não de execução (respeita "sinais não viram trade"). Onde
encaixa:
- Alimentar a `DecisionIntelligence` (doc 10): a **convicção** e o **conflito** modulam o score do gate
  (ex.: conflito alto → reduzir tamanho via doc 12, ou exigir score maior).
- **Fontes ortogonais** a registrar como Views: estratégias (FimatheEngine, doc 08), smart money (doc 03),
  P(win) do ML (doc 11), regime (doc 09). Diversidade > redundância.
- **Orthogonal sweep** (notícias/sentimento) entra como mais uma View quando houver fonte — sem mudar o núcleo.

## 5. Próximos passos

- Wire da síntese no gate (convicção/conflito → score e sizing).
- Provedor de **notícias/sentimento** como View (e aí o `LLMReasoner` brilha).
- Calibrar pesos por fonte a partir do histórico (quais fontes acertam mais, por regime — usa Camada 0).

## Como virar skill

Extrair `camada-sintese` com: o modelo `View`, a mecânica de combinação (peso×confiança×direção, agreement,
conflito), o **contrato `Reasoner`** e a regra de ouro (**núcleo determinístico; LLM é hook opcional com
fallback**). Combinar com [10-camada-decisao](10-camada-decisao.md), [03-smart-money-signals](03-smart-money-signals.md)
e [11-camada-ml](11-camada-ml.md).
