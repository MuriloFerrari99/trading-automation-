---
name: sizing-dinamico
description: Sizing dinâmico (pacote sizing/) — o tamanho da posição varia com a convicção (P(win)/score) via Kelly fracionário, com pisos/tetos e sem operar sem edge. Compõe as primitivas de risk/sizing.
metadata:
  type: reference
---

# Sizing Dinâmico — tamanho por edge/confiança

> **O que fica mais inteligente:** o bot deixa de arriscar o mesmo em todo trade e passa a **arriscar mais
> quando tem mais convicção e menos quando tem menos** — e **nada quando não tem edge**. É o complemento
> natural da Camada 2 (a confiança vem do score da decisão / P(win) do ML).

> ✅ **Implementado** em [`sizing/dynamic.py`](../../sizing/dynamic.py), testado (`tests/test_sizing_dynamic.py`).
> **Compõe** as primitivas de [`risk/sizing.py`](../../risk/sizing.py) (fixed fractional, cap de exposição)
> — aqui decidimos *quanto* arriscar; lá converte risco → quantidade.

## 1. Pipeline

```
confiança (P(win)) + razão R:R  ──Kelly fracionário──►  risco% do equity
risco% + equity + entry/stop     ──risk.sizing────────►  quantidade
quantidade                       ──cap de exposição───►  qty final
```

## 2. Kelly fracionário (e por que fracionário)

`kelly_fraction(p, b) = (p·b − (1−p)) / b`, com `b` = payoff (ganho/perda). O Kelly **cheio** maximiza o
crescimento de longo prazo, mas tem **drawdowns brutais** e é hipersensível ao erro na estimativa de `p`.
Por isso usamos uma **fração** dele (`kelly_cap`, default **¼ Kelly**) — padrão de mercado para suavizar a
variância.

Propriedade-chave: se não há edge (`p ≤ 1/(1+b)`; ex.: R:R 2 → `p ≤ 0.333`), `kelly_fraction = 0` → **qty
0**. O dimensionamento **desliga sozinho** quando a vantagem some.

## 3. `DynamicSizer`

| Parâmetro | Default | Papel |
|---|---|---|
| `kelly_cap` | 0.25 | fração do Kelly cheio (¼ Kelly) |
| `max_risk_pct` | 0.02 | **teto** de risco por trade (2% do equity) |
| `min_risk_pct` | 0.0025 | piso quando há edge (evita "pó") |
| `confidence_floor` | 0.5 | abaixo disso, **não opera** |
| `default_win_loss_ratio` | 2.0 | R:R padrão (alinha com o TP 1:2 da FimatheEngine) |

- `risk_pct(confidence, r:r)` → fração do equity a arriscar (0 = não operar). **Monotônica** na confiança e
  **limitada** pelo teto.
- `qty(confidence, equity, entry, stop, …)` → quantidade, delegando a `fixed_fractional_qty` e aplicando o
  `max_qty_for_exposure` opcional.

> Nota de calibração: com ¼ Kelly sobre R:R 2:1, o risco **satura no teto** (2%) já para confiança
> moderada — comportamento **seguro** (não explode o risco). Para granularidade fina na faixa, baixe o
> `kelly_cap` ou suba o `confidence_floor`.

## 4. Integração (plano)

A confiança natural a injetar é o **score da decisão** (doc 10) — e, quando o ML for promovido (doc 11),
o **P(win)** do `SetupClassifier`. O fluxo: a estratégia define entry/stop estruturais (ex.: FimatheEngine,
doc 08) → o `DynamicSizer` define a quantidade pela confiança → o `RiskManager`/`PortfolioRiskGuard`
(circuit breakers, heat) dá a palavra final. Ordem: **edge dimensiona, guardas de portfólio limitam.**

## Como virar skill

Extrair `sizing-dinamico` com: a fórmula do Kelly fracionário, a tabela de parâmetros do `DynamicSizer`,
a regra de ouro (**sem edge → tamanho 0; mais convicção → mais risco, até um teto**) e o ponto de
integração (score/P(win) como confiança). Combinar com [02-risco-execucao](02-risco-execucao.md),
[10-camada-decisao](10-camada-decisao.md) e [11-camada-ml](11-camada-ml.md).
