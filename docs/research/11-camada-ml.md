---
name: camada-ml
description: Camada 2 — ML (pacote ml/). Classificador de qualidade de setup (P(win)) treinado nos dados do loop de feedback, em champion/challenger contra o baseline determinístico, com promoção só mediante skill estatisticamente comprovado (IC99 da AUC) e melhora de expectancy. Sem sklearn (logística em numpy).
metadata:
  type: reference
---

# Camada 2 — ML (classificador de qualidade de setup)

> **O que faz o sistema mais inteligente aqui:** aprende, a partir do próprio histórico (Camada 0) e das
> features (Camada 1), a estimar **P(win) de um setup** e usa isso para filtrar trades de baixa
> probabilidade. **Mas só ganha poder quando prova que funciona** — disciplina champion/challenger.

> ✅ **Implementado** no pacote [`ml/`](../../ml), em **numpy puro** (sem sklearn — não instalado e o projeto
> é enxuto; logística é um primeiro modelo honesto e baseline para modelos mais fortes depois). Testes em
> `tests/test_ml_*.py`.

## 1. Componentes

| Módulo | Papel |
|---|---|
| `ml/dataset.py` | `build_training_set(records)` — extrai (X, y, pnl) das decisões **fechadas** (win/loss) do `DecisionLog`; ordem cronológica preservada (split temporal) |
| `ml/setup_classifier.py` | `SetupClassifier` — regressão logística (padronização + L2 + GD). Cold-start: **`predict_proba` = 0.5 enquanto não treinado** |
| `ml/champion_challenger.py` | `evaluate(ts)` — treina no passado, avalia no futuro (sem look-ahead) e decide se o modelo deve ser **promovido** |

## 2. A disciplina champion/challenger (o ponto)

O risco nº 1 em ML de trading é **overfitting**: um modelo que parece ótimo no backtest e morre ao vivo.
Por isso o ML **nunca** entra direto no caminho de decisão. Ele roda em **shadow** (observa e loga) e só é
recomendado para promoção quando, num **split temporal**, satisfaz as três condições:

1. **Skill estatisticamente significativo** — o limite inferior do **IC de 99%** da AUC (Hanley–McNeil)
   fica **acima de 0.5**. Isso barra o "lift" espúrio de amostra pequena (a 95%, ~5% do puro ruído passaria;
   a 99% a barra é conservadora — apropriado para algo que toca dinheiro).
2. **Lift mínimo** no ponto (`min_lift`, default 0.03).
3. **Melhora de expectancy** — operar só os setups aprovados (`proba ≥ threshold`) rende P&L médio maior do
   que operar tudo (take-all).

O **champion** é o baseline sem skill (classe majoritária / take-all). O **challenger** (ML) precisa
*ganhar dele com folga estatística*.

## 3. Cold-start e segurança

- Sem dados (DecisionLog vazio / poucas decisões fechadas) → `evaluate` retorna **"dados insuficientes,
  shadow"** e `SetupClassifier.predict_proba` devolve **0.5** (neutro). O sistema **nunca opina sem base**.
- O modelo é serializável (`to_dict`/`from_dict`) para persistir/recarregar sem retreinar.

## 4. Como rodar

```bash
# avalia o que ja existe no DecisionLog (gera dataset, treina, mede skill)
python -m ml.champion_challenger data/trading.sqlite
```
Saída típica enquanto não há dados: `dados insuficientes; rodando em shadow`. Conforme o bot acumula
decisões fechadas (via `FeedbackAgent`) — ou via **backtest** (replay histórico, alimenta o DecisionLog
rápido) — o relatório passa a medir skill real.

```python
from feedback.decision_log import DecisionLog
from ml.dataset import build_training_set
from ml.champion_challenger import evaluate

ts = build_training_set(DecisionLog().all_records())
rep = evaluate(ts)
print(rep.recommend_promote, rep.reason)
```

## 5. Integração (plano)

- **Agora:** shadow — `evaluate`/CLI medem o modelo offline; **não** influencia o gate ao vivo.
- **Quando `recommend_promote=True` de forma estável:** plugar o `SetupClassifier` na
  `DecisionIntelligence` (doc 10) como um sinal adicional de score/gate — começando com peso baixo
  (canary), aumentando conforme a performance ao vivo confirmar o backtest. **Detecção de drift**
  (performance ao vivo descolando) reverte para shadow.
- **Features:** hoje o `context` da decisão é enxuto (regime, score, signal_strength). Ao **enriquecer o
  contexto com as features da FimatheEngine** (doc 08), o mesmo pipeline de ML fica muito mais forte —
  basta adicionar as chaves em `ml/dataset.py`.

## 6. Próximos passos

- Modelos mais fortes (gradient boosting / floresta) comparados contra esta logística como baseline.
- **Regime via ML** em champion/challenger contra o baseline determinístico de `feedback/regime.py`.
- Calibração de probabilidade (Platt/isotônica) e validação **walk-forward** (várias janelas).

## Como virar skill

Extrair `camada-ml` com: o contrato dos 3 módulos, a **regra de promoção** (IC99 da AUC > 0.5 + lift +
expectancy), a postura cold-start (0.5 sem dados) e o plano de integração canary. Combinar com
[09-loop-de-feedback](09-loop-de-feedback.md) (dados), [08-fimathe-engine](08-fimathe-engine.md) (features)
e [10-camada-decisao](10-camada-decisao.md) (onde o score entra no gate).
