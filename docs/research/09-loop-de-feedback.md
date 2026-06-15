---
name: loop-de-feedback
description: Camada 0 — o loop de feedback que torna o sistema "cada vez mais inteligente". Registro estruturado de decisão+contexto+resultado (pacote feedback/), classificador de regime e avaliação de performance por estratégia e por regime, com guia de integração nos agentes.
metadata:
  type: reference
---

# Camada 0 — Loop de Feedback

> **Por que isto vem primeiro.** Num sistema de trading, "ficar mais inteligente" não é colocar um modelo
> maior — é **aprender com o próprio histórico**. Este é o motor disso: registra **toda decisão** (inclusive
> a de NÃO operar), o **contexto** (regime, preço, sinais) e o **resultado** (P&L realizado). Em cima desse
> par decisão→resultado — um **dataset proprietário** que ninguém mais tem — o sistema mede onde ganha,
> onde sangra, e recalibra. Hierarquia de impacto: **Validação > Dados/Features > Modelo.**

> ✅ **Implementado** no pacote [`feedback/`](../../feedback) (puro Python, sem numpy/pandas), com testes
> (`tests/test_feedback_*.py`). Desacoplado de `data/db.py` de propósito: cria a própria tabela `decisions`
> no mesmo SQLite, para evoluir e ser testado isoladamente.

---

## 1. Componentes

| Módulo | Papel |
|---|---|
| `feedback/models.py` | `Decision`, `Outcome`, e os enums `DecisionAction` (buy/sell/**hold**/**skip**), `OutcomeStatus`, `MarketRegime` |
| `feedback/regime.py` | `classify_regime(closes)` — classificador **determinístico e explicável** (trend_up/down, range, high_vol) |
| `feedback/decision_log.py` | `DecisionLog` — grava decisões e anexa resultados (tabela `decisions`, modo WAL) |
| `feedback/evaluation.py` | `evaluate()` + `format_report()` — métricas **por estratégia e por regime** |

## 2. O ciclo (como o sistema aprende)

```
  Planejador decide ──► DecisionLog.record(Decision + contexto + regime)
                              │  (grava ANTES de saber o resultado)
  Executor envia ordem ───────┤  client_order_id liga decisão ↔ ordem
                              │
  Monitor/recon, ao fechar ──► DecisionLog.attach_outcome(...)  (P&L realizado)
                              │
  Periodicamente ────────────► evaluate() ──► relatório por estratégia/regime
                              │
  Humano/sistema ────────────► recalibra parâmetros (swing_period, trail %, thresholds)
```

A chave é **registrar a decisão no momento em que ela acontece** (com o contexto que a motivou) e **anexar
o resultado depois**. Decisões de **não operar** também entram — a coisa mais inteligente que o bot faz, na
maioria dos dias, é não operar sem edge; sem registrar os `skip`, você nunca sabe se ele está sendo
seletivo ou só perdendo oportunidade.

## 3. Métricas (ajustadas ao risco, não retorno bruto)

Por grupo (total, por estratégia, por regime): nº de decisões, trades fechados, **skips**, trades abertos,
**win rate**, **profit factor**, **expectancy** (P&L médio/trade), **P&L total**, **max drawdown** e um
**Sharpe por trade** (proxy de consistência, não anualizado). Exemplo de saída:

```
TOTAL            dec=4  trades=3  skip=1  win%= 66.7 PF=2.06  exp=  28.33 pnl=    85.00 maxDD=  -80.00 shpe=-0.10
Por estrategia:
  trailing_stop  dec=2  trades=2  skip=0  win%= 50.0 PF=1.50  exp=  20.00 pnl=    40.00 maxDD=  -80.00 shpe=-0.25
Por regime:
  high_vol       dec=1  trades=1  skip=0  win%=  0.0 PF=0.00  exp= -80.00 pnl=   -80.00 maxDD=  -80.00 shpe= 0.00
  trend_up       dec=1  trades=1  skip=0  win%=100.0 PF=inf   exp= 120.00 pnl=   120.00 maxDD=    0.00 shpe= 0.00
```

Leitura: este bot **ganha em tendência e apanha em alta volatilidade** → ação óbvia: reduzir/desligar a
estratégia no regime `high_vol`. É exatamente esse tipo de insight que a quebra por regime entrega.

## 4. Uso

```python
from feedback import (
    DecisionLog, Decision, DecisionAction, MarketRegime,
    Outcome, OutcomeStatus, classify_regime, evaluate, format_report,
)

log = DecisionLog()  # usa data/trading.sqlite por padrao

# no momento da decisao:
regime = classify_regime(recent_closes)            # janela de fechamentos M15/M1
did = log.record(Decision(
    strategy="trailing_stop", symbol="AAPL", action=DecisionAction.BUY,
    regime=regime, reference_price=Decimal("190.5"), signal_strength=0.7,
    context={"sma_fast": 189.9, "rsi": 55, "signals": ["congress"]},
    client_order_id=intent.client_order_id,        # liga a ordem
))

# quando o trade fecha (Monitor/reconciliacao):
log.attach_outcome_by_client_order_id(coid, Outcome(
    status=OutcomeStatus.WIN, entry_price=Decimal("190.5"),
    exit_price=Decimal("192.4"), realized_pnl=Decimal("190"), return_pct=0.01,
))

# relatorio (CLI):  python -m feedback.evaluation data/trading.sqlite
print(format_report(evaluate(log.all_records())))
```

## 5. Guia de integração nos agentes (quando o WIP estabilizar)

> Este pacote é **autossuficiente e já testado**. A fiação nos agentes é aditiva (poucas linhas) e foi
> deixada para quando a sessão paralela terminar as mudanças em `data/db.py`/`core/models.py`, para evitar
> colisão. Pontos de fiação:

1. **`agents/planner.py`** — ao decidir (inclusive `SKIP`), chamar `DecisionLog.record(...)`, calculando o
   regime com `classify_regime()` sobre os fechamentos recentes e passando o `client_order_id` da intenção.
2. **`agents/executor.py`** — opcional: no resultado da ordem, garantir que o `client_order_id` gravado na
   decisão é o mesmo da ordem (já é, se vier de `OrderIntent`).
3. **`agents/monitor.py`** — ao detectar fechamento de posição (stop/alvo/saída), chamar
   `attach_outcome_by_client_order_id(...)` com o P&L realizado. Usar `DecisionLog.open_decisions()` para
   saber o que ainda falta conciliar.
4. **`main.py`** / scheduler — um job periódico (ex.: fim de semana) que imprime/loga
   `generate_report()` — o "raio-x" recorrente do bot.

## 6. Próximos passos (de onde a inteligência cresce)

- **Camada 1 — features:** o `context` de cada decisão já é JSON livre; a `FimatheEngine` (doc 08) pode
  popular ali features (`dist_to_zn`, `pcm_score`, ...) → vira o dataset de treino.
- **Camada 2 — modelos:** classificador de qualidade de setup e detecção de regime via ML, **comparados
  contra o baseline determinístico** de `regime.py` (champion/challenger).
- **Otimização guiada por dados:** usar `evaluate()` para recalibrar parâmetros (não chutar) e detectar
  **drift** (performance ao vivo descolando do backtest).

## Como virar skill

Extrair uma skill `loop-de-feedback` com: o contrato de `DecisionLog` (record/attach_outcome), o dicionário
de métricas (§3), o guia de fiação nos agentes (§5) e a regra de ouro — **registrar toda decisão, inclusive
não operar, sempre com o regime**. Combinar com [06-backtesting-testes](06-backtesting-testes.md) (métricas)
e [05-arquitetura-agentes](05-arquitetura-agentes.md) (onde fiar).
