---
name: fimathe-engine
description: Especificação da classe FimatheEngine — o núcleo analítico que implementa a técnica FIMATHE de forma programática (detecção de canais, zona neutra, PCM, Fibonacci dinâmico, sinais, stops/takes), usável em modo rule-based ou como feature generator para ML.
metadata:
  type: reference
---

# FimatheEngine — Especificação da Classe

> **Relação com a metodologia.** Este documento é o **contrato de implementação** (API spec) do motor que
> traduz a metodologia FIMATHE descrita em [07-fimathe-forex.md](07-fimathe-forex.md) em código. O doc 07
> explica os *conceitos* (canal de abertura, zona neutra, virada de mão, fatiamento); este doc define a
> *classe* que os calcula sobre um DataFrame de preços.

> ⚠️ **Escopo (ver doc 07, §nota de escopo).** A FIMATHE é forex/ouro (ex.: `EURUSD_M15.parquet`). A
> **Alpaca não negocia pares de moedas** — esta engine pressupõe um **feed forex/MT5**. Use-a primeiro para
> **backtest das regras determinísticas**, não como módulo do bot Alpaca atual.

> ✅ **Implementado** em [`fimathe/engine.py`](../../fimathe/engine.py) (vetorizado em pandas/numpy), com
> testes (`tests/test_fimathe_engine.py`). Diferenças vs. spec: layout **flat** `fimathe/` (não `src/`);
> **numba** omitido (otimização futura); parâmetros extras de calibração forex no construtor
> (`pip_size`, `account_balance`, `pip_value`, `rsi_period`, `atr_period`, `adx_period`). O **canal usa as
> velas anteriores** (`shift(1)`) para que o rompimento seja detectável.

---

## 1. Visão geral

A classe `FimatheEngine` é o **núcleo analítico** do sistema. Implementa toda a lógica da técnica FIMATHE
de forma programática, incluindo:

- Detecção de **swings** e **canais de referência**
- Cálculo da **Zona Neutra**
- Aplicação do **PCM** (Princípio da Continuidade do Movimento)
- **Níveis de Fibonacci dinâmicos**
- **Geração de sinais** de entrada
- Cálculo de **Stop Loss** e **Take Profit**

É projetada para ser **independente do modelo de Machine Learning**, permitindo uso tanto em **modo
rule-based puro** quanto como **feature generator** para ML.

---

## 2. Estrutura da classe

```python
class FimatheEngine:
    def __init__(
        self,
        swing_period: int = 20,
        zone_neutra_factor: float = 0.5,
        pcm_confirmation: bool = True,
        min_atr_multiplier: float = 0.5,
        risk_reward_min: float = 2.0,
    ):
        ...
```

### Parâmetros do construtor

| Parâmetro | Tipo | Default | Descrição |
|---|---|---|---|
| `swing_period` | `int` | `20` | Período para detecção de swings (High/Low) |
| `zone_neutra_factor` | `float` | `0.5` | Posição da zona neutra dentro do canal (`0.5` = centro) |
| `pcm_confirmation` | `bool` | `True` | Exigir confirmação pelo corpo do candle |
| `min_atr_multiplier` | `float` | `0.5` | Tamanho mínimo do rompimento, em ATR |
| `risk_reward_min` | `float` | `2.0` | R:R mínimo aceito |

---

## 3. Principais métodos

### 3.1 Processamento de dados

```python
def process(self, df: pd.DataFrame) -> pd.DataFrame
```
- Recebe `DataFrame` com colunas: `open`, `high`, `low`, `close`, `volume` (opcional).
- Retorna o mesmo `DataFrame` com **todas as colunas FIMATHE** adicionadas.

### 3.2 Detecção de canais e zona neutra

```python
def detect_channels(self, df: pd.DataFrame) -> pd.DataFrame
```
Cria as colunas:
- `swing_high`
- `swing_low`
- `upper_channel`
- `lower_channel`
- `zone_neutra`
- `channel_width` (em pips)
- `price_position` (acima / dentro / abaixo)

### 3.3 Fibonacci dinâmico

```python
def calculate_fibonacci_levels(self, df: pd.DataFrame, last_swing: bool = True) -> pd.DataFrame
```
Gera níveis:
- **Retrações:** 23,6% · 38,2% · 50% · 61,8% · 78,6%
- **Extensões:** 127,2% · 161,8% · 261,8%

### 3.4 PCM Score e confirmação

```python
def calculate_pcm_score(self, df: pd.DataFrame) -> pd.DataFrame
```
Calcula a **força do rompimento** considerando:
- Tamanho do corpo do candle
- Relação corpo/sombra
- Volume / tick count (se disponível)
- Distância do rompimento

### 3.5 Geração de sinais

```python
def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame
```
Adiciona colunas:
- `signal` → `1` (Compra), `-1` (Venda), `0` (Neutro)
- `signal_strength` (0 a 1)
- `setup_valid` (`True`/`False`)

**Regras principais de entrada:**
- **Buy:** `close > upper_channel` + PCM confirmado + preço **acima** da Zona Neutra
- **Sell:** `close < lower_channel` + PCM confirmado + preço **abaixo** da Zona Neutra

### 3.6 Gestão de risco

```python
def calculate_stops(self, df: pd.DataFrame, risk_percent: float = 1.0) -> pd.DataFrame
```
Retorna:
- `stop_loss`
- `take_profit_1` (R:R 1:1)
- `take_profit_2` (R:R 1:2 ou Fibonacci)
- `position_size` (em lotes)

---

## 4. Features geradas pela engine

| Categoria | Features principais |
|---|---|
| **Canais** | `upper_channel`, `lower_channel`, `zone_neutra`, `channel_width` |
| **Posição** | `price_position`, `dist_to_upper`, `dist_to_lower`, `dist_to_zn` |
| **Fibonacci** | `fib_382`, `fib_618`, `fib_1272`, `fib_1618` |
| **PCM** | `pcm_score`, `body_size`, `breakout_strength` |
| **Tendência** | `trend_regime`, `adx_trend`, `cycle_phase` |
| **Sinal** | `signal`, `signal_strength`, `setup_valid` |

---

## 5. Exemplo de uso

```python
import pandas as pd
from src.fimathe.engine import FimatheEngine

# Carregar dados
df = pd.read_parquet("data/EURUSD_M15.parquet")

# Inicializar engine
engine = FimatheEngine(
    swing_period=25,
    pcm_confirmation=True,
    risk_reward_min=2.2,
)

# Processar
df = engine.process(df)

# Gerar sinais
df = engine.generate_signals(df)

# Calcular stops
df = engine.calculate_stops(df, risk_percent=1.0)

print(df[["close", "signal", "stop_loss", "take_profit_2", "setup_valid"]].tail())
```

---

## 6. Integração com ML

A engine pode ser usada como **feature transformer**:

```python
def get_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas as features relevantes para treinamento."""
    feature_cols = [
        "dist_to_zn", "pcm_score", "channel_width", "price_position_code",
        "fib_618", "breakout_strength", "adx", "rsi",
    ]
    return df[feature_cols]
```

---

## 7. Considerações técnicas

- **Performance:** otimizada com **pandas vetorizado + numba** onde necessário.
- **Adaptabilidade:** parâmetros configuráveis **por timeframe e par**.
- **Não-estacionariedade:** recomenda-se **recalibrar `swing_period`** periodicamente.
- **Multi-timeframe:** suporte nativo (pode receber dados de múltiplos TFs).
- **Testabilidade:** métodos individuais devem ter **testes unitários** (ver doc [06-backtesting-testes](06-backtesting-testes.md)).

---

## 8. Próximos passos / melhorias

- ✅ **Detecção de swings por pivôs** (`swing_method="pivots"`) — implementada em numpy puro
  (equivalente a `scipy.signal.argrelextrema`, **sem a dependência**) e, crucialmente, **causal**: um pivô
  só vira referência `swing_period` velas depois (tempo de confirmação), evitando o **look-ahead bias** que
  a versão ingênua (`argrelextrema` + `ffill`) introduz. `swing_method="rolling"` segue como padrão.
  *(Nota: `pandas_ta` foi descartado — usa `from numpy import NaN`, removido no numpy 2.0; quebraria neste
  ambiente. RSI/ATR/ADX são calculados à mão, sem dependência.)*
- Canais dinâmicos com **regressão linear**.
- Suporte a **Order Blocks** e **Fair Value Gaps (FVG)**.
- **Versão com estado** (para live trading).
- Alvos (`take_profit_2`) em níveis de Fibonacci (R:R variável) em vez de 2R fixo.

---

## Como virar skill

Extrair uma skill `fimathe-engine` que documente a **API determinística** da engine (assinaturas de
`process` / `detect_channels` / `calculate_fibonacci_levels` / `calculate_pcm_score` / `generate_signals` /
`calculate_stops`), o **dicionário de features** (§4) e o **exemplo de uso** (§5) — servindo de contrato
para implementar e testar o módulo `src/fimathe/engine.py`. Combinar com [07-fimathe-forex](07-fimathe-forex.md)
(conceitos) e [02-risco-execucao](02-risco-execucao.md) (position sizing/stops).
