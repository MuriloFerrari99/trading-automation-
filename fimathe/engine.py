"""FimatheEngine — nucleo analitico da tecnica FIMATHE (doc 08).

Recebe um DataFrame OHLC e adiciona as colunas FIMATHE: canais de referencia,
zona neutra, niveis de Fibonacci dinamicos, PCM (forca de rompimento),
indicadores (RSI/ATR/ADX), sinais e gestao de risco. Tudo vetorizado em
pandas/numpy (sem numba — otimizacao futura, ver doc 08 §7).

CONVENCOES (forex):
- precos em pip via `pip_size` (0.0001 majors; 0.01 para JPY).
- O CANAL e definido sobre as `swing_period` velas ANTERIORES (shift(1)) para
  que um rompimento da vela atual seja detectavel (close pode ultrapassar).
- Parametros calibraveis por par/timeframe (doc 08 §7 "Adaptabilidade").

NAO automatiza a parte discricionaria da FIMATHE (linhas do Equador, leitura de
barra elefante, virada de mao) — ver doc 07. Aqui mora o subconjunto
deterministico, proprio para backtest e para gerar features de ML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Niveis de Fibonacci (doc 08 §3.3).
FIB_RETRACE = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_EXTEND = (1.272, 1.618, 2.618)

_REQUIRED_COLS = ("open", "high", "low", "close")

# Colunas usadas como features de ML (doc 08 §6).
ML_FEATURES = (
    "dist_to_zn",
    "pcm_score",
    "channel_width",
    "price_position_code",
    "fib_618",
    "breakout_strength",
    "adx",
    "rsi",
)


class FimatheEngine:
    def __init__(
        self,
        swing_period: int = 20,
        zone_neutra_factor: float = 0.5,
        pcm_confirmation: bool = True,
        min_atr_multiplier: float = 0.5,
        risk_reward_min: float = 2.0,
        *,
        swing_method: str = "rolling",
        pip_size: float = 0.0001,
        account_balance: float = 10_000.0,
        pip_value: float = 10.0,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
    ) -> None:
        if swing_method not in ("rolling", "pivots"):
            raise ValueError("swing_method deve ser 'rolling' ou 'pivots'")
        self.swing_method = swing_method
        self.swing_period = swing_period
        self.zone_neutra_factor = zone_neutra_factor
        self.pcm_confirmation = pcm_confirmation
        self.min_atr_multiplier = min_atr_multiplier
        self.risk_reward_min = risk_reward_min
        self.pip_size = pip_size
        self.account_balance = account_balance
        self.pip_value = pip_value
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.adx_period = adx_period

    # ------------------------------------------------------------------ #
    # Pipeline completo
    # ------------------------------------------------------------------ #
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica toda a engine e devolve o DataFrame enriquecido."""
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame sem colunas obrigatorias: {missing}")
        df = df.copy()
        df = self.detect_channels(df)
        df = self.calculate_fibonacci_levels(df)
        df = self._indicators(df)
        df = self.calculate_pcm_score(df)
        df = self.generate_signals(df)
        df = self.calculate_stops(df)
        return df

    # ------------------------------------------------------------------ #
    # 1. Canais e zona neutra (doc 08 §3.2)
    # ------------------------------------------------------------------ #
    def detect_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        sp = self.swing_period
        high, low, close = df["high"], df["low"], df["close"]

        if self.swing_method == "pivots":
            # Swings por PIVOS (local extrema), versao CAUSAL: um pivo so fica
            # disponivel `swing_period` velas depois (tempo de confirmacao),
            # evitando look-ahead. Equivalente a scipy.argrelextrema, sem a dep.
            df = self.detect_swings_pivots(df)
        else:
            # Canal das `swing_period` velas ANTERIORES (shift 1) -> rompimento
            # da vela atual e detectavel.
            df["swing_high"] = high.rolling(sp, min_periods=sp).max().shift(1)
            df["swing_low"] = low.rolling(sp, min_periods=sp).min().shift(1)

        df["upper_channel"] = df["swing_high"]
        df["lower_channel"] = df["swing_low"]

        rng = df["upper_channel"] - df["lower_channel"]
        df["zone_neutra"] = df["lower_channel"] + self.zone_neutra_factor * rng
        df["channel_width"] = rng / self.pip_size  # em pips

        df["dist_to_upper"] = (df["upper_channel"] - close) / self.pip_size
        df["dist_to_lower"] = (close - df["lower_channel"]) / self.pip_size
        df["dist_to_zn"] = (close - df["zone_neutra"]) / self.pip_size

        above = close > df["upper_channel"]
        below = close < df["lower_channel"]
        df["price_position"] = np.where(above, "acima", np.where(below, "abaixo", "dentro"))
        df["price_position_code"] = np.where(above, 1, np.where(below, -1, 0))
        return df

    @staticmethod
    def _local_extrema(values: np.ndarray, order: int, greater: bool) -> np.ndarray:
        """Indices de extremos locais (pivos), estritos vs `order` vizinhos de
        cada lado. Equivalente a scipy.signal.argrelextrema, sem a dependencia."""
        n = len(values)
        idx = []
        for i in range(order, n - order):
            center = values[i]
            window = values[i - order : i + order + 1]
            neighbors = np.delete(window, order)  # remove o proprio centro
            if greater:
                if center > neighbors.max():
                    idx.append(i)
            elif center < neighbors.min():
                idx.append(i)
        return np.array(idx, dtype=int)

    def detect_swings_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Swings por pivos, CAUSAL: o pivo so vira referencia `swing_period`
        velas apos ocorrer (tempo de confirmacao) -> sem look-ahead."""
        order = self.swing_period
        high, low = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(df)
        sh = pd.Series(np.nan, index=df.index)
        sl = pd.Series(np.nan, index=df.index)
        for i in self._local_extrema(high, order, greater=True):
            j = i + order  # disponivel so apos a confirmacao
            if j < n:
                sh.iat[j] = high[i]
        for i in self._local_extrema(low, order, greater=False):
            j = i + order
            if j < n:
                sl.iat[j] = low[i]
        df["swing_high"] = sh.ffill()
        df["swing_low"] = sl.ffill()
        return df

    # ------------------------------------------------------------------ #
    # 2. Fibonacci dinamico (doc 08 §3.3)
    # ------------------------------------------------------------------ #
    def calculate_fibonacci_levels(
        self, df: pd.DataFrame, last_swing: bool = True
    ) -> pd.DataFrame:
        if "upper_channel" not in df.columns:
            df = self.detect_channels(df)
        upper, lower = df["upper_channel"], df["lower_channel"]
        rng = upper - lower
        # Retracoes medidas do topo do swing para baixo.
        for r in FIB_RETRACE:
            df[f"fib_{int(round(r * 1000))}"] = upper - r * rng
        # Extensoes acima do topo do swing.
        for e in FIB_EXTEND:
            df[f"fib_{int(round(e * 1000))}"] = upper + (e - 1.0) * rng
        return df

    # ------------------------------------------------------------------ #
    # Indicadores auxiliares (RSI, ATR, ADX)
    # ------------------------------------------------------------------ #
    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]

        # RSI (Wilder via EWMA).
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi"] = (100 - 100 / (1 + rs)).fillna(100.0)

        # ATR (Wilder).
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / self.atr_period, adjust=False).mean()
        df["atr"] = atr

        # ADX (Wilder).
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr_safe = atr.replace(0.0, np.nan)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
            alpha=1 / self.adx_period, adjust=False
        ).mean() / atr_safe
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
            alpha=1 / self.adx_period, adjust=False
        ).mean() / atr_safe
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        df["adx"] = dx.ewm(alpha=1 / self.adx_period, adjust=False).mean().fillna(0.0)

        # Regime de tendencia (codigo) e fase de ciclo (heuristica, doc 08 §4).
        trend_up = (plus_di > minus_di) & (df["adx"] > 20)
        trend_down = (minus_di > plus_di) & (df["adx"] > 20)
        df["trend_regime"] = np.where(trend_up, 1, np.where(trend_down, -1, 0))
        df["adx_trend"] = (df["adx"] > 25).astype(int)
        # cycle_phase: 1=markup, -1=markdown, 2=distribuicao, -2=acumulacao, 0=neutro
        pos = df["price_position_code"]
        df["cycle_phase"] = np.select(
            [
                (df["trend_regime"] == 1),
                (df["trend_regime"] == -1),
                (df["trend_regime"] == 0) & (pos >= 0),
                (df["trend_regime"] == 0) & (pos < 0),
            ],
            [1, -1, 2, -2],
            default=0,
        )
        return df

    # ------------------------------------------------------------------ #
    # 3. PCM Score (doc 08 §3.4) — forca/continuidade do rompimento
    # ------------------------------------------------------------------ #
    def calculate_pcm_score(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        candle_range = (h - lo).replace(0.0, np.nan)
        body_ratio = (body / candle_range).clip(0.0, 1.0).fillna(0.0)
        df["body_size"] = body / self.pip_size  # em pips

        atr = df["atr"].replace(0.0, np.nan)
        above = (c - df["upper_channel"]).clip(lower=0.0)
        below = (df["lower_channel"] - c).clip(lower=0.0)
        breakout = above.where(above > 0, below)  # distancia alem do canal
        df["breakout_strength"] = (breakout / atr).fillna(0.0)

        # Fator de volume (se disponivel): rompimento com volume vale mais.
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(self.swing_period, min_periods=1).mean()
            vol_factor = (df["volume"] / vol_ma.replace(0.0, np.nan)).clip(0.0, 2.0).fillna(1.0)
        else:
            vol_factor = pd.Series(1.0, index=df.index)

        raw = 0.5 * body_ratio + 0.5 * np.tanh(df["breakout_strength"])
        df["pcm_score"] = (raw * (0.5 + 0.25 * vol_factor)).clip(0.0, 1.0)
        return df

    # ------------------------------------------------------------------ #
    # 4. Sinais (doc 08 §3.5)
    # ------------------------------------------------------------------ #
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        # Confirmacao PCM: corpo forte (continuidade). Desligavel por config.
        body_strong = df["pcm_score"] >= 0.5
        pcm_ok = body_strong if self.pcm_confirmation else pd.Series(True, index=df.index)

        buy = (c > df["upper_channel"]) & pcm_ok & (c > df["zone_neutra"])
        sell = (c < df["lower_channel"]) & pcm_ok & (c < df["zone_neutra"])

        df["signal"] = np.where(buy, 1, np.where(sell, -1, 0))
        df["signal_strength"] = np.where(df["signal"] != 0, df["pcm_score"], 0.0)
        df["setup_valid"] = (
            (df["signal"] != 0)
            & (df["channel_width"] > 0)
            & (df["breakout_strength"] >= self.min_atr_multiplier)
        )
        return df

    # ------------------------------------------------------------------ #
    # 5. Gestao de risco (doc 08 §3.6) — stop "fora da caixinha"
    # ------------------------------------------------------------------ #
    def calculate_stops(self, df: pd.DataFrame, risk_percent: float = 1.0) -> pd.DataFrame:
        c = df["close"]
        atr = df["atr"].fillna(0.0)
        buf = self.min_atr_multiplier * atr  # stop alem do canal (doc 07)

        is_buy = df["signal"] == 1
        is_sell = df["signal"] == -1

        stop = np.where(
            is_buy, df["lower_channel"] - buf,
            np.where(is_sell, df["upper_channel"] + buf, np.nan),
        )
        stop = pd.Series(stop, index=df.index)
        risk = (c - stop).where(is_buy, (stop - c).where(is_sell, np.nan))

        df["stop_loss"] = stop
        df["take_profit_1"] = np.where(
            is_buy, c + risk, np.where(is_sell, c - risk, np.nan)
        )  # R:R 1:1
        df["take_profit_2"] = np.where(
            is_buy, c + 2 * risk, np.where(is_sell, c - 2 * risk, np.nan)
        )  # R:R 1:2

        # Position size (lotes): risco em $ / (risco em pips * valor do pip).
        risk_pips = (risk.abs() / self.pip_size).replace(0.0, np.nan)
        risk_cash = self.account_balance * (risk_percent / 100.0)
        size = risk_cash / (risk_pips * self.pip_value)
        df["position_size"] = size.where(df["signal"] != 0, 0.0).fillna(0.0)

        # setup so e valido se o R:R (tp2 vs risco) bate o minimo configurado.
        # tp2 = 2x risco por construcao -> R:R = 2.0; refinar para alvos em
        # Fibonacci (R:R variavel) e melhoria futura (doc 08 §8).
        rr = (df["take_profit_2"] - c).abs() / risk.abs()
        df["setup_valid"] = df["setup_valid"] & (rr >= self.risk_reward_min)
        return df

    # ------------------------------------------------------------------ #
    # 6. Integracao com ML (doc 08 §6)
    # ------------------------------------------------------------------ #
    def get_features_for_ml(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Subconjunto de colunas para treinamento. Roda `process` se preciso."""
        if not all(col in df.columns for col in ML_FEATURES):
            df = self.process(df)
        out = df[list(ML_FEATURES)]
        return out.dropna() if dropna else out
