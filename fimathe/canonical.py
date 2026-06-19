"""FIMATHE CANONICA — o motor fiel a docs/FIMATHE_SPEC.md (NAO sobrescreve o antigo).

Este modulo implementa a tecnica REAL de Marcelo Ferreira (destilada na spec),
corrigindo as 10 divergencias listadas em FIMATHE_SPEC.md vs fimathe/engine.py:

  CR (Canal de Referencia) = ultimo topo e ultimo fundo da PERNA da tendencia, por
  swing CAUSAL (sem look-ahead).  ZN (Zona Neutra) = canal ADJACENTE de MESMO tamanho
  do CR, do lado CONTRARIO a tendencia (NAO o meio do canal).  So opera A FAVOR da
  tendencia do timeframe de marcacao.  Entrada A = pullback aos 50% do CR; entrada B =
  rompimento (no fechamento) da fronteira da ZN.  Stop FORA da ZN.  Alvo = projecao 2x
  (com expansoes 100%/161.8%/200% como parciais).  Sub Ciclo de Protecao = trailing:
  breakeven aos 50% do subciclo, trava aos 100%.  MULTI-TIMEFRAME: marca no TF maior
  (D1), opera no menor (H4) — a marcacao do dia D so vale a partir do FECHAMENTO de D.

DETERMINISTICO e SEM LOOK-AHEAD por construcao. Nao depende de fimathe/engine.py.

CONTRATO DE SAIDA (`process` devolve o DataFrame do TF de OPERACAO enriquecido):
  cr_top, cr_bottom, cr_range  — Canal de Referencia vigente (causal) na barra
  zn_near, zn_far              — fronteiras da Zona Neutra (near = junto ao CR)
  trend                        — +1 alta, -1 baixa, 0 sem tendencia (nao opera)
  entry_level                  — preco de entrada planejado p/ a barra (variante ativa)
  arm                          — True se o setup esta ARMADO (espera o gatilho)
  signal                       — +1/-1 quando o gatilho DISPARA no fechamento da barra
  stop_loss, take_profit       — stop (fora da ZN) e alvo (projecao 2x) do disparo
  setup_valid                  — gatilho disparado e niveis coerentes

DECISOES DE DESIGN (discricionario na tecnica -> fixado aqui, e POR QUE):
  * marcacao por SOMBRA (high/low), nao corpo: a sombra define a fronteira REAL do
    canal em preco; gera canal mais largo -> stop mais largo -> o edge precisa vencer
    MAIS custo (conservador, nao lisonjeiro). Parametrizavel via `mark_on`.
  * "perna da tendencia": definida por pivos causais consecutivos. Tendencia de ALTA =
    topos e fundos ascendentes (HH & HL) confirmados; BAIXA = LH & LL; senao trend=0.
    Regra puramente estrutural (a tecnica le geometria, ignora indicador/calendario).
  * CR = ultimo swing-high confirmado (topo) e ultimo swing-low confirmado (fundo).
  * entrada "armada e disparada": o setup so arma quando o preco entra na regiao certa
    (pullback aos 50% p/ A; reentrada apos visitar a ZN p/ B) e dispara no FECHAMENTO
    da barra que cumpre o gatilho. Sinal no close de i -> o tribunal entra no open de i+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_REQUIRED_COLS = ("open", "high", "low", "close")

# Expansoes (alvos parciais) — Fibonacci como REGUA de projecao, nao retracao.
EXPANSIONS = (1.0, 1.618, 2.0)
DEFAULT_TARGET_MULT = 2.0  # alvo primario = 2x a amplitude do CR a partir da entrada


# --------------------------------------------------------------------------- #
# Marcacao causal: pivos e CR/ZN vigentes barra-a-barra (sem look-ahead)
# --------------------------------------------------------------------------- #
def _causal_pivots(
    high: np.ndarray, low: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pivos (swing high/low) CAUSAIS.

    Um topo em i (estritamente > que os `order` vizinhos de cada lado) so fica
    DISPONIVEL em i+order (tempo de confirmacao) — exatamente como o motor antigo,
    para nao olhar o futuro. Devolve dois arrays alinhados ao indice: o valor do
    ULTIMO swing-high / swing-low confirmado ATE cada barra (NaN antes do 1o).
    """
    n = len(high)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(order, n - order):
        c = high[i]
        win = high[i - order : i + order + 1]
        # estritamente maior que todos os vizinhos (remove o centro do teste)
        if c > np.delete(win, order).max():
            j = i + order
            if j < n:
                sh[j] = c
        cl = low[i]
        winl = low[i - order : i + order + 1]
        if cl < np.delete(winl, order).min():
            j = i + order
            if j < n:
                sl[j] = cl
    # ffill manual (numpy) do ultimo pivo confirmado
    _ffill_inplace(sh)
    _ffill_inplace(sl)
    return sh, sl


def _ffill_inplace(a: np.ndarray) -> None:
    last = np.nan
    for i in range(len(a)):
        if np.isnan(a[i]):
            a[i] = last
        else:
            last = a[i]


def _prev_distinct(values: np.ndarray) -> np.ndarray:
    """Para cada barra, o valor ANTERIOR distinto da serie de pivos (p/ comparar
    HH/HL/LH/LL). Ex.: serie de swing-highs ffill-ada -> ultimo topo diferente do atual.
    Causal: so usa o passado."""
    n = len(values)
    out = np.full(n, np.nan)
    prev = np.nan
    cur = np.nan
    for i in range(n):
        v = values[i]
        if not np.isnan(v) and (np.isnan(cur) or v != cur):
            prev = cur  # o topo anterior vira "anterior distinto"
            cur = v
        out[i] = prev
    return out


@dataclass(frozen=True)
class CanonicalParams:
    """Parametros do motor canonico (todos com default fiel a spec)."""

    swing_lookback: int = 8          # ordem do pivo (confirmacao causal)
    entry_variant: str = "A"         # "A" = 50% do CR; "B" = rompimento da fronteira da ZN
    mark_on: str = "wick"            # "wick" (sombra) ou "body" (corpo) p/ marcar o CR
    stop_buffer_frac: float = 0.10   # stop FORA da ZN: buffer = frac * range do CR
    target_mult: float = DEFAULT_TARGET_MULT  # alvo = mult x range a partir da entrada
    subcycle_frac: float = 1.0       # tamanho do subciclo de protecao = frac * range
    pip_size: float = 0.0001
    require_fresh_cr: bool = True    # so opera quando o CR mudou recentemente (perna viva)

    def __post_init__(self) -> None:
        if self.entry_variant not in ("A", "B"):
            raise ValueError("entry_variant deve ser 'A' (50%) ou 'B' (rompimento da ZN)")
        if self.mark_on not in ("wick", "body"):
            raise ValueError("mark_on deve ser 'wick' (sombra) ou 'body' (corpo)")
        if self.swing_lookback < 2:
            raise ValueError("swing_lookback >= 2")


class FimatheCanonical:
    """Motor FIMATHE canonico. Marca CR/ZN/tendencia e arma/dispara entradas A/B.

    Uso single-timeframe:
        eng = FimatheCanonical(CanonicalParams(entry_variant="A"))
        out = eng.process(df_oper)               # marca e opera no mesmo TF

    Uso multi-timeframe (marca macro, opera micro):
        out = eng.process(df_oper, df_mark=df_d1) # CR/ZN/trend vem do D1 (causal)
    """

    def __init__(self, params: CanonicalParams | None = None) -> None:
        self.p = params or CanonicalParams()

    # ------------------------------------------------------------------ #
    # 1. Marcacao do CR + ZN + tendencia (no TF de MARCACAO), causal
    # ------------------------------------------------------------------ #
    def mark(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devolve o DF de marcacao com cr_top/cr_bottom/cr_range/zn_near/zn_far/trend.

        Tudo CAUSAL: na barra i so entram pivos confirmados ate i (a marcacao do
        proprio TF). Em multi-TF, este DF e o macro; o alinhamento ao micro respeita
        o fechamento da barra macro (ver `process`)."""
        self._check(df)
        high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        close = df["close"].to_numpy(float)
        open_ = df["open"].to_numpy(float)

        # Marcacao por sombra (extremos) ou por corpo (open/close).
        if self.p.mark_on == "wick":
            mh, ml = high, low
        else:
            mh = np.maximum(open_, close)
            ml = np.minimum(open_, close)

        sh, sl = _causal_pivots(mh, ml, self.p.swing_lookback)
        prev_sh = _prev_distinct(sh)
        prev_sl = _prev_distinct(sl)

        # Tendencia ESTRUTURAL: HH & HL = alta; LH & LL = baixa; senao 0.
        with np.errstate(invalid="ignore"):
            up = (sh > prev_sh) & (sl > prev_sl)
            dn = (sh < prev_sh) & (sl < prev_sl)
        trend = np.where(up, 1, np.where(dn, -1, 0)).astype(int)

        cr_top = sh
        cr_bottom = sl
        cr_range = cr_top - cr_bottom

        # ZN = canal ADJACENTE de MESMO tamanho, do lado CONTRARIO a tendencia.
        # Alta -> ZN ABAIXO do CR: near = cr_bottom, far = cr_bottom - range.
        # Baixa -> ZN ACIMA do CR: near = cr_top, far = cr_top + range.
        zn_near = np.where(trend > 0, cr_bottom, np.where(trend < 0, cr_top, np.nan))
        zn_far = np.where(
            trend > 0, cr_bottom - cr_range,
            np.where(trend < 0, cr_top + cr_range, np.nan),
        )

        # "CR fresco": a barra em que o swing-high OU swing-low acabou de mudar.
        cr_changed = np.zeros(len(df), dtype=bool)
        cr_changed[1:] = (sh[1:] != sh[:-1]) | (sl[1:] != sl[:-1])

        out = df.copy()
        out["cr_top"] = cr_top
        out["cr_bottom"] = cr_bottom
        out["cr_range"] = cr_range
        out["zn_near"] = zn_near
        out["zn_far"] = zn_far
        out["trend"] = trend
        out["cr_changed"] = cr_changed
        return out

    # ------------------------------------------------------------------ #
    # 2. Alinhamento multi-TF (macro -> micro), sem look-ahead
    # ------------------------------------------------------------------ #
    @staticmethod
    def align_marking(micro_index: pd.Index, marked_macro: pd.DataFrame) -> pd.DataFrame:
        """Projeta a marcacao macro no indice micro de forma CAUSAL.

        A marcacao da barra macro fechada em t (timestamp = inicio da barra) so pode
        ser usada DEPOIS que ela fecha. Para D1, a barra rotulada em D fecha ao fim
        de D; portanto so vale para barras micro com timestamp >= fim de D (inicio de D+1).

        Implementacao: desloca o indice macro para o INSTANTE DE FECHAMENTO e faz
        merge_asof 'backward' (cada barra micro pega a ultima marcacao macro JA fechada).
        """
        cols = ["cr_top", "cr_bottom", "cr_range", "zn_near", "zn_far", "trend", "cr_changed"]
        macro = marked_macro[cols].copy()
        # instante de fechamento da barra macro = proximo timestamp do indice macro
        idx = marked_macro.index
        close_ts = idx.to_series().shift(-1)
        # ultima barra: fecha ~1 passo a frente (usa a mediana do passo macro)
        if len(idx) >= 2:
            step = (idx[1:] - idx[:-1]).to_series().median()
        else:
            step = pd.Timedelta(days=1)
        close_ts = close_ts.fillna(idx[-1] + step) if len(idx) else close_ts
        macro = macro.assign(_close_ts=close_ts.to_numpy())
        macro = macro.sort_values("_close_ts")

        left = pd.DataFrame(index=micro_index).reset_index().rename(
            columns={micro_index.name or "index": "_mts"}
        )
        left["_mts"] = pd.to_datetime(left["_mts"], utc=True)
        macro["_close_ts"] = pd.to_datetime(macro["_close_ts"], utc=True)
        merged = pd.merge_asof(
            left.sort_values("_mts"),
            macro.sort_values("_close_ts"),
            left_on="_mts",
            right_on="_close_ts",
            direction="backward",
        )
        merged = merged.set_index("_mts")
        merged.index.name = micro_index.name
        return merged.reindex(micro_index)[cols]

    # ------------------------------------------------------------------ #
    # 3. Armar + disparar entradas (no TF de OPERACAO), causal
    # ------------------------------------------------------------------ #
    def process(
        self, df: pd.DataFrame, df_mark: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Marca (no proprio TF ou no macro) e gera sinais no TF de operacao.

        df:      OHLC do TF de OPERACAO (micro).
        df_mark: OHLC do TF de MARCACAO (macro). Se None, marca no proprio df.
        Sinal no FECHAMENTO da barra i (entrada do tribunal e no open de i+1).
        """
        self._check(df)
        out = df.copy()
        if df_mark is None:
            marked = self.mark(df)
            mk = marked[
                ["cr_top", "cr_bottom", "cr_range", "zn_near", "zn_far", "trend", "cr_changed"]
            ]
        else:
            marked_macro = self.mark(df_mark)
            mk = self.align_marking(df.index, marked_macro)

        for col in ("cr_top", "cr_bottom", "cr_range", "zn_near", "zn_far", "trend", "cr_changed"):
            out[col] = mk[col].to_numpy()

        n = len(out)
        c = out["close"].to_numpy(float)
        cr_bottom = out["cr_bottom"].to_numpy(float)
        cr_range = out["cr_range"].to_numpy(float)
        cr_top = out["cr_top"].to_numpy(float)
        zn_near = out["zn_near"].to_numpy(float)
        zn_far = out["zn_far"].to_numpy(float)
        trend = out["trend"].to_numpy(float)
        cr_changed = out["cr_changed"].to_numpy(bool)

        entry_level = np.full(n, np.nan)
        signal = np.zeros(n, dtype=int)
        stop_loss = np.full(n, np.nan)
        take_profit = np.full(n, np.nan)
        armed = np.zeros(n, dtype=bool)

        buf = self.p.stop_buffer_frac
        tgt_mult = self.p.target_mult
        variant = self.p.entry_variant
        require_fresh = self.p.require_fresh_cr

        # Estado da maquina de armar (por par/serie): armamos quando a perna esta viva;
        # disparamos quando cumpre o gatilho no fechamento. 1 setup ativo por vez.
        state = 0                 # 0 desarmado; +1 armado p/ compra; -1 armado p/ venda
        armed_trend = 0
        cooldown_until = -1       # apos disparar, espera novo "CR fresco" p/ rearmar
        visited_zn = False        # variante B: o preco ja recuou ATE a ZN? (pre-condicao)

        for i in range(n):
            tr = int(trend[i]) if not np.isnan(trend[i]) else 0
            rng = cr_range[i]
            valid_geo = (
                tr != 0 and np.isfinite(rng) and rng > 0
                and np.isfinite(cr_bottom[i]) and np.isfinite(cr_top[i])
            )
            if not valid_geo:
                state = 0
                visited_zn = False
                continue

            mid = cr_bottom[i] + 0.5 * rng  # 50% do CR (mesmo p/ alta e baixa)
            entry_level[i] = mid if variant == "A" else (
                cr_bottom[i] if tr > 0 else cr_top[i]  # B: fronteira da ZN (= borda do CR)
            )

            # Rearmar so quando a perna esta viva (CR mudou) — evita reusar canal velho.
            fresh = cr_changed[i] or not require_fresh
            if i > cooldown_until and fresh and (state == 0 or armed_trend != tr):
                state = tr
                armed_trend = tr
                visited_zn = False  # novo setup: ainda nao visitou a ZN

            if state == 0:
                continue
            armed[i] = True

            price = c[i]
            fired = False
            if variant == "A":
                # Pullback aos 50% do CR, A FAVOR da tendencia.
                # Alta: preco recuou ATE/ABAIXO dos 50% -> dispara compra no fechamento.
                # Baixa: preco subiu ATE/ACIMA dos 50% -> dispara venda no fechamento.
                # Stop FORA da ZN ainda nao foi atingido (price > zn_far na alta).
                if tr > 0 and price <= mid and price > zn_far[i]:
                    fired = True
                elif tr < 0 and price >= mid and price < zn_far[i]:
                    fired = True
            else:  # variante B: rompimento da fronteira da ZN a favor da tendencia
                # Spec: "se ja passou dos 50%, espera o fechamento ALEM do limite da ZN".
                # Faithful: (1) o preco PRECISA primeiro recuar ATE a ZN (entrar na zona
                # de nao-operacao = fechar alem da fronteira NEAR, no sentido contrario a
                # tendencia); (2) so entao um fechamento de VOLTA alem da fronteira da ZN
                # (no sentido da tendencia) dispara. Sem (1), B colapsaria em "comprar
                # sempre dentro do CR" — nao seria rompimento de nada.
                if tr > 0:
                    if price <= zn_near[i]:          # recuou para dentro da ZN
                        visited_zn = True
                    if visited_zn and price >= zn_near[i] and price > zn_far[i]:
                        fired = True
                else:
                    if price >= zn_near[i]:
                        visited_zn = True
                    if visited_zn and price <= zn_near[i] and price < zn_far[i]:
                        fired = True

            if not fired:
                continue

            side = 1 if tr > 0 else -1
            if side > 0:
                stop = zn_far[i] - buf * rng        # FORA da ZN (abaixo da borda distante)
                entry_ref = mid if variant == "A" else zn_near[i]
                target = entry_ref + tgt_mult * rng
                ok = stop < entry_ref < target
            else:
                stop = zn_far[i] + buf * rng        # FORA da ZN (acima da borda distante)
                entry_ref = mid if variant == "A" else zn_near[i]
                target = entry_ref - tgt_mult * rng
                ok = target < entry_ref < stop
            if not ok:
                state = 0
                continue

            signal[i] = side
            stop_loss[i] = stop
            take_profit[i] = target
            state = 0  # desarma; rearmara no proximo CR fresco
            cooldown_until = i

        out["entry_level"] = entry_level
        out["arm"] = armed
        out["signal"] = signal
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["setup_valid"] = (
            (out["signal"] != 0)
            & np.isfinite(out["stop_loss"])
            & np.isfinite(out["take_profit"])
        )
        # alvos de expansao (parciais) — informativos. Medidos a partir do entry_ref
        # do disparo (50% do CR na variante A; fronteira da ZN na B).
        if variant == "A":
            entry_ref_arr = np.where(signal != 0, cr_bottom + 0.5 * cr_range, np.nan)
        else:
            entry_ref_arr = np.where(
                signal > 0, zn_near, np.where(signal < 0, cr_top, np.nan)
            )
        for e in EXPANSIONS:
            out[f"exp_{int(e*1000)}"] = np.where(
                signal > 0, entry_ref_arr + e * cr_range,
                np.where(signal < 0, entry_ref_arr - e * cr_range, np.nan),
            )
        return out

    # ------------------------------------------------------------------ #
    # 4. Sub Ciclo de Protecao (trailing) — aplicado pelo tribunal por trade
    # ------------------------------------------------------------------ #
    def subcycle_stops(
        self, entry: float, side: int, cr_range: float, initial_stop: float
    ) -> list[tuple[float, float]]:
        """Niveis do Sub Ciclo de Protecao a partir da entrada.

        Devolve uma lista de (gatilho_de_preco, novo_stop) em ordem: ao TOCAR o
        gatilho, o stop sobe (alta) / desce (baixa) para `novo_stop`.
          - +50% do subciclo  -> breakeven (stop = entry)
          - +100% do subciclo -> trava (stop = entry + 0.5*subciclo no sentido do trade)
          - +150%, +200% ...   -> sobe meio subciclo por vez, surfando ate a reversao.
        subciclo = subcycle_frac * cr_range.
        """
        sub = self.p.subcycle_frac * cr_range
        if not (np.isfinite(sub) and sub > 0):
            return []
        levels: list[tuple[float, float]] = []
        # passos de meio subciclo: 0.5 (BE), 1.0 (trava), 1.5, 2.0...
        steps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for k in steps:
            trigger = entry + side * k * sub
            # stop sobe para "k-1 meio-subciclo" atras do gatilho (>= breakeven)
            lock = entry + side * max(0.0, (k - 1.0)) * sub
            if side > 0:
                lock = max(lock, entry if k >= 0.5 else initial_stop)
            else:
                lock = min(lock, entry if k >= 0.5 else initial_stop)
            levels.append((float(trigger), float(lock)))
        return levels

    # ------------------------------------------------------------------ #
    def _check(self, df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame sem colunas obrigatorias: {missing}")
