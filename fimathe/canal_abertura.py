"""CANAL DE ABERTURA — o motor MECANICO da FIMATHE REAL (intraday, day trade).

Implementa o nucleo determinístico de docs/FIMATHE_SPEC.md, a tecnica de
Marcelo Ferreira como ele REALMENTE opera (NAO o swing D1->H4 que foi mal-
testado antes; ver as divergencias #2/#4/#9/#10 da spec). Resumo fiel:

  CANAL DE ABERTURA (CA): as 4 PRIMEIRAS VELAS do M15 de cada sessao formam
  uma "caixinha" [min(low), max(high)] dessas 4 velas. Termo "quatro velas"
  aparece em 33 transcricoes. NAO se opera durante a formacao do CA.

  ENTRADA = ROMPIMENTO: "a regra da Fimathe: entrada NO ROMPIMENTO" [sKnrcI9YN8s].
  Quando uma vela FECHA acima da caixa -> compra; abaixo -> venda, no sentido do
  rompimento. (Sinal no fechamento da vela i; o tribunal entra no OPEN de i+1 —
  sem look-ahead.)

  STOP = FORA da ZN: a Zona Neutra e o canal adjacente de MESMO tamanho do lado
  contrario; o stop fica uma largura-de-canal de margem alem do lado oposto.
  Compra: stop = base_do_CA - range. Venda: stop = topo_do_CA + range.
  ("stop fora da caixinha", 20 transcricoes).

  ALVO = PROJECAO DE CICLO (clones do canal = expansoes de Fibonacci): clona a
  altura do CA a partir da ENTRADA. take=1 nivel (~100%): entry +/- 1*range.
  take=2 niveis (~200%): entry +/- 2*range. A escolha 1 vs 2 e de GESTAO, nao
  sinal [J9x9J9Vue8k].

  INTRADAY: zera no fim da sessao (sem overnight) -> SEM SWAP, so SPREAD. A
  liquidacao de fim-de-dia e responsabilidade do tribunal (precisa do OHLC), mas
  este motor fornece `session_id` e `is_session_last` para implementa-la.

DETERMINISTICO e SEM LOOK-AHEAD por construcao. Independente de fimathe/engine.py
e fimathe/canonical.py (ambos modelam o CR ERRADO — pivos de "perna da tendencia").

CONTRATO DE SAIDA (`process` devolve o DF de M15 enriquecido):
  session_id        — id ordinal da sessao (dia de pregao) de cada barra
  bar_in_session    — posicao da barra dentro da sessao (0,1,2,...); 0..3 = CA
  ca_top, ca_bottom — caixa do Canal de Abertura vigente na sessao (NaN ate fixar)
  ca_range          — ca_top - ca_bottom
  ca_fixed          — True a partir da 5a barra (CA fechado, pode operar)
  is_session_last   — True na ultima barra da sessao (gatilho de liquidacao intraday)
  signal            — +1/-1 no FECHAMENTO da barra que rompe o CA (1 entrada/sessao)
  stop_loss         — stop fora da ZN do disparo
  take_profit_1     — alvo de 1 nivel (~100% = entry +/- 1*range)  [a partir do entry de ref]
  take_profit_2     — alvo de 2 niveis (~200% = entry +/- 2*range)
  setup_valid       — sinal disparado e niveis coerentes

DECISOES DE DESIGN (discricionario na tecnica -> fixado aqui, e POR QUE):
  * CA marcado por SOMBRA (high/low das 4 velas), nao corpo: a sombra define a
    fronteira REAL em preco -> caixa mais larga -> stop mais largo -> o edge
    precisa vencer MAIS custo (conservador, nao lisonjeiro). Param: `mark_on`.
  * ROMPIMENTO confirmado por FECHAMENTO de vela M15 (nao intrabar). Marcelo
    antecipa no M1 (1 vela M15 = 15 velas M1); sem dados M1 a versao por
    fechamento M15 e a aproximacao HONESTA (entra 1 barra mais tarde, nunca mais
    cedo) e nao olha o futuro. Param: `breakout_on='close'`.
  * UMA entrada por sessao (a do rompimento do CA). Reentradas/viradas-de-mao/
    subciclos sao camadas discricionarias acima do nucleo — fora deste motor.
  * Sessao = dia de calendario UTC por padrao (a abertura "real" do XAU/forex e
    descentralizada; UTC-day e um corte determinístico e reproduzivel). O
    `session_break_hour` permite ancorar a sessao a uma hora especifica (ex. 22h
    UTC ~ rollover NY), sem mudar a logica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_REQUIRED_COLS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class CanalAberturaParams:
    """Parametros do motor (defaults fieis a spec)."""

    opening_bars: int = 4            # "as 4 primeiras velas do M15" (spec, 33 transcricoes)
    mark_on: str = "wick"            # "wick" (sombra=high/low) ou "body" (open/close)
    breakout_on: str = "close"       # "close" (fechamento alem da caixa) — honesto, sem look-ahead
    stop_zn_widths: float = 1.0      # margem do stop = N larguras-de-canal alem do lado oposto
    session_break_hour: int | None = None  # None=dia UTC; int=hora UTC que inicia a sessao

    def __post_init__(self) -> None:
        if self.opening_bars < 1:
            raise ValueError("opening_bars >= 1")
        if self.mark_on not in ("wick", "body"):
            raise ValueError("mark_on deve ser 'wick' ou 'body'")
        if self.breakout_on not in ("close",):
            raise ValueError("breakout_on suportado: 'close' (sem look-ahead)")
        if self.session_break_hour is not None and not (0 <= self.session_break_hour < 24):
            raise ValueError("session_break_hour em [0,24)")


def _session_ids(index: pd.DatetimeIndex, break_hour: int | None) -> np.ndarray:
    """Id ordinal da sessao para cada barra (sessoes contíguas, 0,1,2,...).

    break_hour=None -> sessao = dia de calendario UTC (corte a meia-noite UTC).
    break_hour=h    -> sessao vira quando a hora UTC cruza h (a barra das h:00
                       inicia uma nova sessao). Determinístico, sem look-ahead.
    """
    idx = pd.DatetimeIndex(index)
    if break_hour is None:
        keys = idx.normalize()  # meia-noite UTC do dia da barra
    else:
        shifted = idx - pd.Timedelta(hours=break_hour)
        keys = shifted.normalize()  # "dia de pregao" comeca em break_hour UTC
    # ids ordinais 0..K-1 preservando a ordem temporal
    codes = pd.factorize(keys, sort=True)[0]
    return codes.astype(int)


class CanalAbertura:
    """Motor do Canal de Abertura. Marca a caixa das 4 velas e dispara o
    rompimento, com alvos por clone (1/2 niveis) e stop fora da ZN.

    Uso:
        eng = CanalAbertura()
        out = eng.process(df_m15)   # df_m15: OHLC M15 com indice datetime UTC
        # 'signal' != 0 nas barras de rompimento; tribunal entra no open de i+1
        # e zera onde 'is_session_last' (saida intraday).
    """

    def __init__(self, params: CanalAberturaParams | None = None) -> None:
        self.p = params or CanalAberturaParams()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check(df)
        out = df.copy()
        idx = pd.DatetimeIndex(out.index)
        n = len(out)

        sid = _session_ids(idx, self.p.session_break_hour)
        # posicao da barra na sessao e ultima-barra-da-sessao (gatilho de liquidacao)
        bar_in_session = np.zeros(n, dtype=int)
        is_last = np.zeros(n, dtype=bool)
        if n:
            start = 0
            for i in range(1, n + 1):
                if i == n or sid[i] != sid[start]:
                    block = np.arange(i - start)
                    bar_in_session[start:i] = block
                    is_last[i - 1] = True
                    start = i

        high = out["high"].to_numpy(float)
        low = out["low"].to_numpy(float)
        close = out["close"].to_numpy(float)
        open_ = out["open"].to_numpy(float)

        if self.p.mark_on == "wick":
            mh, ml = high, low
        else:
            mh = np.maximum(open_, close)
            ml = np.minimum(open_, close)

        ob = self.p.opening_bars
        ca_top = np.full(n, np.nan)
        ca_bottom = np.full(n, np.nan)
        ca_fixed = np.zeros(n, dtype=bool)

        signal = np.zeros(n, dtype=int)
        stop_loss = np.full(n, np.nan)
        tp1 = np.full(n, np.nan)
        tp2 = np.full(n, np.nan)

        sw = self.p.stop_zn_widths

        # itera por sessao
        i = 0
        while i < n:
            j = i
            while j < n and sid[j] == sid[i]:
                j += 1
            length = j - i
            if length > ob:
                # caixa do CA = extremos das `ob` primeiras velas da sessao
                top = float(np.max(mh[i : i + ob]))
                bot = float(np.min(ml[i : i + ob]))
                rng = top - bot
                # preenche o CA vigente da sessao a partir da barra ob (CA fechado)
                ca_top[i + ob : j] = top
                ca_bottom[i + ob : j] = bot
                ca_fixed[i + ob : j] = True
                if np.isfinite(rng) and rng > 0:
                    # 1 entrada/sessao: o PRIMEIRO fechamento que rompe a caixa,
                    # entre a barra ob e a PENULTIMA (precisa de i+1 p/ entrar, e
                    # i+1 ainda na MESMA sessao p/ ser day trade — nao na ultima).
                    for k in range(i + ob, j - 1):
                        cpx = close[k]
                        if cpx > top:        # rompimento de alta -> compra
                            side = 1
                        elif cpx < bot:      # rompimento de baixa -> venda
                            side = -1
                        else:
                            continue
                        # entry de REFERENCIA p/ os alvos: a borda rompida do CA
                        # (a entrada real do tribunal e o open de k+1; os alvos por
                        # clone sao medidos da fronteira, fiel a "clona a altura do
                        # CA a partir da entrada"). Stop fora da ZN do lado oposto.
                        if side > 0:
                            entry_ref = top
                            stop = bot - sw * rng          # fora da ZN (abaixo)
                            t1 = entry_ref + 1.0 * rng     # ~100%
                            t2 = entry_ref + 2.0 * rng     # ~200%
                            ok = stop < entry_ref < t1 < t2
                        else:
                            entry_ref = bot
                            stop = top + sw * rng          # fora da ZN (acima)
                            t1 = entry_ref - 1.0 * rng
                            t2 = entry_ref - 2.0 * rng
                            ok = t2 < t1 < entry_ref < stop
                        if not ok:
                            break
                        signal[k] = side
                        stop_loss[k] = stop
                        tp1[k] = t1
                        tp2[k] = t2
                        break  # 1 entrada por sessao
            i = j

        out["session_id"] = sid
        out["bar_in_session"] = bar_in_session
        out["is_session_last"] = is_last
        out["ca_top"] = ca_top
        out["ca_bottom"] = ca_bottom
        out["ca_range"] = ca_top - ca_bottom
        out["ca_fixed"] = ca_fixed
        out["signal"] = signal
        out["stop_loss"] = stop_loss
        out["take_profit_1"] = tp1
        out["take_profit_2"] = tp2
        out["setup_valid"] = (
            (out["signal"] != 0)
            & np.isfinite(out["stop_loss"])
            & np.isfinite(out["take_profit_1"])
        )
        return out

    def _check(self, df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame sem colunas obrigatorias: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("indice precisa ser DatetimeIndex (UTC) — sessoes intraday")
