"""Tribunal estatistico de edge — anti-overfitting.

Tres testes que decidem se um Sharpe de backtest e SINAL ou RUIDO. Um backtest com
Sharpe alto nao prova nada: basta testar muitas configuracoes que uma "vence" por acaso.
Estes testes corrigem isso.

- PSR (Probabilistic Sharpe Ratio): probabilidade de o Sharpe VERDADEIRO exceder um
  benchmark, corrigida por tamanho de amostra, assimetria (skew) e curtose — porque
  retornos de trading nao sao normais (tem caudas gordas).
- DSR (Deflated Sharpe Ratio): o PSR avaliado contra o benchmark que voce esperaria do
  MELHOR de N tentativas por puro acaso. Corrige o data-snooping de testar N configs e
  reportar so a melhor. DSR >= 0.95  <=>  Sharpe significativo a p < 0.05.
- PBO (Probability of Backtest Overfitting) via CSCV: dada a matriz de retornos de varias
  configs, estima a probabilidade de a melhor in-sample ficar ABAIXO da mediana
  out-of-sample. PBO alto => o processo de selecao esta escolhendo sorte, nao skill.

Refs: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014), SSRN 2460551;
Lopez de Prado, "Advances in Financial Machine Learning", cap. 11-12.

Sem dependencia nova: usa statistics.NormalDist (stdlib) para CDF/PPF normais.

Convencao de unidades: Sharpe POR PERIODO (de uma observacao da serie) em todo o modulo,
EXCETO onde o nome diz "_annual". Anualize multiplicando por sqrt(periods_per_year).
Use periods_per_year=365 para cripto (24/7) e 252 para equities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np

_NORM = NormalDist()
_EULER = 0.5772156649015329  # constante de Euler-Mascheroni
CRYPTO_PERIODS = 365
EQUITY_PERIODS = 252


def _cdf(x: float) -> float:
    return _NORM.cdf(x)


def _ppf(p: float) -> float:
    # clamp para evitar inf nas pontas (N grande no DSR puxa p -> 1)
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return _NORM.inv_cdf(p)


def skew(returns: np.ndarray) -> float:
    """Assimetria populacional (Fisher). 0 = simetrico."""
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 0.0
    s = r.std(ddof=0)
    if s == 0:
        return 0.0
    return float(((r - r.mean()) ** 3).mean() / s**3)


def kurtosis(returns: np.ndarray) -> float:
    """Curtose NAO-excesso (normal = 3.0)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 4:
        return 3.0
    s = r.std(ddof=0)
    if s == 0:
        return 3.0
    return float(((r - r.mean()) ** 4).mean() / s**4)


def observed_sharpe(returns: np.ndarray, periods_per_year: int | None = None) -> float:
    """Sharpe da serie. Por periodo por padrao; anualizado se periods_per_year for dado."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = float(r.mean() / sd)
    if periods_per_year:
        sr *= float(np.sqrt(periods_per_year))
    return sr


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew_: float, kurt: float, sr_benchmark: float = 0.0
) -> float:
    """PSR: P(Sharpe_verdadeiro > sr_benchmark). `sr` e `sr_benchmark` POR PERIODO.

    kurt e NAO-excesso (normal=3). Retorna probabilidade em [0,1].
    """
    if n < 2:
        return 0.0
    denom = 1.0 - skew_ * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(_cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Sharpe (por periodo) ESPERADO do melhor de n_trials tentativas, por puro acaso.

    E[max] ≈ sqrt(Var_SR) * [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    (Bailey & Lopez de Prado 2014). sr_variance = variancia dos Sharpes entre as N trials.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = _ppf(1.0 - 1.0 / n_trials)
    z2 = _ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - _EULER) * z1 + _EULER * z2))


@dataclass
class EdgeVerdict:
    """Veredito do tribunal sobre UMA estrategia. `passes` = barra de aceitacao do projeto."""

    n_obs: int
    n_trials: int
    sharpe_annual: float
    psr: float  # P(SR verdadeiro > 0)
    dsr: float  # P(SR verdadeiro > E[max de n_trials])  -> >=0.95 e o teto do projeto
    skew: float
    kurtosis: float
    sr_benchmark_annual: float  # o "obstaculo" do data-snooping, anualizado
    passes_dsr: bool
    passes_sharpe: bool

    @property
    def passes(self) -> bool:
        return self.passes_dsr and self.passes_sharpe

    def summary(self) -> str:
        flag = "PASSA" if self.passes else "FALHA"
        return (
            f"[{flag}] Sharpe_anual={self.sharpe_annual:.2f} "
            f"(obstaculo data-snooping={self.sr_benchmark_annual:.2f}) | "
            f"PSR={self.psr:.3f} DSR={self.dsr:.3f} | "
            f"n={self.n_obs} trials={self.n_trials} "
            f"skew={self.skew:.2f} kurt={self.kurtosis:.2f}"
        )


def evaluate_edge(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sharpes: list[float] | None = None,
    periods_per_year: int = CRYPTO_PERIODS,
    min_sharpe_annual: float = 0.8,
    dsr_threshold: float = 0.95,
) -> EdgeVerdict:
    """Submete os retornos (POR PERIODO) de uma estrategia ao tribunal.

    n_trials: quantas configuracoes/hipoteses foram testadas para chegar nesta (anti-snooping).
    trial_sharpes: Sharpes POR PERIODO de todas as trials (ideal). Se ausente, a variancia
        dos Sharpes e estimada da propria serie (mais conservador na ausencia de info).
    periods_per_year: 365 cripto, 252 equities.

    Barra de aceitacao do projeto: DSR >= dsr_threshold E Sharpe anual liquido >= min_sharpe.
    `returns` ja deve estar LIQUIDO de custos (use o cenario estressado para ser honesto).
    """
    r = np.asarray(returns, dtype=float)
    n = int(r.size)
    sr_period = observed_sharpe(r)  # por periodo
    sk = skew(r)
    ku = kurtosis(r)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        # Variancia do ESTIMADOR de Sharpe (Lo, 2002), usada como proxy da dispersao
        # entre trials quando nao temos a distribuicao real. Conservador.
        sr_var = (
            (1.0 - sk * sr_period + (ku - 1.0) / 4.0 * sr_period**2) / (n - 1)
            if n > 1
            else 0.0
        )

    sr0 = expected_max_sharpe(sr_var, n_trials)  # obstaculo por periodo
    dsr = probabilistic_sharpe_ratio(sr_period, n, sk, ku, sr_benchmark=sr0)
    psr = probabilistic_sharpe_ratio(sr_period, n, sk, ku, sr_benchmark=0.0)

    ann = float(np.sqrt(periods_per_year))
    sharpe_annual = sr_period * ann
    return EdgeVerdict(
        n_obs=n,
        n_trials=n_trials,
        sharpe_annual=sharpe_annual,
        psr=psr,
        dsr=dsr,
        skew=sk,
        kurtosis=ku,
        sr_benchmark_annual=sr0 * ann,
        passes_dsr=dsr >= dsr_threshold,
        passes_sharpe=sharpe_annual >= min_sharpe_annual,
    )


def _col_sharpe(block: np.ndarray) -> np.ndarray:
    """Sharpe por coluna (config) de um bloco de retornos (linhas=tempo)."""
    mean = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    out = np.zeros_like(mean)
    nz = sd > 0
    out[nz] = mean[nz] / sd[nz]
    return out


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_splits: int = 16
) -> float:
    """PBO via CSCV (Combinatorially Symmetric Cross-Validation).

    returns_matrix: shape (T observacoes, N configs). Cada coluna e a serie de retornos
    de uma configuracao testada. Retorna PBO em [0,1]: probabilidade de a config que foi
    a MELHOR in-sample cair ABAIXO da mediana out-of-sample. PBO < 0.5 e o minimo aceitavel;
    quanto menor, mais robusto o processo de selecao.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return 0.0
    T, N = M.shape
    S = n_splits - (n_splits % 2)  # precisa ser par
    if S < 2:
        S = 2
    S = min(S, T)  # nao mais blocos que observacoes
    if S < 2:
        return 0.0

    blocks = np.array_split(np.arange(T), S)
    logits: list[float] = []
    for is_combo in combinations(range(S), S // 2):
        is_set = set(is_combo)
        is_rows = np.concatenate([blocks[b] for b in is_combo])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_set])
        if is_rows.size < 2 or oos_rows.size < 2:
            continue
        is_perf = _col_sharpe(M[is_rows])
        oos_perf = _col_sharpe(M[oos_rows])
        best = int(np.argmax(is_perf))  # melhor in-sample
        # rank relativo OOS da escolhida: 1=pior ... N=melhor
        oos_rank = int(oos_perf.argsort().argsort()[best]) + 1
        w = oos_rank / (N + 1.0)  # em (0,1); 0.5 = mediana
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))

    if not logits:
        return 0.0
    arr = np.asarray(logits)
    return float((arr <= 0.0).mean())  # fracao em que a melhor IS ficou <= mediana OOS
