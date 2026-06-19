"""Scorer de sentimento financeiro para MANCHETES (Python puro, sem deps).

Lexico no espirito Loughran-McDonald (o padrao academico p/ texto financeiro)
ESTENDIDO com os verbos de acao que dominam manchetes de mercado (Benzinga):
beat/miss/surge/plunge/upgrade/downgrade etc. — termos ausentes no LM original
(feito p/ 10-Ks) mas de altissimo sinal em headlines.

score_headline -> sentimento liquido em [-1, 1]: (pos - neg)/(pos + neg).
Trata NEGACAO ("not strong", "fails to beat") invertendo o termo seguinte.
Transparente e reproduzivel: sem modelo, sem treino, sem look-ahead.
"""

from __future__ import annotations

import re

# --- termos POSITIVOS (LM positive + verbos de manchete de alta) ---
POSITIVE = {
    # acao de preco / manchete
    "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped",
    "rally", "rallies", "rallied", "rallying", "climb", "climbs", "climbed", "gain", "gains",
    "gained", "rise", "rises", "rose", "rising", "soaring", "spike", "spikes", "spiked",
    "rebound", "rebounds", "rebounded", "recover", "recovers", "recovered", "recovery",
    "outperform", "outperforms", "outperformed", "outperforming", "topped", "tops", "top",
    "beat", "beats", "beating", "exceed", "exceeds", "exceeded", "upbeat",
    # analista / corporativo
    "upgrade", "upgrades", "upgraded", "raise", "raises", "raised", "boost", "boosts", "boosted",
    "bullish", "buy", "overweight", "outperform", "strong", "stronger", "strongest", "strength",
    "record", "records", "high", "highs", "growth", "grow", "grows", "growing", "profit",
    "profits", "profitable", "gains", "win", "wins", "winning", "won", "approval", "approved",
    "approve", "launch", "launches", "launched", "expansion", "expand", "expands", "partnership",
    "breakthrough", "milestone", "robust", "solid", "optimistic", "optimism", "positive",
    "accelerate", "accelerates", "accelerating", "momentum", "leading", "leader", "best",
    "opportunity", "opportunities", "successful", "success", "succeed", "innovative", "innovation",
    "dividend", "buyback", "buybacks", "surpass", "surpasses", "surpassed", "rebounding",
}

# --- termos NEGATIVOS (LM negative + verbos de manchete de baixa) ---
NEGATIVE = {
    # acao de preco / manchete
    "plunge", "plunges", "plunged", "plummet", "plummets", "plummeted", "tumble", "tumbles",
    "tumbled", "slump", "slumps", "slumped", "sink", "sinks", "sank", "drop", "drops", "dropped",
    "fall", "falls", "fell", "falling", "decline", "declines", "declined", "declining", "slide",
    "slides", "slid", "crash", "crashes", "crashed", "selloff", "sell-off", "dip", "dips", "dipped",
    "slip", "slips", "slipped", "retreat", "retreats", "lower", "sinking", "plunging", "tumbling",
    # analista / corporativo
    "downgrade", "downgrades", "downgraded", "cut", "cuts", "lowered", "lowers", "bearish",
    "sell", "underweight", "underperform", "underperforms", "underperformed", "weak", "weaker",
    "weakness", "loss", "losses", "lose", "loses", "losing", "lost", "miss", "misses", "missed",
    "warn", "warns", "warning", "warned", "concern", "concerns", "concerned", "fear", "fears",
    "risk", "risks", "risky", "worry", "worries", "worried", "fail", "fails", "failed", "failure",
    "lawsuit", "lawsuits", "sue", "sued", "sues", "probe", "investigation", "investigate",
    "recall", "recalls", "recalled", "fraud", "scandal", "bankruptcy", "bankrupt", "default",
    "layoff", "layoffs", "cuts", "slash", "slashes", "slashed", "halt", "halts", "halted",
    "delay", "delays", "delayed", "disappoint", "disappoints", "disappointing", "disappointed",
    "negative", "pessimistic", "downturn", "recession", "slowdown", "struggle", "struggles",
    "struggling", "trouble", "troubled", "crisis", "collapse", "collapses", "collapsed", "debt",
    "deficit", "shortfall", "guidance", "weakens", "weakening", "pressure", "pressured", "drag",
    "worst", "worse", "caution", "cautious", "headwind", "headwinds", "stalls", "stalled",
}

NEGATORS = {"no", "not", "never", "without", "fails", "fail", "failed", "lacks", "lack", "unable", "isn't", "doesn't", "won't"}

_TOKEN_RE = re.compile(r"[a-z'\-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def score_headline(text: str) -> float:
    """Sentimento liquido de UMA manchete em [-1, 1]. 0 = neutro/sem termo."""
    toks = _tokens(text)
    pos = neg = 0
    for i, t in enumerate(toks):
        is_pos = t in POSITIVE
        is_neg = t in NEGATIVE
        if not (is_pos or is_neg):
            continue
        # negacao nas 2 palavras anteriores inverte o sinal
        negated = any(toks[j] in NEGATORS for j in range(max(0, i - 2), i))
        if negated:
            is_pos, is_neg = is_neg, is_pos
        pos += int(is_pos)
        neg += int(is_neg)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def score_series(headlines) -> list[float]:
    return [score_headline(h or "") for h in headlines]
