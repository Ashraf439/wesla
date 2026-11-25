"""The Learning Judge (proposal §4.F) - the core novelty.

    L_s = w_r*R + w_q*Q + w_n*N + w_u*U + w_s*S      (sum w = 1)

R  source reliability      (verification authority, adjusted by feedback)
Q  quality / consensus     (multi-source agreement x (1 - contradiction))
N  novelty                 (1 - similarity to knowledge the model already learned)
U  recurrence / utility    (how often the fact is retrieved or re-observed + feedback)
S  temporal stability      (how likely the fact is to remain true)

Decision:  L_s < tau1 -> DISCARD ;  tau1 <= L_s < tau2 -> STORE (LTM, RAG only)
           L_s >= tau2 and S, Q pass hard gates -> LEARN (queued for offline training)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .config import JudgeConfig
from .types import Action, JudgeDecision

# words that mark facts as ephemeral (likely to change soon)
_VOLATILE = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|this (week|month|morning|evening)|right now|currently|"
    r"breaking|live|price[sd]?|stock|shares? (rose|fell)|trading|score[sd]?|weather|forecast|"
    r"rumou?r(ed)?|reportedly|leak(ed)?|expected to|could|might|may|plans? to|upcoming|"
    r"deal|sale|discount|minutes ago|hours ago|just now|trending|"
    # rankings, wealth and market figures change constantly
    # ("rank" alone is excluded: "low-rank" is everywhere in ML text; so are "cost" and "leads to")
    r"richest|billionaires?|net worth|wealthiest|ranked|rankings?|leaderboard|top \d+|"
    r"as of|latest|newest|valuation|market cap|revenue|pric(ing|e tag)|"
    r"release date|launch(es|ed|ing)?|this year|so far|polls?)\b"
    # money amounts: "$480 billion", "$7,350"
    r"|\$\s?\d[\d,.]*\s*(?:billion|million|trillion|[bmk]\b)?", re.I)
# words that mark durable, definitional or historical facts. Years count only when they are
# clearly in the past (see _years): "introduced in 2021" is stable, "richest man in 2026" is not.
_STABLE = re.compile(
    r"\b(is defined as|refers to|is a|are a|was founded|was born|invented|introduced|"
    r"consists of|known as|algorithm|theorem|architecture|protocol|standard|specification|"
    r"principle|method|equation)\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _years(text: str, now_year: int) -> tuple[int, int]:
    """(past years, current-or-future years) mentioned in the text."""
    past = recent = 0
    for y in map(int, _YEAR.findall(text)):
        if y >= now_year - 1:
            recent += 1
        else:
            past += 1
    return past, recent


def temporal_stability(text: str, now_year: int | None = None) -> float:
    if now_year is None:
        from datetime import datetime, timezone

        now_year = datetime.now(timezone.utc).year
    words = max(1, len(text.split()))
    past, recent = _years(text, now_year)
    vol = len(_VOLATILE.findall(text)) + recent
    stab = len(_STABLE.findall(text)) + min(past, 3)
    density = (vol - 0.7 * stab) / math.sqrt(words)
    return float(min(1.0, max(0.0, 0.6 - 1.2 * density)))


@dataclass
class Candidate:
    """Everything the judge needs about one candidate fact."""
    text: str
    reliability: float          # R
    consensus: float            # from verification
    contradiction: float        # from verification
    novelty: float              # N (computed by caller against learned knowledge)
    recurrences: int = 0        # times retrieved / re-observed (C_i)
    feedback: float = 0.0       # running user feedback in [-1, 1]
    stability: float | None = None  # S; computed from text when None


class LearningJudge:
    def __init__(self, cfg: JudgeConfig, stability_fn=temporal_stability):
        self.cfg = cfg
        self.stability_fn = stability_fn

    def utility(self, recurrences: int, feedback: float) -> float:
        sat = max(1, self.cfg.utility_saturation)
        base = min(1.0, math.log1p(recurrences) / math.log1p(sat))
        fb = 0.5 + 0.5 * max(-1.0, min(1.0, feedback))  # map [-1,1] -> [0,1]
        return float(0.85 * base + 0.15 * fb) if feedback >= 0 else float((0.85 * base + 0.15 * fb) * fb)

    def components(self, c: Candidate) -> dict[str, float]:
        s = c.stability if c.stability is not None else self.stability_fn(c.text)
        return {
            "R": float(min(1, max(0, c.reliability))),
            "Q": float(min(1, max(0, c.consensus * (1 - c.contradiction)))),
            "N": float(min(1, max(0, c.novelty))),
            "U": self.utility(c.recurrences, c.feedback),
            "S": float(s),
        }

    def score(self, comp: dict[str, float]) -> float:
        w = self.cfg
        return (w.w_r * comp["R"] + w.w_q * comp["Q"] + w.w_n * comp["N"] +
                w.w_u * comp["U"] + w.w_s * comp["S"])

    def judge(self, c: Candidate) -> JudgeDecision:
        comp = self.components(c)
        ls = self.score(comp)
        if ls < self.cfg.tau1:
            return JudgeDecision(ls, comp, Action.DISCARD, f"L_s={ls:.2f} < tau1={self.cfg.tau1}")
        if ls >= self.cfg.tau2:
            if comp["S"] < self.cfg.learn_min_stability:
                return JudgeDecision(ls, comp, Action.STORE,
                                     f"L_s={ls:.2f} >= tau2 but S={comp['S']:.2f} too volatile to train")
            if c.feedback < 0:
                return JudgeDecision(ls, comp, Action.STORE,
                                     f"L_s={ls:.2f} >= tau2 but net-negative user feedback")
            if c.recurrences < self.cfg.learn_min_recurrence:
                return JudgeDecision(ls, comp, Action.STORE,
                                     f"L_s={ls:.2f} >= tau2 but seen {c.recurrences}x; stored until it recurs")
            if comp["Q"] < self.cfg.learn_min_quality:
                return JudgeDecision(ls, comp, Action.STORE,
                                     f"L_s={ls:.2f} >= tau2 but Q={comp['Q']:.2f} lacks consensus")
            return JudgeDecision(ls, comp, Action.LEARN, f"L_s={ls:.2f} >= tau2={self.cfg.tau2}")
        return JudgeDecision(ls, comp, Action.STORE, f"tau1 <= L_s={ls:.2f} < tau2")
