"""Query Router (proposal §4.A).

Order of checks (cheapest first):
 1. Time-sensitive cue in the query            -> WEB   (parametric + LTM may be stale)
 2. Enough confident LTM hits                  -> MEMORY
 3. Parametric draft with low predictive entropy -> PARAMETRIC (reuses the draft)
 4. Otherwise                                  -> WEB
"""
from __future__ import annotations

from .activity import Stopped
import re
from dataclasses import dataclass, field

from .config import MemoryConfig, RouterConfig
from .llm import Generation, LLMChain
from .memory import LongTermMemory
from .types import MemoryItem, Route


@dataclass
class RouteDecision:
    route: Route
    reason: str
    memory_hits: list[tuple[MemoryItem, float]] = field(default_factory=list)
    draft: Generation | None = None


class QueryRouter:
    def __init__(self, cfg: RouterConfig, mem_cfg: MemoryConfig, ltm: LongTermMemory, llm: LLMChain):
        self.cfg = cfg
        self.mem_cfg = mem_cfg
        self.ltm = ltm
        self.llm = llm
        self._cue = re.compile(r"\b(" + "|".join(re.escape(c) for c in cfg.fresh_cues) + r")\b", re.I)

    def is_time_sensitive(self, query: str) -> bool:
        return bool(self._cue.search(query))

    def memory_covers(self, query: str) -> bool:
        """Cheap check: would the MEMORY route be taken (no fresh cue, enough LTM hits)?"""
        if self.is_time_sensitive(query):
            return False
        hits = self.ltm.search(query, k=self.mem_cfg.top_k, touch=False)
        good = [s for _, s in hits if s >= self.mem_cfg.hit_threshold]
        return len(good) >= self.cfg.memory_min_hits or bool(good and good[0] >= 0.92)

    def route(self, query: str, column_model: str | None = None, *, history: str = "", require_sources: bool = False) -> RouteDecision:
        hits = self.ltm.search(query, k=self.mem_cfg.top_k, touch=False)
        good = [(it, s) for it, s in hits if s >= self.mem_cfg.hit_threshold]

        if self.is_time_sensitive(query):
            return RouteDecision(Route.WEB, "time-sensitive query", good)

        if len(good) >= self.cfg.memory_min_hits or (good and good[0][1] >= 0.92):
            return RouteDecision(Route.MEMORY, f"{len(good)} LTM hit(s) >= {self.mem_cfg.hit_threshold}", good)

        if self.cfg.use_entropy and not require_sources:
            prompt = f"Recent conversation:\n{history}\n\nQuestion: {query}" if history else query
            try:
                draft = self.llm.chat(
                    [{"role": "system", "content": "Answer the question directly. For requested tables, use a Markdown pipe table with a header and separator row. If you do not know, say 'I don't know'."},
                     {"role": "user", "content": prompt}],
                    model=column_model, max_tokens=512, want_logprobs=True)
            except Stopped:
                raise
            except Exception as exc:
                import logging
                logging.getLogger("wesla.router").warning("parametric probe failed: %s", exc)
                return RouteDecision(Route.WEB, f"parametric probe failed: {exc!s:.60}", good)
            abstained = "don't know" in draft.text.lower() or "not sure" in draft.text.lower()
            if draft.mean_entropy is not None and draft.mean_entropy <= self.cfg.entropy_threshold and not abstained:
                return RouteDecision(Route.PARAMETRIC,
                                     f"entropy {draft.mean_entropy:.2f} <= {self.cfg.entropy_threshold}", good, draft)
            why = "no logprobs from provider" if draft.mean_entropy is None else f"entropy {draft.mean_entropy:.2f}"
            return RouteDecision(Route.WEB, f"uncertain parametric answer ({why})", good, draft)

        return RouteDecision(Route.WEB, "no memory coverage", good)
