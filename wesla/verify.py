"""Information Verification Layer (proposal §4.E).

Every candidate chunk gets four signals before it can be used or stored:
  authority     - source reputation prior (domain rules + feedback-learned offsets)
  consensus     - agreement from chunks of *other* domains (embedding similarity)
  recency       - exponential freshness of the publication date
  contradiction - max NLI contradiction probability against trusted LTM items
Chunks whose contradiction exceeds the threshold are blocked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .embeddings import Embedder
from .memory import LongTermMemory
from .log import get_logger
from .types import Chunk, VerifiedChunk, domain_of, utcnow

log = get_logger("verify")

# ------------------------------------------------------------ authority
_RULES: list[tuple[str, float]] = [
    (r"\.gov(\.[a-z]{2})?$", 0.92), (r"\.edu$", 0.88), (r"\.ac\.[a-z]{2}$", 0.88),
    (r"(^|\.)wikipedia\.org$", 0.82), (r"(^|\.)arxiv\.org$", 0.80),
    (r"(^|\.)(reuters|apnews|bbc|nature|science)\.(com|org|co\.uk)$", 0.88),
    (r"(^|\.)(nytimes|theguardian|ft|bloomberg|economist|wsj)\.(com|co\.uk)$", 0.82),
    (r"(^|\.)(theverge|arstechnica|techcrunch|wired|zdnet|engadget)\.com$", 0.72),
    (r"(^|\.)(github|docs\.python|developer\.mozilla|huggingface)\.(com|org|co)$", 0.75),
    (r"(^|\.)(medium|substack|blogspot|wordpress)\.com$", 0.45),
    (r"(^|\.)(reddit|quora|twitter|x|facebook|tiktok)\.com$", 0.30),
]


class SourceReputation:
    """Domain authority = rule prior + bounded offset learned from user feedback."""

    def __init__(self, path: str | None = None, default: float = 0.55):
        self.default = default
        self.path = Path(path) if path else None
        self.offsets: dict[str, float] = {}
        if self.path and self.path.exists():
            self.offsets = json.loads(self.path.read_text())

    def prior(self, domain: str) -> float:
        for pattern, score in _RULES:
            if re.search(pattern, domain):
                return score
        return self.default

    def score(self, domain: str) -> float:
        return float(np.clip(self.prior(domain) + self.offsets.get(domain, 0.0), 0.05, 0.98))

    def adjust(self, domain: str, delta: float, cap: float = 0.3) -> None:
        self.offsets[domain] = float(np.clip(self.offsets.get(domain, 0.0) + delta, -cap, cap))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.offsets, indent=2))


# ------------------------------------------------------------ NLI
class NLIModel(Protocol):
    def contradiction(self, premise: str, hypothesis: str) -> float: ...


class CrossEncoderNLI:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")
        labels = getattr(self.model.model.config, "id2label", None) or {}
        names = {i: str(l).lower() for i, l in labels.items()}
        self.c_idx = next((i for i, l in names.items() if "contra" in l), 0)

    def contradiction(self, premise: str, hypothesis: str) -> float:
        logits = np.asarray(self.model.predict([(premise, hypothesis)]))[0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[self.c_idx])


_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_NEG = re.compile(r"\b(not|no|never|none|cannot|isn't|wasn't|aren't|doesn't|didn't|won't|false)\b", re.I)
_WORDS = re.compile(r"[a-z]+")
_STOP = set("""a an the is are was were be been being of in on at to for from by with and or but if
then than that this these those it its as into about over after before such which who whom what when
where why how can could would should will may might has have had do does did so also very more most
less least only just not no never none reportedly""".split())


class HeuristicNLI:
    """Dependency-free stand-in: flags pairs that talk about the same thing but
    disagree on numbers or polarity. Good enough for tests and offline demos."""

    def contradiction(self, premise: str, hypothesis: str) -> float:
        a = set(_WORDS.findall(premise.lower())) - _STOP
        b = set(_WORDS.findall(hypothesis.lower())) - _STOP
        if not a or not b:
            return 0.0
        overlap = len(a & b) / min(len(a), len(b))
        if overlap < 0.5:
            return 0.0
        na, nb = set(_NUM.findall(premise)), set(_NUM.findall(hypothesis))
        num_conflict = bool(na and nb and not (na & nb))
        neg_conflict = bool(_NEG.search(premise)) != bool(_NEG.search(hypothesis))
        if num_conflict or neg_conflict:
            return min(0.95, 0.6 + 0.4 * overlap)
        return 0.05


def make_nli(use_nli: bool, model_name: str) -> NLIModel:
    if use_nli:
        try:
            return CrossEncoderNLI(model_name)
        except Exception as exc:
            log.warning("NLI model unavailable (%s); using heuristic NLI", exc)
    return HeuristicNLI()


# ------------------------------------------------------------ verifier
class Verifier:
    def __init__(self, embedder: Embedder, ltm: LongTermMemory, nli: NLIModel,
                 reputation: SourceReputation, consensus_sim: float = 0.7, min_sources: int = 2,
                 contradiction_threshold: float = 0.8, recency_half_life_days: float = 365.0,
                 trusted_reliability: float = 0.6, contradiction_min_sim: float = 0.5):
        self.embedder = embedder
        self.ltm = ltm
        self.nli = nli
        self.rep = reputation
        self.consensus_sim = consensus_sim
        self.min_sources = min_sources
        self.c_thresh = contradiction_threshold
        self.half_life = recency_half_life_days
        self.trusted = trusted_reliability
        self.c_min_sim = contradiction_min_sim

    def _recency(self, c: Chunk) -> float:
        if c.published is None:
            return 0.6  # unknown date: neutral-ish
        pub = c.published if c.published.tzinfo else c.published.replace(tzinfo=utcnow().tzinfo)
        age = max(0.0, (utcnow() - pub).total_seconds() / 86400)
        return float(0.5 ** (age / self.half_life))

    def verify(self, chunks: list[Chunk]) -> tuple[list[VerifiedChunk], np.ndarray]:
        """Returns verified chunks (same order) and their embeddings."""
        if not chunks:
            return [], np.zeros((0, self.embedder.dim), dtype=np.float32)
        vecs = self.embedder.embed([c.text for c in chunks])
        sims = vecs @ vecs.T
        domains = [c.domain for c in chunks]
        out: list[VerifiedChunk] = []
        for i, c in enumerate(chunks):
            support = {domains[j] for j in range(len(chunks))
                       if j != i and domains[j] != domains[i] and sims[i, j] >= self.consensus_sim}
            # near-duplicate copies merged by dedup still count as (syndicated) sources
            support |= {domain_of(u) for u in c.mirrors if domain_of(u) != domains[i]}
            support = sorted(support)
            need = max(1, self.min_sources - 1)
            consensus = 0.25 + 0.75 * min(1.0, len(support) / need)
            # authority: own domain, nudged up by the best supporter
            auth = self.rep.score(domains[i])
            if support:
                auth = max(auth, 0.5 * auth + 0.5 * max(self.rep.score(d) for d in support))
            # contradiction vs trusted memory
            contra, reason = 0.0, ""
            for item, sim in self.ltm.search(vecs[i], k=3, touch=False, min_sim=self.c_min_sim):
                if item.reliability < self.trusted:
                    continue
                p = self.nli.contradiction(item.knowledge, c.text)
                if p > contra:
                    contra, reason = p, f"contradicts memory #{item.id}"
            blocked = contra >= self.c_thresh
            out.append(VerifiedChunk(chunk=c, authority=auth, consensus=consensus,
                                     recency=self._recency(c), contradiction=contra,
                                     supporting_domains=support, blocked=blocked,
                                     reason=reason if blocked else ""))
        return out, vecs
