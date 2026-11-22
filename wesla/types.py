"""Shared data structures that flow through the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlparse


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published: datetime | None = None
    content: str = ""  # full extracted page text, if fetched


@dataclass
class Chunk:
    text: str
    url: str
    title: str = ""
    published: datetime | None = None
    chunk_id: str = ""
    mirrors: list[str] = field(default_factory=list)  # URLs of near-duplicates merged into this chunk

    @property
    def domain(self) -> str:
        return domain_of(self.url)


@dataclass
class VerifiedChunk:
    chunk: Chunk
    authority: float          # source authority prior, 0..1
    consensus: float          # multi-source agreement, 0..1
    recency: float            # temporal freshness, 0..1
    contradiction: float      # max NLI contradiction prob vs trusted memory, 0..1
    supporting_domains: list[str] = field(default_factory=list)
    blocked: bool = False
    reason: str = ""

    @property
    def score(self) -> float:
        """Overall verification score used to rank chunks for RAG."""
        if self.blocked:
            return 0.0
        base = 0.4 * self.authority + 0.4 * self.consensus + 0.2 * self.recency
        return base * (1.0 - self.contradiction)


class Route(str, Enum):
    PARAMETRIC = "parametric"
    MEMORY = "memory"
    WEB = "web"


class Action(str, Enum):
    DISCARD = "discard"
    STORE = "store"
    LEARN = "learn"


@dataclass
class MemoryItem:
    """Long-term memory tuple M_i = (K_i, S_i, R_i, T_i, C_i) plus bookkeeping."""
    knowledge: str                 # K_i
    sources: list[str]             # S_i (URLs)
    reliability: float             # R_i
    timestamp: datetime            # T_i (when the fact was observed / published)
    access_count: int = 0          # C_i
    id: int | None = None
    domain: str = ""
    quality: float = 0.0           # Q at time of storage
    stability: float = 0.5         # temporal stability estimate
    learning_score: float = 0.0
    status: str = "ltm"            # ltm | queued | learned
    created_at: datetime = field(default_factory=utcnow)
    last_access: datetime = field(default_factory=utcnow)
    feedback: float = 0.0          # running feedback signal in [-1, 1]


@dataclass
class JudgeDecision:
    score: float
    components: dict[str, float]
    action: Action
    rationale: str = ""


@dataclass
class Answer:
    text: str
    route: Route
    contexts: list[str]
    sources: list[str]
    decisions: list[JudgeDecision] = field(default_factory=list)
    debug: dict = field(default_factory=dict)
