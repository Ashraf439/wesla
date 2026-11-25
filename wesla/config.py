"""Central configuration for WESLA.

Values load from `config.toml` (if present) and can be overridden by
environment variables of the form WESLA_<SECTION>__<KEY>, e.g.
WESLA_JUDGE__TAU2=0.75.
"""
from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    # Ordered list of providers to try: "ollama", "groq", "openrouter", "echo"
    providers: list[str] = field(default_factory=lambda: ["ollama", "groq"])
    ollama_url: str = "http://localhost:11434"
    ollama_vision_model: str = "qwen3-vl:2b-instruct"
    ollama_model: str = "llama3.2:1b"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: float = 60.0


@dataclass
class EmbeddingConfig:
    # "sentence-transformers" or "hashing" (dependency-free fallback)
    backend: str = "sentence-transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384  # used by the hashing backend


@dataclass
class SearchConfig:
    # Ordered providers: "ddgs", "tavily"
    providers: list[str] = field(default_factory=lambda: ["ddgs", "tavily"])
    max_results: int = 6
    fetch_pages: bool = True
    max_retries: int = 2
    backoff_s: float = 1.0
    fetch_budget_s: float = 7.0     # stop waiting for slow pages after this; their snippet is used
    fetch_timeout_s: float = 6.0    # per page


@dataclass
class ChunkConfig:
    min_tokens: int = 256
    max_tokens: int = 512
    overlap_tokens: int = 32
    max_chunks: int = 8             # chunks verified / judged per web query (most query-relevant)
    max_chunks_per_page: int = 3


@dataclass
class DedupConfig:
    num_perm: int = 128
    threshold: float = 0.8  # Jaccard similarity above which chunks are duplicates
    shingle_size: int = 5


@dataclass
class VerifyConfig:
    consensus_sim: float = 0.85     # bge-small: same-topic chunks from different sites sit at 0.70-0.85; 0.85+ is the top 0.2%
    min_sources: int = 2            # distinct domains needed for full consensus
    contradiction_threshold: float = 0.80  # NLI contradiction prob to block
    contradiction_min_sim: float = 0.80    # only NLI-check memory items about nearly the same claim
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    use_nli: bool = True
    nli_memory_k: int = 2           # trusted memory items NLI-checked per chunk
    recency_half_life_days: float = 365.0
    min_verified_score: float = 0.35  # below this a chunk is not used at all


@dataclass
class MemoryConfig:
    db_path: str = "data/ltm.sqlite"
    stm_turns: int = 8
    stm_max_tokens: int = 2048
    decay_half_life_days: float = 30.0
    hit_threshold: float = 0.74      # cosine sim for an LTM hit (bge-small; calibrate per embedder)
    top_k: int = 5


@dataclass
class RouterConfig:
    # if LTM hit coverage is good enough we skip the web
    memory_min_hits: int = 1
    # queries with these cues are treated as time-sensitive -> always web
    fresh_cues: list[str] = field(default_factory=lambda: [
        "today", "latest", "current", "now", "this week", "this month",
        "recent", "2025", "2026", "new", "price", "score", "release",
    ])
    # entropy threshold (nats/token) for the parametric path, when logprobs exist
    entropy_threshold: float = 0.45  # llama3.2:1b is confidently wrong above ~0.5 nats/token
    use_entropy: bool = True


@dataclass
class JudgeConfig:
    w_r: float = 0.25   # source reliability
    w_q: float = 0.25   # quality / consensus
    w_n: float = 0.15   # novelty
    w_u: float = 0.20   # recurrence / utility
    w_s: float = 0.15   # temporal stability
    tau1: float = 0.40  # below -> discard
    tau2: float = 0.70  # at/above -> learn
    # access count at which U saturates to 1.0
    utility_saturation: int = 5
    # hard gates on top of L_s: ephemeral or single-source facts are never trained
    learn_min_stability: float = 0.5
    learn_min_quality: float = 0.6
    # a fact must be re-observed / retrieved this many times before it is trained
    learn_min_recurrence: int = 1
    # cosine sim at which a new candidate is the *same* fact as an LTM item
    merge_sim: float = 0.90


@dataclass
class LearnConfig:
    # run the remaining contradiction checks + Learning Judge after the answer is returned,
    # so the user does not wait for them (the next question waits for them instead)
    background: bool = True


@dataclass
class QueueConfig:
    path: str = "data/train_queue.jsonl"
    hf_repo: str = ""   # e.g. "username/wesla-train-queue" (dataset repo); empty = local only


@dataclass
class Config:
    domain: str = "tech-news"
    data_dir: str = "data"
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)

    def validate(self) -> None:
        j = self.judge
        total = j.w_r + j.w_q + j.w_n + j.w_u + j.w_s
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Learning Judge weights must sum to 1 (got {total:.4f})")
        if not 0 <= j.tau1 < j.tau2 <= 1:
            raise ValueError("Need 0 <= tau1 < tau2 <= 1")


def _apply(obj, data: dict) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _apply(cur, val)
        else:
            setattr(obj, f.name, val)


def _coerce(raw: str, current):
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else Path(os.environ.get("WESLA_CONFIG", "config.toml"))
    if p.exists():
        with open(p, "rb") as fh:
            _apply(cfg, tomllib.load(fh))
    for key, raw in os.environ.items():
        if not key.startswith("WESLA_") or "__" not in key:
            continue
        section, _, name = key[len("WESLA_"):].lower().partition("__")
        sub = getattr(cfg, section, None)
        if sub is not None and is_dataclass(sub) and hasattr(sub, name):
            setattr(sub, name, _coerce(raw, getattr(sub, name)))
    cfg.validate()
    return cfg
