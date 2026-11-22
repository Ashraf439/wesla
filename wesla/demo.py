"""Fully offline walkthrough of the pipeline with canned web results.

Shows every Learning Judge outcome: a durable two-source fact (Store, then Learn on
recurrence), an ephemeral price (Store, never Learn), a low-authority rumour
(Discard), a syndicated duplicate (dedup), and a claim that contradicts trusted
memory (blocked by verification).
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone

from .agent import WESLA
from .config import Config
from .embeddings import HashingEmbedder
from .llm import EchoLLM, LLMChain
from .search import SearchChain, StaticSearch
from .types import SearchResult
from .verify import HeuristicNLI

NOW = datetime.now(timezone.utc)

RAG_FACT = ("Retrieval-augmented generation is a method that combines a retriever over an external "
            "document index with a generator model. It was introduced in 2020 by Lewis et al. and lets "
            "a language model ground its answers in retrieved passages without retraining.")
LORA_FACT = ("LoRA, or low-rank adaptation, is a parameter-efficient fine-tuning method introduced in 2021. "
             "It freezes the pretrained weights and trains small low-rank matrices added to attention "
             "projections, which reduces trainable parameters by orders of magnitude.")

CORPUS = {
    "retrieval-augmented": [
        SearchResult("RAG - Wikipedia", "https://en.wikipedia.org/wiki/Retrieval-augmented_generation", RAG_FACT, NOW - timedelta(days=90)),
        SearchResult("RAG paper", "https://arxiv.org/abs/2005.11401", RAG_FACT.replace("lets a language model", "allows a language model"), NOW - timedelta(days=900)),
        SearchResult("RAG explained (repost)", "https://someblog.medium.com/rag", RAG_FACT, NOW - timedelta(days=10)),
    ],
    "lora": [
        SearchResult("LoRA - Wikipedia", "https://en.wikipedia.org/wiki/LoRA", LORA_FACT, NOW - timedelta(days=200)),
        SearchResult("LoRA paper", "https://arxiv.org/abs/2106.09685", LORA_FACT.replace("reduces", "cuts"), NOW - timedelta(days=1200)),
        SearchResult("Hot take", "https://reddit.com/r/ml/abc",
                     "LoRA is not a parameter-efficient method and it was introduced in 2019, reportedly.", NOW),
    ],
    "gpu price": [
        SearchResult("GPU prices fall", "https://www.theverge.com/gpu-prices",
                     "Today the RTX 5090 price dropped to 1899 dollars at major retailers, currently the lowest this month.", NOW),
        SearchResult("GPU price cut", "https://arstechnica.com/gpu-price-cut",
                     "The RTX 5090 price dropped to 1899 dollars today at major retailers, the lowest price this month.", NOW),
        SearchResult("Rumour mill", "https://x.com/leaker/1",
                     "Reportedly a new GPU could launch next week, rumoured leak says price may be lower.", NOW),
    ],
}


def run_demo() -> None:
    d = tempfile.mkdtemp(prefix="wesla-demo-")
    cfg = Config(data_dir=d)
    cfg.memory.db_path = f"{d}/ltm.sqlite"
    cfg.queue.path = f"{d}/train_queue.jsonl"
    cfg.search.fetch_pages = False
    cfg.memory.hit_threshold = 0.35  # hashing embedder gives lower similarities than bge
    cfg.verify.contradiction_min_sim = 0.25
    agent = WESLA(cfg, embedder=HashingEmbedder(), llm=LLMChain([EchoLLM(entropy=None)]),
                  search=SearchChain([StaticSearch(CORPUS)], backoff_s=0), nli=HeuristicNLI())

    script = [
        "Explain retrieval-augmented generation",
        "What is LoRA low-rank adaptation fine-tuning?",
        "What is the current GPU price today?",
        "retrieval-augmented generation retriever generator Lewis 2020",
        "What is LoRA low-rank adaptation fine-tuning?",
        "latest news about LoRA",
    ]
    for q in script:
        ans = agent.ask(q)
        print(f"\nQ: {q}\n   route={ans.route.value} ({ans.debug['route_reason']})")
        for v in ans.debug.get("verification", []):
            flag = "BLOCKED" if v["blocked"] else ""
            print(f"   verify {v['url'][:48]:48s} auth={v['auth']:.2f} cons={v['cons']:.2f} "
                  f"contra={v['contra']:.2f} {flag}")
        for dec in ans.decisions:
            comp = " ".join(f"{k}={v:.2f}" for k, v in dec.components.items())
            print(f"   judge  {dec.action.value:7s} L_s={dec.score:.2f}  {comp}  | {dec.rationale}")
    s = agent.stats()
    print("\nSummary:", {k: s[k] for k in s if k not in ("memory",)})
    print("Memory:", s["memory"])
    print(f"Training queue ({len(agent.queue.pending())} item(s)): {cfg.queue.path}")
