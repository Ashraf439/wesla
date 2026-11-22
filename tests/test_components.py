"""Offline unit tests (no models, no network)."""
from datetime import timedelta

import numpy as np
import pytest

from wesla.config import Config, JudgeConfig
from wesla.dedup import Deduplicator
from wesla.embeddings import HashingEmbedder
from wesla.extract import chunk_text
from wesla.judge import Candidate, LearningJudge, temporal_stability
from wesla.memory import LongTermMemory, ShortTermMemory
from wesla.types import Action, Chunk, MemoryItem, utcnow
from wesla.verify import HeuristicNLI, SourceReputation, Verifier

E = HashingEmbedder(256)


def test_config_weights_must_sum_to_one():
    c = Config()
    c.validate()
    c.judge.w_r = 0.9
    with pytest.raises(ValueError):
        c.validate()


def test_embedder_similarity_orders_sensibly():
    v = E.embed(["neural network training", "training a neural network", "banana bread recipe"])
    assert v[0] @ v[1] > v[0] @ v[2]
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)


def test_chunking_respects_bounds():
    text = " ".join(f"Sentence number {i} talks about transformers and memory." for i in range(300))
    chunks = chunk_text(text, 256, 512, 32)
    assert len(chunks) > 3
    for c in chunks[:-1]:
        assert len(c.split()) / 0.75 <= 512 * 1.05


def test_dedup_drops_near_duplicates_and_keeps_mirror():
    base = "The quick brown fox jumps over the lazy dog near the river bank every single morning at dawn."
    d = Deduplicator(threshold=0.7, num_perm=128, shingle_size=3)
    chunks = [Chunk(base, "https://a.com/1", chunk_id="a"),
              Chunk(base + " Indeed.", "https://b.com/2", chunk_id="b"),
              Chunk("Completely different text about GPUs and VRAM budgets for LoRA training.", "https://c.com", chunk_id="c")]
    kept, dropped = d.filter(chunks)
    assert dropped == 1 and len(kept) == 2
    assert kept[0].mirrors == ["https://b.com/2"]


def test_stm_eviction_keeps_relevant_turns():
    stm = ShortTermMemory(E, max_turns=3, max_tokens=10_000)
    for t in ["we discussed LoRA adapters", "the weather is nice", "lunch was pasta", "more on LoRA rank", "hello"]:
        stm.add("user", t)
    kept = [t.text for t in stm.context("LoRA adapters rank")]
    assert len(kept) == 3
    assert "more on LoRA rank" in kept


def test_ltm_roundtrip_search_and_prune(tmp_path):
    ltm = LongTermMemory(str(tmp_path / "m.sqlite"), E, decay_half_life_days=1)
    i1 = ltm.add(MemoryItem("LoRA trains low rank matrices", ["https://arxiv.org"], 0.8, utcnow()))
    old = MemoryItem("stale gossip about phones", ["https://x.com"], 0.1, utcnow())
    old.last_access = utcnow() - timedelta(days=30)
    ltm.add(old)
    hits = ltm.search("low rank matrices LoRA", k=1)
    assert hits[0][0].id == i1 and hits[0][0].access_count == 1
    assert ltm.prune(min_strength=0.05) == 1
    # index persists and reloads
    ltm2 = LongTermMemory(str(tmp_path / "m.sqlite"), E)
    assert len(ltm2) == 1 and ltm2.search("LoRA", k=1)[0][0].id == i1


def test_temporal_stability_separates_ephemeral_from_durable():
    assert temporal_stability("Today the stock price is trading up right now after a rumoured deal.") < 0.3
    assert temporal_stability("The transformer is a neural network architecture introduced in 2017.") > 0.6


def _judge():
    return LearningJudge(JudgeConfig())


def test_judge_three_tiers():
    j = _judge()
    noisy = Candidate("someone reportedly said the price may drop today", 0.3, 0.25, 0.0, 0.8)
    assert j.judge(noisy).action == Action.DISCARD
    first_seen = Candidate("LoRA is a parameter-efficient method introduced in 2021.", 0.85, 1.0, 0.0, 1.0)
    assert j.judge(first_seen).action == Action.STORE          # needs to recur first
    recurring = Candidate(first_seen.text, 0.85, 1.0, 0.0, 1.0, recurrences=3)
    assert j.judge(recurring).action == Action.LEARN


def test_judge_never_learns_volatile_or_unconfirmed_facts():
    j = _judge()
    volatile = Candidate("Today the GPU price is currently trading down in a flash sale right now.",
                         0.9, 1.0, 0.0, 1.0, recurrences=10)
    d = j.judge(volatile)
    assert d.action == Action.STORE and "volatile" in d.rationale
    single = Candidate("The protocol is a standard defined in 2019.", 0.95, 0.25, 0.0, 1.0, recurrences=10, stability=1.0)
    assert j.judge(single).action == Action.STORE


def test_judge_score_is_weighted_sum():
    j = _judge()
    comp = {"R": 1, "Q": 0, "N": 0, "U": 0, "S": 0}
    assert j.score(comp) == pytest.approx(JudgeConfig().w_r)


def test_negative_feedback_reduces_utility():
    j = _judge()
    assert j.utility(3, -1.0) < j.utility(3, 0.0) < j.utility(3, 1.0)


def test_verifier_consensus_and_contradiction(tmp_path):
    ltm = LongTermMemory(str(tmp_path / "m.sqlite"), E)
    ltm.add(MemoryItem("LoRA was introduced in 2021 by Hu and colleagues at Microsoft.", ["https://arxiv.org"], 0.9, utcnow()))
    v = Verifier(E, ltm, HeuristicNLI(), SourceReputation(), consensus_sim=0.6, contradiction_min_sim=0.2)
    fact = "Low rank adaptation LoRA freezes pretrained weights and trains small matrices in attention layers."
    chunks = [Chunk(fact, "https://en.wikipedia.org/wiki/LoRA"),
              Chunk(fact.replace("small", "tiny"), "https://arxiv.org/abs/2106"),
              Chunk("LoRA was introduced in 2019 by Hu and colleagues at Microsoft.", "https://reddit.com/r/x")]
    out, vecs = v.verify(chunks)
    assert out[0].consensus == 1.0 and out[0].supporting_domains == ["arxiv.org"]
    assert out[2].blocked and out[2].contradiction >= 0.8
    assert vecs.shape == (3, E.dim)


def test_source_reputation_feedback(tmp_path):
    rep = SourceReputation(str(tmp_path / "rep.json"))
    assert rep.score("cdc.gov") > rep.score("reddit.com")
    before = rep.score("example.com")
    rep.adjust("example.com", 0.1)
    assert SourceReputation(str(tmp_path / "rep.json")).score("example.com") == pytest.approx(before + 0.1)
