"""End-to-end agent flow + torch-free trainer logic."""
import json

import numpy as np
import pytest

from wesla.agent import WESLA
from wesla.config import Config
from wesla.embeddings import HashingEmbedder
from wesla.llm import EchoLLM, LLMChain, _entropy_from_top_logprobs
from wesla.search import SearchChain, StaticSearch
from wesla.types import Route, SearchResult
from wesla.verify import HeuristicNLI
from trainer.clustering import assign, lateral_parents, leader_cluster, update_centroid
from trainer.registry import Column, Registry
from trainer.replay import ReplayBuffer
from trainer.train import main as train_main, read_queue

FACT = ("Elastic weight consolidation is a regularisation method introduced in 2017 that slows down "
        "learning on weights important for earlier tasks, using the Fisher information as importance.")


def make_agent(tmp_path, results, entropy=None):
    cfg = Config(data_dir=str(tmp_path))
    cfg.memory.db_path = str(tmp_path / "ltm.sqlite")
    cfg.queue.path = str(tmp_path / "train_queue.jsonl")
    cfg.search.fetch_pages = False
    cfg.memory.hit_threshold = 0.3
    return WESLA(cfg, embedder=HashingEmbedder(), llm=LLMChain([EchoLLM(entropy=entropy)]),
                 search=SearchChain([StaticSearch(results)], backoff_s=0), nli=HeuristicNLI())


RESULTS = [SearchResult("EWC", "https://en.wikipedia.org/wiki/EWC", FACT),
           SearchResult("EWC paper", "https://www.pnas.org/doi/ewc", FACT.replace("slows down", "reduces")),
           SearchResult("spam", "https://spam.example.biz/x", "Buy cheap GPUs today, price drop right now, deal!")]


def test_web_then_memory_then_queue(tmp_path):
    a = make_agent(tmp_path, RESULTS)
    ans = a.ask("what is elastic weight consolidation")
    assert ans.route == Route.WEB and ans.sources
    assert "Elastic weight consolidation" in ans.text
    acts = ans.debug["decision_counts"]
    assert acts.get("discard", 0) >= 1
    # two agreeing sources -> recurrence -> queued for learning
    assert len(a.queue.pending()) == 1
    rec = a.queue.pending()[0]
    assert rec["vec"] and len(rec["vec"]) == a.embedder.dim
    ans2 = a.ask("elastic weight consolidation Fisher information")
    assert ans2.route == Route.MEMORY
    cands = (tmp_path / "candidates.jsonl").read_text().strip().splitlines()
    assert len(cands) == 3


def test_parametric_route_when_confident(tmp_path):
    a = make_agent(tmp_path, RESULTS, entropy=0.3)
    ans = a.ask("define a neural network")
    assert ans.route == Route.PARAMETRIC
    assert a.metrics.get("web_searches", 0) == 0


def test_time_sensitive_goes_to_web(tmp_path):
    a = make_agent(tmp_path, RESULTS, entropy=0.1)
    assert a.ask("latest elastic weight consolidation news").route == Route.WEB


def test_negative_feedback_retracts_queued_item(tmp_path):
    a = make_agent(tmp_path, RESULTS)
    a.ask("what is elastic weight consolidation")
    ans = a.ask("elastic weight consolidation Fisher information")
    assert len(a.queue.pending()) == 1
    out = a.feedback(ans.debug["answer_id"], -1, "EWC was proposed by Kirkpatrick et al.")
    assert len(out["updated_memory_ids"]) == 2
    assert a.queue.pending() == []
    assert read_queue(str(tmp_path / "train_queue.jsonl"), None) == []


def test_sync_marks_learned(tmp_path):
    a = make_agent(tmp_path, RESULTS)
    a.ask("what is elastic weight consolidation")
    mid = a.queue.pending()[0]["memory_id"]
    run = tmp_path / "run"
    run.mkdir()
    (run / "trained.json").write_text(json.dumps({"accepted": [mid], "rejected": []}))
    assert a.sync_training_results(str(run))["learned"] == 1
    assert a.ltm.get(mid).status == "learned"


def test_entropy_from_logprobs():
    certain = [{"top_logprobs": [{"logprob": 0.0}, {"logprob": -20}]}]
    unsure = [{"top_logprobs": [{"logprob": np.log(0.5)}, {"logprob": np.log(0.5)}]}]
    assert _entropy_from_top_logprobs(certain) < 0.01
    assert _entropy_from_top_logprobs(unsure) == pytest.approx(np.log(2))
    assert _entropy_from_top_logprobs([]) is None


# ------------------------------------------------------------ trainer (torch-free)
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_leader_cluster_groups_by_direction():
    vecs = np.stack([_unit([1, 0.1, 0]), _unit([1, 0, 0.1]), _unit([0, 1, 0]), _unit([0.1, 1, 0])])
    groups = leader_cluster(vecs, 0.8)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3]]


def test_assign_reuse_new_defer_and_cap():
    reg = Registry(theta_domain=0.8, columns=[Column("col_000", "a", list(_unit([1, 0, 0])), n_items=5, version=1, rank=16)])
    vecs = np.stack([_unit([1, 0.1, 0])] * 3 + [_unit([0, 1, 0.05])] * 9 + [_unit([0, 0, 1])] * 2)
    a = assign(reg, vecs, min_new_items=8, max_columns=3)
    assert a.reuse == {"col_000": [0, 1, 2]}
    assert len(a.new) == 1 and len(a.new[0]) == 9
    assert sorted(a.deferred) == [12, 13]
    capped = assign(reg, vecs, min_new_items=8, max_columns=1)
    assert not capped.new and len(capped.reuse["col_000"]) == 14


def test_update_centroid_running_mean():
    c = Column("c", "c", list(_unit([1, 0])), n_items=1)
    update_centroid(c, np.stack([_unit([0, 1])]))
    assert c.n_items == 2
    assert np.allclose(c.centroid, _unit([1, 1]), atol=1e-5)


def test_lateral_parents_softmax():
    reg = Registry(columns=[Column("a", "a", list(_unit([1, 0])), version=1),
                            Column("b", "b", list(_unit([0.7, 0.7])), version=1),
                            Column("c", "c", list(_unit([0, 1])), version=0)])  # untrained: ignored
    ids, w = lateral_parents(reg, _unit([1, 0.2]), k=2)
    assert ids == ["a", "b"] and w[0] > w[1] and sum(w) == pytest.approx(1)


def test_replay_buffer_mix_and_holdout(tmp_path):
    rb = ReplayBuffer(tmp_path / "r.json", capacity=100, holdout_frac=0.2, seed=1)
    rb.add([f"old fact {i}" for i in range(200)])
    assert 0 < len(rb.holdout) <= 20 and len(rb.train) <= 80
    batch = rb.mix([f"new {i}" for i in range(7)], ratio=0.3)
    assert sum(b.startswith("old") for b in batch) == 3
    rb.save()
    assert ReplayBuffer(tmp_path / "r.json").seen == 200


def test_trainer_dry_run_plans_from_agent_queue(tmp_path):
    a = make_agent(tmp_path, RESULTS)
    a.ask("what is elastic weight consolidation")
    rep = train_main(["--queue", str(tmp_path / "train_queue.jsonl"), "--workdir", str(tmp_path / "run"),
                      "--dry-run", "--min-items", "1", "--min-new-items", "1"])
    assert rep["items"] == 1 and not rep["deferred"]
