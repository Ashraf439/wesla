"""Latency guards, follow-up rewriting and the trainer's optimizer-step fix."""
import numpy as np
import pytest

from wesla.extract import select_chunks
from wesla.llm import EchoLLM, LLMChain
from wesla.rewrite import needs_rewrite
from wesla.types import Chunk, SearchResult
from tests.test_agent_and_trainer import RESULTS, make_agent


def test_select_chunks_caps_total_and_per_page():
    chunks = [Chunk(text=("lora rank adapters " if i % 4 == 0 else "cooking pasta recipe ") * 20,
                    url=f"https://site{i % 3}.com/p", title="t", chunk_id=str(i)) for i in range(40)]
    out = select_chunks(chunks, "what LoRA rank do adapters use", max_total=6, max_per_page=2)
    assert len(out) == 6
    assert max(sum(c.url == u for c in out) for u in {c.url for c in out}) <= 2
    assert all("lora" in c.text for c in out)


def test_needs_rewrite():
    prev = ["what is LoRA in fine-tuning?"]
    assert needs_rewrite("what rank is typically used for it?", prev)
    assert needs_rewrite("what rank is usually used?", prev)          # short, no pronoun
    assert not needs_rewrite("what rank is typically used for it?", [])  # no history
    assert not needs_rewrite("explain how gradient checkpointing reduces GPU memory in transformers", prev)
    assert not needs_rewrite("what are the newest open-weight LLMs released this month?", prev)


def test_rewrite_falls_back_to_previous_topic():
    from wesla.rewrite import rewrite_followup

    class Parrot(EchoLLM):        # small models sometimes just repeat the earlier question
        def chat(self, messages, **kw):
            g = super().chat(messages, **kw)
            g.text = "What is QLoRA?"
            return g

    out = rewrite_followup(LLMChain([Parrot()]), "how much GPU memory does it save?", ["what is QLoRA?"])
    assert out.startswith("how much GPU memory does it save?") and "QLoRA" in out


class RewritingLLM(EchoLLM):
    """Answers the rewrite prompt with a standalone question; echoes otherwise."""
    def chat(self, messages, **kw):
        if "Rewrite the FOLLOW-UP" in messages[0]["content"]:
            g = super().chat(messages, **kw)
            g.text = "what does elastic weight consolidation use as importance?"
            return g
        return super().chat(messages, **kw)


def test_followup_is_rewritten_before_routing(tmp_path):
    from wesla.memory import ShortTermMemory

    a = make_agent(tmp_path, RESULTS)
    a.llm = LLMChain([RewritingLLM()])
    a.router.llm = a.llm
    stm = ShortTermMemory(a.embedder)
    stm.add("user", "what is elastic weight consolidation")
    stm.add("assistant", "EWC is a regularisation method ...")
    ans = a.ask("what does it use as importance?", conversation=stm)
    assert ans.debug["rewritten_query"].startswith("what does elastic weight consolidation")
    assert stm.turns[-2].text == "what does it use as importance?"   # history keeps the user's words


def test_verifier_batches_nli(tmp_path):
    calls = []

    class BatchNLI:
        def contradiction(self, a, b):
            raise AssertionError("per-pair call used")

        def contradiction_batch(self, pairs):
            calls.append(len(pairs))
            return [0.0] * len(pairs)

    a = make_agent(tmp_path, RESULTS)
    a.ask("what is elastic weight consolidation")          # fills memory
    a.verifier.nli = BatchNLI()
    a.ask("latest elastic weight consolidation Fisher information")   # web again (fresh cue)
    assert calls and len(calls) <= 2      # answer chunks first, the rest in one more batch


def test_learning_runs_after_the_answer(tmp_path):
    a = make_agent(tmp_path, RESULTS)
    a.cfg.learn.background = True
    ans = a.ask("what is elastic weight consolidation")
    assert ans.text and ans.debug["learning"].startswith("running in the background")
    a.wait_for_learning()
    assert len(a.queue.pending()) == 1           # same outcome as the synchronous path
    assert a.stats()["candidates"] == 3


def test_train_lora_steps_on_small_queue():
    torch = pytest.importorskip("torch")
    from trainer.lora_ops import train_lora

    class Tok:
        def __call__(self, texts, **kw):
            ids = torch.ones((len(texts), 4), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Out:
        def __init__(self, loss):
            self.loss = loss

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, input_ids, attention_mask, labels):
            return Out((self.lora_A ** 2).sum() * input_ids.float().mean())

    m = Toy()
    before = m.lora_A.detach().clone()
    # 8 texts, batch 4 -> 2 batches per epoch, fewer than grad_accum=4: used to make 0 steps
    stats = train_lora(m, Tok(), ["x"] * 8, ewc=None, lam=0.0, frozen_rank=0, epochs=2, lr=0.1,
                       bs=4, grad_accum=4, max_len=8, device="cpu", log=lambda *_: None)
    assert stats["steps"] == 2
    assert not torch.equal(before, m.lora_A.detach())
