"""Regression tests: junk filtering, contradiction evidence, stability scoring, rewrite labels."""
from wesla.boilerplate import boilerplate_reason
from wesla.judge import temporal_stability
from wesla.llm import Generation
from wesla.rewrite import rewrite_followup

NAV = ("Skip to content Bloomberg the Company & Its Products Bloomberg Terminal Demo Request Bloomberg "
       "Anywhere Remote Login Bloomberg Customer Support Sign In Subscribe Latest News Markets Economics "
       "Technology Politics Pursuits Green Equality Businessweek Newsletters Follow us")
COOKIES = ("Your privacy, your choice. We use essential cookies to make sure the site can function. We also "
           "use optional cookies for advertising, personalisation of content, usage analysis, and social "
           "media. By accepting optional cookies, you consent to the processing of your personal data. "
           "Accept all cookies. Manage preferences. Privacy policy.")
REFS = ("In: Proceedings of the 34th International Conference on Machine Learning, pp. 3987-3995 (2017) "
        "Google Scholar Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., Tuytelaars, T.: Memory aware "
        "synapses. In: Proceedings of ECCV, pp. 139-154 (2018) Google Scholar Kirkpatrick, J. et al.: "
        "Overcoming catastrophic forgetting. PNAS 114(13), 3521-3526 (2017) Crossref Google Scholar")
SPRINGER = ("Zhang Q, Chen M, Bukharin A, He P, Cheng Y, Chen W, Zhao T. Adaptive budget allocation for "
            "parameter-efficient fine-tuning. 2023. Hu S, Chen Y, Chan C M, Chen W, Yi J, Zhao W, Wang X, "
            "Liu Z, Zheng H T. Parameter-efficient fine-tuning methods. Nature Machine Intelligence. 2023.")
HEADER = ("Computer Science > Machine Learning [Submitted on 19 Mar 2026 (v1), last revised 26 Mar 2026 (v2)] "
          "Title: Elastic weight consolidation revisited Subjects: Machine Learning (cs.LG) Cite as: "
          "arXiv:2603.18596 [cs.LG]")
PROSE = ("LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large language models. "
         "It freezes the pre-trained weights and injects trainable low-rank matrices into each layer, which "
         "reduces the number of trainable parameters by orders of magnitude. It was introduced in 2021 by "
         "researchers at Microsoft and is widely used to adapt models to specific tasks.")
CITED_PROSE = ("In the class-incremental scenario it resolves the decision boundary conflict between old and new "
               "classes in EWC (Kirkpatrick et al., 2017). On Split-MNIST it reaches 74.34% accuracy, far above "
               "standard EWC at 20.01% (Zenke et al., 2017). It also performs well in domain-incremental and "
               "task-incremental settings, surpassing several replay-based methods (Rebuffi et al., 2017).")


def test_boilerplate_is_detected():
    for name, text in [("nav", NAV), ("cookies", COOKIES), ("refs", REFS), ("springer", SPRINGER),
                       ("header", HEADER)]:
        assert boilerplate_reason(text), name


def test_real_content_is_kept():
    assert boilerplate_reason(PROSE) is None
    assert boilerplate_reason(CITED_PROSE) is None      # prose with inline citations is still knowledge
    assert boilerplate_reason("LoRA rank 8") is None    # short snippets are left to verification


def test_results_to_chunks_drops_boilerplate():
    from wesla.extract import results_to_chunks
    from wesla.types import SearchResult

    res = [SearchResult("LoRA", "https://en.wikipedia.org/wiki/LoRA", "", content=PROSE),
           SearchResult("Bloomberg", "https://www.bloomberg.com/", "", content=NAV)]
    chunks = results_to_chunks(res, fetch=False, min_tokens=20, max_tokens=512, overlap_tokens=0)
    assert [c.url for c in chunks] == ["https://en.wikipedia.org/wiki/LoRA"]


def test_boilerplate_memory_is_not_contradiction_evidence(tmp_path):
    import numpy as np

    from wesla.embeddings import HashingEmbedder
    from wesla.memory import LongTermMemory
    from wesla.types import Chunk, MemoryItem, utcnow
    from wesla.verify import SourceReputation, Verifier

    class AlwaysContradicts:
        calls = 0

        def contradiction(self, a, b):
            AlwaysContradicts.calls += 1
            return 0.99

    emb = HashingEmbedder()
    ltm = LongTermMemory(str(tmp_path / "ltm.sqlite"), emb)
    ltm.add(MemoryItem(knowledge=REFS, sources=["https://link.springer.com/x"], reliability=0.9,
                       timestamp=utcnow(), domain="t"))
    v = Verifier(emb, ltm, AlwaysContradicts(), SourceReputation(), contradiction_min_sim=0.0)
    out, _ = v.verify([Chunk(text=REFS.replace("2017", "2016"), url="https://arxiv.org/abs/1", title="")])
    assert not out[0].blocked
    assert AlwaysContradicts.calls == 0


def test_stability_separates_volatile_from_durable_facts():
    now = 2026
    volatile = ["Elon Musk is the richest man in 2026 with $480 billion.",
                "The RTX 5090 currently costs $7,350 at Newegg.",
                "The United States accounted for 403 billionaires, followed by China with 89."]
    durable = ["LoRA was introduced in 2021 by Microsoft researchers.",
               "Elastic weight consolidation is a method introduced in 2017 that uses Fisher information.",
               PROSE]
    for t in volatile:
        assert temporal_stability(t, now) < 0.5, t
    for t in durable:
        assert temporal_stability(t, now) >= 0.5, t


def test_low_rank_is_not_volatile():
    text = "LoRA learns a low-rank update. The rank r of the decomposition is usually small, such as 8."
    assert temporal_stability(text, 2026) >= 0.5


def test_rewrite_strips_echoed_prompt_labels():
    class Echo:
        def chat(self, messages, **kw):
            return Generation("FOLLOW-UP: what rank is typically used for LoRA?", "stub", "stub")

    out = rewrite_followup(Echo(), "what rank is typically used for it?", ["What is LoRA?"])
    assert out == "what rank is typically used for LoRA?"
