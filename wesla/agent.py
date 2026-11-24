"""WESLA orchestrator: wires every component into the proposal's workflow.

Query -> Router -> [Parametric | Memory | Web] -> Extraction & Dedup -> Verification
      -> Dynamic RAG -> Grounded generation -> Learning Judge -> [Discard | Store | Learn-queue]
User feedback -> utility / reliability updates (never direct weight updates).
"""
from __future__ import annotations

import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .adapters import AdapterRegistry
from .config import Config
from .dedup import Deduplicator
from .embeddings import Embedder, make_embedder
from .extract import results_to_chunks, select_chunks
from .judge import Candidate, LearningJudge
from .llm import LLMChain, make_llm
from .log import get_logger, stage
from .memory import LongTermMemory, ShortTermMemory
from .rag import build_messages
from .rewrite import rewrite_followup
from .router import QueryRouter
from .search import SearchChain, make_search
from .train_queue import TrainQueue
from .types import Action, Answer, JudgeDecision, MemoryItem, Route, VerifiedChunk, utcnow
from .verify import NLIModel, SourceReputation, Verifier, make_nli


log = get_logger("agent")


@dataclass
class _AnswerRecord:
    memory_ids: list[int] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


class WESLA:
    def __init__(self, cfg: Config, *, embedder: Embedder | None = None, llm: LLMChain | None = None,
                 search: SearchChain | None = None, nli: NLIModel | None = None):
        self.cfg = cfg
        data = Path(cfg.data_dir)
        data.mkdir(parents=True, exist_ok=True)
        with stage(log, "load embedder", backend=cfg.embedding.backend, model=cfg.embedding.model):
            self.embedder = embedder or make_embedder(cfg.embedding.backend, cfg.embedding.model,
                                                      cfg.embedding.dim)
        log.info("embedder: %s (dim %d)", type(self.embedder).__name__, self.embedder.dim)
        self.llm = llm or make_llm(cfg.llm)
        log.info("LLM providers: %s", [p.name for p in self.llm.providers])
        self.search = search or make_search(cfg.search.providers, cfg.search.max_retries, cfg.search.backoff_s)
        log.info("search providers: %s", [p.name for p in self.search.providers])
        with stage(log, "open long-term memory", db=cfg.memory.db_path):
            self.ltm = LongTermMemory(cfg.memory.db_path, self.embedder, cfg.memory.decay_half_life_days)
        self.stm = ShortTermMemory(self.embedder, cfg.memory.stm_turns, cfg.memory.stm_max_tokens)
        self.reputation = SourceReputation(str(data / "source_reputation.json"))
        v = cfg.verify
        with stage(log, "load NLI model", model=v.nli_model if v.use_nli else "heuristic"):
            nli = nli or make_nli(v.use_nli, v.nli_model)
        self.verifier = Verifier(self.embedder, self.ltm, nli,
                                 self.reputation, v.consensus_sim, v.min_sources,
                                 v.contradiction_threshold, v.recency_half_life_days,
                                 contradiction_min_sim=v.contradiction_min_sim,
                                 memory_k=v.nli_memory_k)
        self.judge = LearningJudge(cfg.judge)
        self.router = QueryRouter(cfg.router, cfg.memory, self.ltm, self.llm)
        self.queue = TrainQueue(cfg.queue.path, cfg.queue.hf_repo)
        self.adapters = AdapterRegistry(str(data / "columns.json"), self.embedder, cfg.llm.ollama_model)
        self.metrics: Counter = Counter()
        self._answers: dict[str, _AnswerRecord] = {}
        self._learning: threading.Thread | None = None
        self._learn_lock = threading.Lock()
        from concurrent.futures import ThreadPoolExecutor
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wesla-search")
        log.info("WESLA ready: %d LTM items, %d trained column(s)", len(self.ltm), len(self.adapters.columns))

    # ------------------------------------------------------------ helpers
    def _novelty(self, vec: np.ndarray) -> float:
        """1 - similarity to knowledge already learned (or queued for learning)."""
        best = 0.0
        for item, sim in self.ltm.search(vec, k=10, touch=False):
            if item.status in ("learned", "queued"):
                best = max(best, sim)
        return float(1.0 - max(0.0, best))

    def _judge_item(self, item: MemoryItem, vec: np.ndarray | None = None) -> JudgeDecision:
        vec = vec if vec is not None else self.embedder.embed([item.knowledge])[0]
        # novelty must ignore the item itself
        best = 0.0
        for other, sim in self.ltm.search(vec, k=10, touch=False):
            if other.id != item.id and other.status in ("learned", "queued"):
                best = max(best, sim)
        cand = Candidate(text=item.knowledge, reliability=item.reliability, consensus=item.quality,
                         contradiction=0.0, novelty=1.0 - best, recurrences=item.access_count,
                         feedback=item.feedback, stability=item.stability)
        return self.judge.judge(cand)

    def _apply_decision(self, item: MemoryItem, d: JudgeDecision) -> None:
        item.learning_score = d.score
        if d.action == Action.LEARN and item.status == "ltm":
            item.status = "queued"
            self.ltm.update(item)
            self.queue.enqueue(item, d.components, self.embedder.embed([item.knowledge])[0])
            self.metrics["promoted_to_learn"] += 1
        else:
            self.ltm.update(item)

    def _log_candidate(self, vc: VerifiedChunk, vec: np.ndarray, d: JudgeDecision) -> None:
        """Every candidate is logged so the 'train on everything' baseline can be replayed."""
        import json

        rec = {"text": vc.chunk.text, "url": vc.chunk.url, "action": d.action.value,
               "score": round(d.score, 4), "components": {k: round(v, 4) for k, v in d.components.items()},
               "vec": [round(float(x), 5) for x in vec], "at": utcnow().isoformat()}
        with open(Path(self.cfg.data_dir) / "candidates.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _ingest(self, vc: VerifiedChunk, vec: np.ndarray) -> JudgeDecision | None:
        d = self._ingest_inner(vc, vec)
        if d is not None:
            self._log_candidate(vc, vec, d)
        return d

    def _ingest_inner(self, vc: VerifiedChunk, vec: np.ndarray) -> JudgeDecision | None:
        """Run one verified chunk through the Learning Judge and act on the decision."""
        self.metrics["candidates"] += 1
        if vc.blocked:
            self.metrics["blocked"] += 1
            self.metrics["discard"] += 1
            self.ltm.log("blocked", {"url": vc.chunk.url, "reason": vc.reason})
            return JudgeDecision(0.0, {}, Action.DISCARD, vc.reason)

        # recurrence: same fact already in LTM -> strengthen it instead of duplicating
        hits = self.ltm.search(vec, k=1, touch=False)
        if hits and hits[0][1] >= self.cfg.judge.merge_sim:
            item = hits[0][0]
            item.access_count += 1
            if vc.chunk.url not in item.sources:
                item.sources.append(vc.chunk.url)
            item.reliability = max(item.reliability, vc.authority)
            item.quality = max(item.quality, vc.consensus * (1 - vc.contradiction))
            item.last_access = utcnow()
            d = self._judge_item(item, vec)
            self._apply_decision(item, d)
            self.metrics["recurrence"] += 1
            return d

        cand = Candidate(text=vc.chunk.text, reliability=vc.authority, consensus=vc.consensus,
                         contradiction=vc.contradiction, novelty=self._novelty(vec))
        d = self.judge.judge(cand)
        self.metrics[d.action.value] += 1
        if d.action == Action.DISCARD:
            self.ltm.log("discard", {"url": vc.chunk.url, "score": d.score, "c": d.components})
            return d
        item = MemoryItem(knowledge=vc.chunk.text, sources=[vc.chunk.url], reliability=vc.authority,
                          timestamp=vc.chunk.published or utcnow(), domain=self.cfg.domain,
                          quality=d.components["Q"], stability=d.components["S"],
                          learning_score=d.score, status="ltm")
        self.ltm.add(item, vec)
        if d.action == Action.LEARN:
            item.status = "queued"
            self.ltm.update(item)
            self.queue.enqueue(item, d.components, vec)
        return d

    # ------------------------------------------------------------ background learning
    def wait_for_learning(self, timeout: float | None = None) -> None:
        """Block until the previous answer's learning step (NLI + Learning Judge) has finished."""
        t = self._learning
        if t is not None and t.is_alive():
            log.info("waiting for the previous answer's learning step")
            t.join(timeout)

    def _learn_from(self, verified, vecs, checked: set[int]) -> list[JudgeDecision]:
        """Contradiction-check the chunks not used in the answer, then run every candidate through
        the Learning Judge (discard / store in LTM / queue for training)."""
        with self._learn_lock:
            rest = [i for i in range(len(verified)) if i not in checked]
            if rest:
                with stage(log, "verify (rest)", chunks=len(rest)):
                    self.verifier.check_contradictions(verified, vecs, rest)
            decisions = []
            with stage(log, "learning judge", candidates=len(verified)):
                for vc, v in zip(verified, vecs):
                    d = self._ingest(vc, v)
                    if d:
                        decisions.append(d)
                        log.debug("judge %s L_s=%.3f %s %s", d.action.value, d.score,
                                  {k: round(x, 2) for k, x in d.components.items()}, vc.chunk.url)
            log.info("learning: %s", dict(Counter(d.action.value for d in decisions)))
            return decisions

    def _learn_in_background(self, verified, vecs, checked: set[int]) -> None:
        def run():
            try:
                self._learn_from(verified, vecs, checked)
            except Exception:
                log.exception("background learning failed")
        self._learning = threading.Thread(target=run, name="wesla-learn", daemon=True)
        self._learning.start()

    # ------------------------------------------------------------ main entry
    def ask(self, query: str, *, model_override: str | None = None, conversation=None, research_query: str | None = None) -> Answer:
        self.wait_for_learning()
        qid = uuid.uuid4().hex[:6]
        log.info("=== query %s: %r", qid, query[:200])
        try:
            with stage(log, f"query {qid} total"):
                ans = self._ask(query, model_override, conversation, research_query)
        except Exception:
            self.metrics["errors"] += 1
            raise
        log.info("=== query %s answered via %s; judge=%s; sources=%d", qid, ans.route.value,
                 ans.debug.get("decision_counts"), len(ans.sources))
        return ans

    def _ask(self, query: str, model_override=None, conversation=None, research_query: str | None = None) -> Answer:
        self.metrics["queries"] += 1
        with stage(log, "select column"):
            model, model_reason = self.adapters.model_for(query)
        if model_override:
            model, model_reason = model_override, model_override
        log.info("model: %s", model_reason)
        stm = conversation if conversation is not None else self.stm
        original = query
        previous = [t.text for t in stm.turns if t.role == "user"]
        with stage(log, "rewrite follow-up"):
            query = research_query or rewrite_followup(self.llm, query, previous)
        history = stm.render(query)
        # start the web search while the router decides (its model probe takes seconds);
        # if the answer comes from memory or the model itself, the result is simply unused
        prefetch = None
        if not research_query and not self.router.memory_covers(query):
            prefetch = self._prefetch_pool.submit(self.search.search, query, self.cfg.search.max_results)
        with stage(log, "route"):
            rd = self.router.route(query, column_model=model, history=history, require_sources=bool(research_query))
        log.info("route=%s (%s); memory hits=%d", rd.route.value, rd.reason, len(rd.memory_hits))
        self.metrics[f"route_{rd.route.value}"] += 1
        debug: dict = {"route_reason": rd.reason, "model": model_reason}
        if query != original:
            debug["rewritten_query"] = query
        rec = _AnswerRecord()
        decisions: list[JudgeDecision] = []
        pending_learn = None

        if rd.route == Route.PARAMETRIC and rd.draft is not None:
            debug["generator"] = f"{rd.draft.provider}:{rd.draft.model}"
            text, contexts, sources = rd.draft.text, [], []
        else:
            ctx: list[tuple[str, str]] = []
            sources: list[str] = []
            if rd.route == Route.MEMORY:
                hits = self.ltm.search(query, k=self.cfg.memory.top_k, touch=True)
                hits = [(it, s) for it, s in hits if s >= self.cfg.memory.hit_threshold * 0.9]
                for it, _ in hits:
                    ctx.append((it.knowledge, f"memory #{it.id}, R={it.reliability:.2f}"))
                    sources.extend(it.sources[:1])
                    rec.memory_ids.append(it.id)
                    # retrieval raises C_i -> re-judge; frequently used facts get promoted to Learn
                    decisions.append(self._judge_item(it))
                    self._apply_decision(it, decisions[-1])
            else:
                self.metrics["web_searches"] += 1
                with stage(log, "web search", providers=[p.name for p in self.search.providers]):
                    results = (prefetch.result() if prefetch is not None
                               else self.search.search(query, self.cfg.search.max_results))
                debug["search_errors"] = getattr(self.search, "last_errors", [])
                for e in debug["search_errors"]:
                    log.warning("search error: %s", e)
                log.info("search returned %d result(s)", len(results))
                if not results:
                    log.warning("no web results - answering without web context")
                c = self.cfg.chunk
                with stage(log, "fetch + chunk pages", n=len(results), fetch=self.cfg.search.fetch_pages):
                    chunks = results_to_chunks(results, self.cfg.search.fetch_pages, c.min_tokens,
                                               c.max_tokens, c.overlap_tokens,
                                               self.cfg.search.fetch_budget_s, self.cfg.search.fetch_timeout_s)
                log.info("%d chunk(s) from %d page(s)", len(chunks), len(results))
                # dedup is per batch: re-observing a known fact in a later query must
                # reach the LTM merge step, because recurrence is the utility signal U
                dedup = Deduplicator(self.cfg.dedup.threshold, self.cfg.dedup.num_perm,
                                     self.cfg.dedup.shingle_size)
                chunks, dropped = dedup.filter(chunks)
                n_before = len(chunks)
                chunks = select_chunks(chunks, query, c.max_chunks, c.max_chunks_per_page)
                if len(chunks) < n_before:
                    log.info("kept the %d most relevant of %d chunks", len(chunks), n_before)
                self.metrics["dedup_dropped"] += dropped
                log.info("dedup kept %d, dropped %d", len(chunks), dropped)
                # cheap signals for every chunk; the slow NLI contradiction check only for the
                # chunks that will actually be shown to the model, the rest after answering
                with stage(log, "verify", chunks=len(chunks)):
                    verified, vecs = self.verifier.verify(chunks, run_nli=False)
                qv = self.embedder.embed([query])[0]
                order = sorted(range(len(verified)),
                               key=lambda i: -(0.6 * float(vecs[i] @ qv) + 0.4 * verified[i].score))
                order = [i for i in order if verified[i].score >= self.cfg.verify.min_verified_score]
                top_k = self.cfg.memory.top_k
                checked: set[int] = set()
                picked: list[int] = []
                with stage(log, "contradiction check (answer chunks)"):
                    while len(picked) < top_k and len(checked) < len(order):
                        batch = [i for i in order if i not in checked][: top_k - len(picked) + 1]
                        self.verifier.check_contradictions(verified, vecs, batch)
                        checked.update(batch)
                        picked += [i for i in batch if not verified[i].blocked]
                picked = sorted(picked, key=order.index)[:top_k]
                log.info("verified: %d of %d checked chunk(s) blocked; %d used",
                         sum(verified[i].blocked for i in checked), len(checked), len(picked))
                for i in picked:
                    vc = verified[i]
                    ctx.append((vc.chunk.text, f"{vc.chunk.domain}, verified {vc.score:.2f}"))
                    sources.append(vc.chunk.url)
                    rec.domains.append(vc.chunk.domain)
                # include any decent memory hits too
                for it, s in rd.memory_hits[:2]:
                    ctx.append((it.knowledge, f"memory #{it.id}"))
                    rec.memory_ids.append(it.id)
                # every verified candidate goes through the Learning Judge
                if self.cfg.learn.background:
                    pending_learn = (verified, vecs, checked)   # started after generation
                    debug["learning"] = "running in the background after this answer"
                else:
                    decisions += self._learn_from(verified, vecs, checked)
                debug["verification"] = [
                    {"url": vc.chunk.url, "auth": round(vc.authority, 2), "cons": round(vc.consensus, 2),
                     "rec": round(vc.recency, 2), "contra": round(vc.contradiction, 2),
                     "support": vc.supporting_domains, "blocked": vc.blocked} for vc in verified]
            msgs = build_messages(query, ctx, history)
            with stage(log, "generate answer", model=model, contexts=len(ctx)):
                gen = self.llm.chat(msgs, model=model, max_tokens=self.cfg.llm.max_tokens,
                                    temperature=self.cfg.llm.temperature)
            for e in self.llm.last_errors:
                log.warning("LLM provider error (fell back): %s", e)
            log.info("generated %d chars with %s:%s", len(gen.text), gen.provider, gen.model)
            text, contexts = gen.text, [c for c, _ in ctx]
            if research_query and rd.route == Route.WEB and not sources:
                text = "I searched for supporting sources but couldn’t verify an answer to this question."
            debug["generator"] = f"{gen.provider}:{gen.model}"

        if pending_learn is not None:
            self._learn_in_background(*pending_learn)
        stm.add("user", original)
        stm.add("assistant", text)
        answer_id = uuid.uuid4().hex[:10]
        self._answers[answer_id] = rec
        debug["answer_id"] = answer_id
        debug["decision_counts"] = dict(Counter(d.action.value for d in decisions))
        return Answer(text=text, route=rd.route, contexts=contexts, sources=list(dict.fromkeys(sources)),
                      decisions=decisions, debug=debug)

    # ------------------------------------------------------------ feedback (§4.I)
    def feedback(self, answer_id: str, rating: int, correction: str = "") -> dict:
        """rating: +1 helpful / -1 wrong. Feedback only changes memory scores and
        source reputation; it never touches model weights directly."""
        self.wait_for_learning()
        rec = self._answers.get(answer_id)
        if rec is None:
            raise KeyError(f"unknown answer id {answer_id}")
        rating = 1 if rating > 0 else -1
        changed = []
        for mid in rec.memory_ids:
            item = self.ltm.get(mid)
            if not item:
                continue
            item.feedback = 0.7 * item.feedback + 0.3 * rating
            item.reliability = float(np.clip(item.reliability + 0.05 * rating, 0.05, 0.98))
            if rating < 0 and item.status == "queued":
                item.status = "ltm"
                self.queue.retract(item.id, "negative user feedback")
                self.metrics["retracted"] += 1
            d = self._judge_item(item)
            self._apply_decision(item, d)
            changed.append(mid)
        for dom in set(rec.domains):
            self.reputation.adjust(dom, 0.02 * rating)
        if correction.strip():
            item = MemoryItem(knowledge=correction.strip(), sources=["user-correction"], reliability=0.65,
                              timestamp=utcnow(), domain=self.cfg.domain, quality=0.5, stability=0.6,
                              feedback=0.3, access_count=1)
            self.ltm.add(item)
            changed.append(item.id)
        self.ltm.log("feedback", {"answer_id": answer_id, "rating": rating, "items": changed})
        self.metrics["feedback"] += 1
        return {"updated_memory_ids": changed}

    def sync_training_results(self, run_dir: str) -> dict:
        """Apply a trainer run directory: copy its columns.json and apply trained.json
        (accepted ids -> 'learned'; rejected, i.e. rolled back -> 'ltm' so RAG still serves them)."""
        import json
        import shutil

        self.wait_for_learning()
        run = Path(run_dir)
        if (run / "columns.json").exists():
            shutil.copy(run / "columns.json", Path(self.cfg.data_dir) / "columns.json")
        res = json.loads((run / "trained.json").read_text())
        n_ok = n_back = 0
        for mid in res.get("accepted", []):
            item = self.ltm.get(int(mid))
            if item:
                item.status = "learned"
                self.ltm.update(item)
                n_ok += 1
        for mid in res.get("rejected", []):
            item = self.ltm.get(int(mid))
            if item and item.status == "queued":
                item.status = "ltm"
                self.ltm.update(item)
                n_back += 1
        self.adapters.reload()
        self.ltm.log("sync", {"accepted": n_ok, "rejected": n_back})
        return {"learned": n_ok, "returned_to_ltm": n_back, "columns": len(self.adapters.columns)}

    def stats(self) -> dict:
        self.wait_for_learning()
        m = dict(self.metrics)
        cands = m.get("candidates", 0)
        if cands:
            m["pct_learn"] = round(100 * (m.get("learn", 0) + m.get("promoted_to_learn", 0)) / cands, 1)
            m["pct_store"] = round(100 * m.get("store", 0) / cands, 1)
            m["pct_discard"] = round(100 * m.get("discard", 0) / cands, 1)
        m["memory"] = self.ltm.stats()
        m["queue_pending"] = len(self.queue.pending())
        return m

