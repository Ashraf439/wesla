# WESLA — implementation (Mini-WESLA, phase 1)

Web-Enhanced Self-Learning Agent with Adaptive Memory and Selective Continual Learning.
This repo implements the architecture in `proposedSolution.docx` on a free stack:
a live CPU app (laptop / HF Space), plus an offline GPU trainer (Kaggle / Colab).

```
Query → Router → [Parametric | Memory | Web] → Extract + MinHash dedup → Verification
      → Dynamic RAG → Answer → Learning Judge → [Discard | Store (LTM) | Learn-queue]
Learn-queue ──(HF dataset repo)──► Kaggle trainer: domain clustering → LoRA column (+EWC, replay,
      frozen lateral blocks) → ΔPPL validation gate → accept / rollback ──► columns.json → app
```

## Where each proposal component lives

| Proposal § | Component | File |
|---|---|---|
| 4.A | Query Router (fresh-cue → memory hit → entropy probe → web) | `wesla/router.py` |
| 4.B | Web search (ddgs → Tavily, backoff), extraction, 256–512-token chunks, MinHash LSH | `wesla/search.py`, `wesla/extract.py`, `wesla/dedup.py` |
| 4.C | Dynamic RAG prompt | `wesla/rag.py` |
| 4.D | STM (attention-weighted eviction) + LTM 5-tuple (K,S,R,T,C) in SQLite + FAISS, decay & prune | `wesla/memory.py`, `wesla/vector_index.py` |
| 4.E | Verification: authority, multi-source consensus, recency, NLI contradiction vs trusted memory | `wesla/verify.py` |
| 4.F | **Learning Judge** L_s = w_r R + w_q Q + w_n N + w_u U + w_s S, τ1/τ2 tiers | `wesla/judge.py` |
| 4.G | Adaptive PNN domain clustering (reuse if s* ≥ θ, new column only for large novel groups, cap) | `trainer/clustering.py` |
| 4.H | LoRA column training + EWC (Fisher on LoRA params) + replay + ΔPPL ≤ 0.3 gate with rollback | `trainer/lora_ops.py`, `trainer/train.py`, `trainer/replay.py` |
| 4.I | Feedback → memory utility/reliability + source reputation only (never weights); retraction from queue | `WESLA.feedback` in `wesla/agent.py` |

**PNN column = LoRA adapter.** "Lateral connections" are implemented as *frozen rank blocks*: a new
column's adapter is `[w₁A₁; w₂A₂; A_new]`, `[B₁, B₂, B_new]`, which reproduces
`Σ wᵢ·ΔWᵢ + ΔW_new` exactly while only the new block trains. The result is a normal PEFT adapter,
so it exports to GGUF and runs in Ollama.

Design choices beyond the proposal text (all configurable in `config.toml`):
- **Hard gates on Learn:** even with L_s ≥ τ2, a fact is only queued if S ≥ 0.5 (not volatile),
  Q ≥ 0.6 (multi-source), it has recurred at least once, and net feedback is not negative.
  First sightings go to LTM and get promoted when they recur or are retrieved again.
- **Dedup is per query batch.** Near-duplicates are merged and their URLs kept as `mirrors`
  (counted as syndicated support); a fact re-seen in a later query reaches the LTM merge step, which
  is what raises U.
- **Novelty N** is measured against knowledge already learned or queued, not against all memory.

## Quick start (Windows / macOS / Linux)

```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                # 25 offline tests, no models needed
python -m wesla demo            # offline walkthrough of every Judge outcome
```

Live mode:
```bash
ollama pull llama3.2:1b
set GROQ_API_KEY=...            # optional fallback generator (export on macOS/Linux)
set TAVILY_API_KEY=...          # optional second search provider
python -m wesla ui              # Gradio at http://localhost:7860
python -m wesla chat            # terminal; '+' / '-' / '-! correction' give feedback
python -m wesla stats
```
The first run downloads `bge-small-en-v1.5` (~130 MB) and the NLI cross-encoder (~140 MB).

### Logs and troubleshooting
- Every run writes `data/logs/wesla.log` (full DEBUG detail, rotated at 5 MB). Each query is logged
  stage by stage (`>> web search` ... `<< web search done in 3.2s`), so a hang shows the stage it is
  stuck in and an error shows the full traceback.
- Native crashes that kill Python without a traceback (e.g. the Windows OpenMP clash between
  faiss-cpu and PyTorch) are written to `data/logs/crash.log` by `faulthandler`. On Windows the
  NumPy vector index is used by default to avoid that clash (`WESLA_USE_FAISS=1` forces faiss).
- In the UI, errors appear in the chat and the **Logs** tab shows the end of the log; the server keeps running.
- `python -m wesla doctor` checks each dependency on its own (imports, torch/CUDA, Ollama + model,
  embedder, NLI, search, page fetch, generation). `--quick` skips the network checks.
- `python -m wesla logs -n 100` prints the end of the log; `--log-level DEBUG` (before the command)
  makes the console verbose too.

### Calibrate thresholds first
Similarity thresholds depend on the embedder. With bge-small, ask ~20 questions, then look at the
router reasons and verification JSON in the UI and tune `memory.hit_threshold`,
`verify.consensus_sim`, and `judge.merge_sim`. Ollama's token logprobs (needed for the entropy
route) only exist in recent Ollama versions; without them the router falls back to the web, and the
UI's route reason says "no logprobs from provider". Update Ollama if you see that.

## Training loop

1. In the app: `python -m wesla push-queue` (set `queue.hf_repo` and `HF_TOKEN`). This uploads
   `train_queue.jsonl` and `candidates.jsonl`.
2. On Kaggle: open `notebooks/kaggle_trainer.ipynb` (GPU T4, Internet on, `HF_TOKEN` secret). It runs
   a dry-run plan, the WESLA training run, the train-all baseline, the comparison table, GGUF export,
   and saves state back to the Hub.
   Or directly: `python -m trainer.train --hf-queue user/wesla-train-queue --workdir runs/wesla`
3. Back on the laptop: download `runs/wesla`, then
   ```bash
   cd runs/wesla/ollama && ollama create wesla-col_000 -f Modelfile.col_000
   python -m wesla sync runs/wesla
   ```
   Accepted items become `learned`; rolled-back items go back to `ltm` and RAG still serves them.
   Queries close to a column's centroid now use that column's model; if the Ollama model isn't
   created yet, the app falls back to the base model.

## Evaluation (what to report)

- `python -m eval.run_qa --questions your_set.jsonl` gives post-cutoff QA accuracy, route mix, Judge
  mix and latency. Convert RealtimeQA / StreamingQA / a domain set to `{"question","answers"}` lines.
- `python -m eval.compare_runs runs/wesla runs/baseline --markdown` gives items trained, % of
  candidates trained, update jobs, rollbacks, GPU minutes, columns allocated, mean ΔPPL on prior
  holdouts and the general set, and the new-knowledge PPL drop: WESLA vs the base-paper-style
  baseline.

Report the "70–85 % fewer updates" and "zero forgetting" claims as **measured** numbers from these
scripts, not as givens.

## Status / limitations

- The live app and the trainer's planning logic (clustering, replay, registry, queue) are covered by
  offline tests. **The GPU code path (`trainer/lora_ops.py`, `run_job`) has not been executed yet.**
  Run it on Kaggle first with `--epochs 1` on a small queue.
- `HeuristicNLI` and `HashingEmbedder` are test/demo stand-ins; real runs use the cross-encoder and bge.
- Mini-WESLA scope: one domain, ≤ 3 columns (`--max-columns`). Next steps: multi-domain eval, an LLM-based
  stability estimate for S, learned Judge weights, and query-time adapter mixing.
