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

## Run WESLA locally

The chat app runs on a CPU. Install Git, Python (tested with 3.12), and Ollama. Start Ollama, then clone this version and create a Python environment.

```sh
git clone --branch feature/document-aware-chat https://github.com/Ashraf439/wesla.git
cd wesla
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then run:

```sh
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull qwen3-vl:2b-instruct
python -m wesla ui
```

Open [the local chat workspace](http://localhost:7860). The second Ollama model handles photos and scanned documents. The first app launch also downloads the embedding and verification models. Web research needs internet access. The default search provider works without an API key. Set `TAVILY_API_KEY` for the optional second search provider or `GROQ_API_KEY` for the optional generation fallback.

Run `python -m pytest -q` for the offline tests, `python -m wesla demo` for the offline walkthrough, or `python -m wesla chat` for the terminal interface.

## Ask about a file or part of an answer

1. Use **+** in the composer to attach a file and send a question. Select its card above your message to preview it. Open **Artifacts** to download, remove, or attach a saved file to another chat.
2. Ask a follow-up in the same chat. For example, after uploading a certificate, ask “Which college is named?” WESLA can answer from the page. To keep the answer within the document, say “Use only the PDF.”
3. For outside information, ask “Tell me more about that college.” WESLA uses the excerpt and recent conversation to identify the college. When it recognizes that the question needs outside facts, it checks long-term memory and searches the web if coverage is insufficient. Time-sensitive lookups go to the web. Open the answer's sources to inspect the evidence.
4. Highlight text in an answer and choose **Add to chat** to quote it in your next message. Choose **Ask in side chat** for a separate follow-up that preserves the main draft. Closing the side panel ends that side conversation.

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

### Speed and follow-up questions
- A web answer used to take 50–390 s, almost all of it in NLI verification of every chunk and in
  fetching pages one by one. Now pages are fetched in parallel with a deadline
  (`search.fetch_budget_s`), only the `chunk.max_chunks` most query-relevant chunks are processed, the
  web search starts while the router is still deciding, and NLI runs in one batch and only on the
  chunks shown to the model. The remaining NLI checks and the Learning Judge run in a background thread
  after the answer is returned (`[learn] background`); the next question waits for them, so memory is
  never written twice at once. Latency varies with the CPU, model, network, and verification workload.
- Follow-ups such as "what rank is used for it?" are rewritten into standalone questions
  (`wesla/rewrite.py`) **before** routing, memory search and web search, using the earlier questions of
  the same chat. If the small model fails to rewrite, the previous question's topic words are attached
  instead. A new chat starts with no history, by design.

### Calibrate thresholds first
Similarity thresholds depend on the embedder. With bge-small, ask ~20 questions, then look at the
router reasons and verification JSON in the UI and tune `memory.hit_threshold`,
`verify.consensus_sim`, and `judge.merge_sim`. Ollama's token logprobs (needed for the entropy
route) only exist in recent Ollama versions; without them the router falls back to the web, and the
UI's route reason says "no logprobs from provider". Update Ollama if you see that.

## Inspect memory, uploads, and training

- **Memory & training → Short-term memory** previews the recent turns available from the
  selected chat. Actual recall is query-dependent. WESLA stores chat history in this
  browser and rebuilds the short-term context for each normal chat request. Use
  **Forget turn** or **Clear short-term memory** to exclude turns from future
  context and training exports while keeping the transcript visible.
- **Long-term memory** lists stored passages, their sources, and their status from
  `data/ltm.sqlite`. The Learning Judge can queue eligible records in
  `data/train_queue.jsonl`. Check the counts under **Training** for pending records and
  registered adapters. **Remove record** deletes that record from the SQLite
  store, vector index, and local queue. Copies in logs, exports, or existing
  adapters are separate.
- **Artifacts** holds original uploads under `data/artifacts`. From there you can
  download originals or attach readable documents to another chat. Unsupported uploads can be downloaded. Scanned PDFs use local OCR for up to 12
  scanned pages per file. On Windows, OCR uses an installed Windows OCR language. Photos use local vision via Ollama. Install the vision
  model with `ollama pull qwen3-vl:2b-instruct`. File analysis uses extracted excerpts and
  does not add them to the training queue.
- **Training evidence** appears in a trainer run's `train.log`, `report_*.json`,
  and `trained.json`. Inspect the completed jobs and accepted IDs before importing a
  run. An imported column also needs its exported Ollama adapter installed for model
  routing to use it.

### Train on a supplied dataset

Use **Export this chat for training** to build JSONL examples from the chat's
retained question-and-answer pairs. Review the answers before using the export.
The export prepares data for the trainer and chat itself does not update weights.

For a curated dataset, create a separate UTF-8 JSONL file with one factual passage per
line and a unique negative integer ID for each record. Negative IDs keep manually
supplied samples separate from existing SQLite memory IDs. Example:

```json
{"kind":"learn","memory_id":-1,"text":"Write one reviewed training passage here."}
```

On a GPU machine, install `requirements-train.txt` into a GPU-capable Python
environment and plan the run with the command below. By default the trainer
requires four eligible items to start and eight similar items to create a new column.

```bash
python -m trainer.train --queue my-data.jsonl --workdir runs/custom --dry-run
```

Remove `--dry-run` to train after reviewing the planned jobs. See the training loop
below for exporting and installing accepted adapters. Manually supplied passages
are adapter training data. Their negative IDs have no corresponding SQLite records
to mark as learned.

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
