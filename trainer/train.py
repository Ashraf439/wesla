"""Offline selective continual-learning job (proposal §4.G-H).

Run on a Kaggle / Colab GPU:

    python -m trainer.train --queue data/train_queue.jsonl --workdir runs/wesla \
        --base-model unsloth/Llama-3.2-1B-Instruct

Pipeline per run:
 1. read queue, apply retractions, skip already-trained ids
 2. adaptive domain clustering: reuse a column if s* >= theta, else new column (min size, cap)
 3. for each column job: LoRA training (+EWC for reused columns, +replay, frozen lateral
    blocks for new columns), then the validation gate:
        dPPL(prior holdout) <= delta  and  dPPL(general set) <= delta
    pass -> commit new adapter version, update Fisher/replay/registry
    fail -> rollback (nothing written), items returned to LTM
 4. write columns.json (for the app), trained.json (accepted/rejected ids), report.json

`--mode train-all` is the base-paper baseline: every candidate is trained, one new
column per run, no Judge filter, no EWC/replay, no gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .clustering import assign, lateral_parents, update_centroid
from .registry import Column, Registry
from .replay import ReplayBuffer


# ------------------------------------------------------------------ queue I/O
def read_queue(path: str | None, hf_repo: str | None) -> list[dict]:
    if hf_repo:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(hf_repo, "train_queue.jsonl", repo_type="dataset")
    learn: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "learn":
            learn[r["memory_id"]] = r
        elif r.get("kind") == "retract":
            learn.pop(r["memory_id"], None)
    return list(learn.values())


def read_candidates(path: str) -> list[dict]:
    """Baseline input: every verified candidate the app saw (data/candidates.jsonl)."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            r = json.loads(line)
            out.append({"memory_id": -(i + 1), "text": r["text"], "vec": r.get("vec")})
    return out


def embed_missing(records: list[dict], model_name: str) -> np.ndarray:
    missing = [i for i, r in enumerate(records) if not r.get("vec")]
    if missing:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_name)
        vs = st.encode([records[i]["text"] for i in missing], normalize_embeddings=True)
        for i, v in zip(missing, vs):
            records[i]["vec"] = v.tolist()
    return np.asarray([r["vec"] for r in records], dtype=np.float32)


def load_general_set(path: str | None, n: int = 200) -> list[str]:
    """Reference text for 'did we damage general ability?'. Default: WikiText-2 test."""
    if path:
        txt = Path(path).read_text(encoding="utf-8")
        paras = [p.strip() for p in txt.split("\n\n") if len(p.split()) > 30]
        return paras[:n]
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    paras = [t.strip() for t in ds["text"] if len(t.split()) > 40]
    return paras[:n]


def fmt(texts: list[str], eos: str) -> list[str]:
    return [t.strip() + eos for t in texts]


# ------------------------------------------------------------------ one column job
def run_job(args, reg: Registry, job: dict, general: list[str], log) -> dict:
    import torch

    from . import lora_ops as L

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(args.workdir)
    t0 = time.time()
    base, tok = L.load_base(args.base_model, dev)
    eos = tok.eos_token or ""
    new_texts = fmt([r["text"] for r in job["records"]], eos)
    general_f = fmt(general, eos)

    if job["kind"] == "reuse":
        col = reg.get(job["column"])
        replay = ReplayBuffer(work / "columns" / col.id / "replay.json")
        model = L.load_column(base, col.adapter_dir)
        frozen = col.frozen_rank
        ewc = None if args.mode == "train-all" else L.load_ewc(Path(col.adapter_dir) / "ewc.pt", dev)
        prior = fmt(replay.holdout, eos)
        before = {"prior": L.perplexity(model, tok, prior, args.max_len, device=dev),
                  "general": L.perplexity(model, tok, general_f, args.max_len, device=dev),
                  "new": L.perplexity(model, tok, new_texts, args.max_len, device=dev)}
        train_texts = new_texts if args.mode == "train-all" else replay.mix(new_texts, args.replay_ratio)
        lam = 0.0 if args.mode == "train-all" else args.ewc_lambda
    else:
        cid = reg.new_id()
        centroid = job["vecs"].mean(axis=0)
        parents, weights = ([], []) if args.mode == "train-all" else lateral_parents(reg, centroid, k=args.lateral_k)
        pinfo, kept_p, kept_w = [], [], []
        for pid, w in zip(parents, weights):
            pc = reg.get(pid)
            if sum(x[2] for x in pinfo) + pc.rank + args.rank <= args.max_rank:
                pinfo.append((pc.adapter_dir, w, pc.rank))
                kept_p.append(pid)
                kept_w.append(w)
        before = {"prior": float("nan"),
                  "general": L.perplexity(base, tok, general_f, args.max_len, device=dev),
                  "new": L.perplexity(base, tok, new_texts, args.max_len, device=dev)}
        model, frozen = L.new_column(base, args.rank, pinfo)
        col = Column(id=cid, name=f"{args.domain}-{cid}", centroid=[float(x) for x in centroid],
                     parents=kept_p, parent_weights=kept_w,
                     rank=frozen + args.rank, frozen_rank=frozen)
        replay = ReplayBuffer(work / "columns" / cid / "replay.json")
        prior = []
        train_texts, ewc, lam = new_texts, None, 0.0

    stats = L.train_lora(model, tok, list(train_texts), ewc=ewc, lam=lam, frozen_rank=frozen,
                         epochs=args.epochs, lr=args.lr, bs=args.batch_size, grad_accum=args.grad_accum,
                         max_len=args.max_len, device=dev, log=log)
    after = {"prior": L.perplexity(model, tok, prior, args.max_len, device=dev) if prior else float("nan"),
             "general": L.perplexity(model, tok, general_f, args.max_len, device=dev),
             "new": L.perplexity(model, tok, new_texts, args.max_len, device=dev)}
    d_prior = after["prior"] - before["prior"] if prior else 0.0
    d_gen = after["general"] - before["general"]
    passed = args.mode == "train-all" or (d_prior <= args.delta_ppl and d_gen <= args.delta_ppl)
    result = {"kind": job["kind"], "column": col.id, "n_new": len(new_texts), "n_train": len(train_texts),
              "before": before, "after": after, "dppl_prior": d_prior, "dppl_general": d_gen,
              "passed": passed, "steps": stats["steps"]}

    if passed:
        col.version += 1
        out = work / "columns" / col.id / f"v{col.version}"
        model.save_pretrained(str(out))
        if args.mode != "train-all":
            fisher_texts = replay.sample(64) + new_texts[:64]
            fisher = L.fisher_diagonal(model, tok, fisher_texts, args.max_len, dev)
            L.save_ewc(out / "ewc.pt", model, fisher)
        replay.add([r["text"] for r in job["records"]])
        replay.save()
        col.adapter_dir = str(out)
        col.ollama_model = f"wesla-{col.id}"
        col.trained_ids += [r["memory_id"] for r in job["records"]]
        update_centroid(col, job["vecs"])
        col.history.append({k: v for k, v in result.items() if k != "kind"} | {"version": col.version})
        if job["kind"] == "new":
            reg.columns.append(col)
        log(f"  ACCEPT {col.id} v{col.version}  dPPL prior={d_prior:+.3f} general={d_gen:+.3f}  "
            f"new-knowledge PPL {before['new']:.2f} -> {after['new']:.2f}")
    else:
        log(f"  ROLLBACK {col.id}  dPPL prior={d_prior:+.3f} general={d_gen:+.3f} > {args.delta_ppl}")

    result["gpu_minutes"] = (time.time() - t0) / 60 if dev == "cuda" else 0.0
    result["wall_minutes"] = (time.time() - t0) / 60
    del model, base
    if dev == "cuda":
        torch.cuda.empty_cache()
    return result


# ------------------------------------------------------------------ export
def export_gguf(col: Column, base_hf_dir: str, llama_cpp: str, ollama_base: str, out_dir: Path) -> Path:
    """Convert a PEFT adapter to GGUF and write an Ollama Modelfile next to it."""
    import subprocess
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    gguf = out_dir / f"{col.id}.gguf"
    base_flag = "--base" if Path(base_hf_dir).exists() else "--base-model-id"
    subprocess.run([sys.executable, str(Path(llama_cpp) / "convert_lora_to_gguf.py"), base_flag, base_hf_dir,
                    "--outfile", str(gguf), "--outtype", "f16", col.adapter_dir], check=True)
    (out_dir / f"Modelfile.{col.id}").write_text(f"FROM {ollama_base}\nADAPTER ./{gguf.name}\n")
    return gguf


# ------------------------------------------------------------------ main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="data/train_queue.jsonl")
    ap.add_argument("--hf-queue", default=None, help="HF dataset repo holding train_queue.jsonl")
    ap.add_argument("--candidates", default=None, help="baseline: data/candidates.jsonl")
    ap.add_argument("--workdir", default="runs/wesla")
    ap.add_argument("--mode", choices=["wesla", "train-all"], default="wesla")
    ap.add_argument("--domain", default="tech-news")
    ap.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--general-eval", default=None, help="text file; default WikiText-2 test")
    ap.add_argument("--theta", type=float, default=0.55)
    ap.add_argument("--min-new-items", type=int, default=8)
    ap.add_argument("--max-columns", type=int, default=3)
    ap.add_argument("--min-items", type=int, default=4, help="skip a run with fewer queued items")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--lateral-k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--ewc-lambda", type=float, default=50.0)
    ap.add_argument("--replay-ratio", type=float, default=0.3)
    ap.add_argument("--delta-ppl", type=float, default=0.3)
    ap.add_argument("--export-gguf", default=None, help="path to a llama.cpp checkout")
    ap.add_argument("--ollama-base", default="llama3.2:1b")
    ap.add_argument("--push-hf", default=None, help="HF user/org to push accepted adapters to")
    ap.add_argument("--dry-run", action="store_true", help="cluster and plan only; no torch")
    args = ap.parse_args(argv)

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    reg_path = work / "columns.json"
    reg = Registry.load(reg_path)
    reg.embedder, reg.base_model, reg.theta_domain = args.embedder, args.base_model, args.theta

    if args.mode == "train-all":
        records = read_candidates(args.candidates or "data/candidates.jsonl")
    else:
        done = reg.all_trained_ids()
        records = [r for r in read_queue(args.queue, args.hf_queue) if r["memory_id"] not in done]
    log(f"[{time.strftime('%Y-%m-%d %H:%M')}] mode={args.mode} items={len(records)} columns={len(reg.columns)}")
    report = {"mode": args.mode, "items": len(records), "jobs": [], "accepted": [], "rejected": [],
              "deferred": []}
    if len(records) < args.min_items:
        log(f"  fewer than --min-items={args.min_items}; nothing to do")
        return report

    vecs = embed_missing(records, args.embedder)
    if args.mode == "train-all":
        jobs = [{"kind": "new", "records": records, "vecs": vecs}]
    else:
        a = assign(reg, vecs, args.theta, args.min_new_items, args.max_columns)
        jobs = [{"kind": "reuse", "column": cid, "records": [records[i] for i in idx], "vecs": vecs[idx]}
                for cid, idx in a.reuse.items()]
        jobs += [{"kind": "new", "records": [records[i] for i in g], "vecs": vecs[g]} for g in a.new]
        report["deferred"] = [records[i]["memory_id"] for i in a.deferred]
    for j in jobs:
        log(f"  plan: {j['kind']:5s} {j.get('column', '(new)'):8s} items={len(j['records'])}")
    if report["deferred"]:
        log(f"  deferred {len(report['deferred'])} item(s): too few for a new column")
    if args.dry_run:
        return report

    general = load_general_set(args.general_eval)
    for j in jobs:
        res = run_job(args, reg, j, general, log)
        report["jobs"].append(res)
        ids = [r["memory_id"] for r in j["records"]]
        (report["accepted"] if res["passed"] else report["rejected"]).extend(ids)
        reg.save(reg_path)  # checkpoint after every job (Kaggle sessions can die)

    if args.export_gguf:
        for c in reg.trained():
            g = export_gguf(c, args.base_model, args.export_gguf, args.ollama_base, work / "ollama")
            log(f"  exported {g}")
    if args.push_hf:
        from huggingface_hub import HfApi

        api = HfApi()
        for c in reg.trained():
            repo = f"{args.push_hf}/wesla-{c.id}"
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_folder(folder_path=c.adapter_dir, repo_id=repo, commit_message=f"{c.id} v{c.version}")
            c.hf_repo = repo
        reg.save(reg_path)

    report["gpu_minutes"] = sum(j["gpu_minutes"] for j in report["jobs"])
    report["columns"] = len(reg.columns)
    (work / "trained.json").write_text(json.dumps(
        {"accepted": report["accepted"], "rejected": report["rejected"]}, indent=1))
    (work / f"report_{int(time.time())}.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"  done: accepted={len(report['accepted'])} rejected={len(report['rejected'])} "
        f"gpu_min={report['gpu_minutes']:.1f}")
    return report


if __name__ == "__main__":
    main()
