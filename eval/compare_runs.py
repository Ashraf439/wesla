"""Compare trainer runs: WESLA (selective) vs the base-paper baseline (train-all).

    python -m eval.compare_runs runs/wesla runs/baseline [--markdown]

Aggregates every report_*.json in each workdir:
  candidates seen / items trained / % trained, parameter-update jobs, rollbacks,
  GPU minutes, columns allocated, mean dPPL on prior holdouts and on the general set,
  new-knowledge PPL drop.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarise(workdir: str, n_candidates: int | None = None) -> dict:
    reps = [json.loads(p.read_text()) for p in sorted(Path(workdir).glob("report_*.json"))]
    jobs = [j for r in reps for j in r.get("jobs", [])]
    ok = [j for j in jobs if j["passed"]]

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    trained = sum(j["n_new"] for j in ok)
    cols = json.loads((Path(workdir) / "columns.json").read_text()).get("columns", []) \
        if (Path(workdir) / "columns.json").exists() else []
    out = {
        "runs": len(reps), "update_jobs": len(jobs), "accepted_jobs": len(ok),
        "rollbacks": len(jobs) - len(ok), "items_trained": trained,
        "gpu_minutes": sum(j.get("gpu_minutes", 0) for j in jobs),
        "columns": len(cols),
        "mean_dppl_prior": mean([j["dppl_prior"] for j in jobs if j["kind"] == "reuse"]),
        "mean_dppl_general": mean([j["dppl_general"] for j in jobs]),
        "mean_new_ppl_drop": mean([j["before"]["new"] - j["after"]["new"] for j in ok]),
    }
    if n_candidates:
        out["pct_candidates_trained"] = 100 * trained / n_candidates
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("wesla_dir")
    ap.add_argument("baseline_dir")
    ap.add_argument("--candidates", default="data/candidates.jsonl",
                    help="to express trained items as %% of all candidates")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    n = None
    if Path(args.candidates).exists():
        n = sum(1 for l in Path(args.candidates).read_text(encoding="utf-8").splitlines() if l.strip())
    a, b = summarise(args.wesla_dir, n), summarise(args.baseline_dir, n)
    keys = list(dict.fromkeys(list(a) + list(b)))
    if args.markdown:
        print("| metric | WESLA | baseline (train-all) |\n|---|---|---|")
        for k in keys:
            f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"| {k} | {f(a.get(k, ''))} | {f(b.get(k, ''))} |")
    else:
        print(json.dumps({"wesla": a, "baseline": b}, indent=1))
    if a.get("items_trained") is not None and b.get("items_trained"):
        print(f"\nReduction in trained items vs baseline: {100 * (1 - a['items_trained'] / b['items_trained']):.1f}%")
    return a, b


if __name__ == "__main__":
    main()
