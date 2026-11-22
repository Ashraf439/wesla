"""Post-cutoff QA evaluation of the live agent.

Input JSONL, one object per line:
    {"question": "...", "answers": ["gold", "alias", ...]}
(e.g. converted from RealtimeQA / StreamingQA / your own domain set)

    python -m eval.run_qa --questions eval/questions.example.jsonl --out runs/qa_wesla.json

Reports accuracy (normalised alias containment), route distribution, Learning
Judge distribution, web-search count and latency. Run it twice - once on a fresh
data dir and once after training + `wesla sync` - to see the parametric gain.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def correct(pred: str, golds: list[str]) -> bool:
    p = norm(pred)
    return any(norm(g) and norm(g) in p for g in golds)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/qa_report.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from wesla.agent import WESLA
    from wesla.config import load_config

    agent = WESLA(load_config(args.config))
    rows = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    per, routes, acts = [], Counter(), Counter()
    for r in rows:
        t0 = time.time()
        ans = agent.ask(r["question"])
        dt = time.time() - t0
        ok = correct(ans.text, r["answers"])
        routes[ans.route.value] += 1
        acts.update(d.action.value for d in ans.decisions)
        per.append({"q": r["question"], "ok": ok, "route": ans.route.value, "latency_s": round(dt, 2),
                    "pred": ans.text[:400]})
        print(f"[{'OK' if ok else '--'}] {ans.route.value:10s} {dt:5.1f}s  {r['question'][:70]}")
    n = max(1, len(per))
    rep = {"n": len(per), "accuracy": sum(p["ok"] for p in per) / n,
           "accuracy_by_route": {rt: sum(p["ok"] for p in per if p["route"] == rt) / c for rt, c in routes.items()},
           "routes": dict(routes), "judge_actions": dict(acts),
           "mean_latency_s": sum(p["latency_s"] for p in per) / n, "agent_stats": agent.stats(), "items": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, indent=1, default=str))
    return rep


if __name__ == "__main__":
    main()
