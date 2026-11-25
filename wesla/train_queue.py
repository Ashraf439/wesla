"""Training queue between the live app and the offline trainer.

Append-only JSONL. Two record kinds:
  {"kind": "learn",   "memory_id": 12, "text": ..., "domain": ..., "score": ..., ...}
  {"kind": "retract", "memory_id": 12, "reason": "negative feedback"}
The trainer applies retractions before training. Optionally the file is pushed
to a Hugging Face *dataset* repo so a Kaggle notebook can pull it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .types import MemoryItem, utcnow


class TrainQueue:
    def __init__(self, path: str, hf_repo: str = ""):
        self.path = Path(path)
        self.hf_repo = hf_repo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, rec: dict) -> None:
        rec["at"] = utcnow().isoformat()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def enqueue(self, item: MemoryItem, components: dict[str, float], vec=None) -> None:
        # the embedding travels with the record so the trainer clusters in the same space
        self._append({"kind": "learn", "memory_id": item.id, "text": item.knowledge,
                      "vec": [round(float(x), 5) for x in vec] if vec is not None else None,
                      "domain": item.domain, "sources": item.sources, "score": item.learning_score,
                      "components": components, "timestamp": item.timestamp.isoformat()})

    def retract(self, memory_id: int, reason: str) -> None:
        self._append({"kind": "retract", "memory_id": memory_id, "reason": reason})

    def forget(self, memory_id: int) -> None:
        """Remove this record's payload from the local queue."""
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines
                if not line.strip() or json.loads(line).get("memory_id") != memory_id]
        if len(kept) != len(lines):
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                    dir=self.path.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
                temporary.writelines(kept)
            try:
                temp_path.replace(self.path)
            finally:
                temp_path.unlink(missing_ok=True)

    def pending(self) -> list[dict]:
        if not self.path.exists():
            return []
        learn: dict[int, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "learn":
                learn[r["memory_id"]] = r
            elif r["kind"] == "retract":
                learn.pop(r["memory_id"], None)
        return list(learn.values())

    def push_to_hub(self) -> str:
        if not self.hf_repo:
            raise RuntimeError("queue.hf_repo is not configured")
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(self.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(self.path), path_in_repo="train_queue.jsonl",
                        repo_id=self.hf_repo, repo_type="dataset",
                        commit_message=f"queue update {utcnow():%Y-%m-%d %H:%M}")
        cands = self.path.parent / "candidates.jsonl"  # needed for the train-all baseline
        if cands.exists():
            api.upload_file(path_or_fileobj=str(cands), path_in_repo="candidates.jsonl",
                            repo_id=self.hf_repo, repo_type="dataset", commit_message="candidates update")
        return f"https://huggingface.co/datasets/{self.hf_repo}"
