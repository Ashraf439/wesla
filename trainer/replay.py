"""Experience replay buffer per column (proposal §4.H). Pure Python.

Each column keeps a reservoir of texts it was trained on. When a column is
updated, a replay sample is mixed into the new batch; when a column is
evaluated, a held-out slice is its "prior task" set for the dPPL gate.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class ReplayBuffer:
    def __init__(self, path: str | Path, capacity: int = 2000, holdout_frac: float = 0.15, seed: int = 0):
        self.path = Path(path)
        self.capacity = capacity
        self.holdout_frac = holdout_frac
        self.rng = random.Random(seed)
        self.train: list[str] = []
        self.holdout: list[str] = []
        self.seen = 0
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.train, self.holdout, self.seen = d["train"], d["holdout"], d["seen"]

    def add(self, texts: list[str]) -> None:
        """Reservoir sampling; a fixed fraction is routed to the holdout set."""
        for t in texts:
            self.seen += 1
            target = self.holdout if self.rng.random() < self.holdout_frac else self.train
            cap = int(self.capacity * (self.holdout_frac if target is self.holdout else 1 - self.holdout_frac))
            cap = max(cap, 1)
            if len(target) < cap:
                target.append(t)
            else:
                j = self.rng.randrange(self.seen)
                if j < cap:
                    target[j] = t

    def sample(self, n: int) -> list[str]:
        if not self.train or n <= 0:
            return []
        return self.rng.sample(self.train, min(n, len(self.train)))

    def mix(self, new_texts: list[str], ratio: float = 0.3) -> list[str]:
        """Return new + replay so that replay is ~ratio of the final batch."""
        n_replay = int(round(len(new_texts) * ratio / max(1e-9, 1 - ratio)))
        batch = list(new_texts) + self.sample(n_replay)
        self.rng.shuffle(batch)
        return batch

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"train": self.train, "holdout": self.holdout, "seen": self.seen}))
