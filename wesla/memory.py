"""Dual-tier Adaptive Memory (proposal §4.D).

STM  - active conversation context with attention-weighted sliding eviction:
       each turn has a weight = recency decay x relevance to the current query;
       when the token budget is exceeded, the lowest-weight turn is evicted.
LTM  - SQLite store of 5-tuples M_i = (K_i, S_i, R_i, T_i, C_i) + a vector
       index for retrieval. Items decay exponentially when not accessed; items
       with high access counts become candidates for model learning.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .types import MemoryItem, utcnow
from .vector_index import VectorIndex


def _ntok(text: str) -> int:
    # cheap token estimate (~0.75 words per token for English)
    return max(1, int(len(text.split()) / 0.75))


# --------------------------------------------------------------------- STM
@dataclass
class Turn:
    role: str
    text: str
    vec: np.ndarray
    step: int


class ShortTermMemory:
    def __init__(self, embedder: Embedder, max_turns: int = 8, max_tokens: int = 2048,
                 recency_decay: float = 0.85):
        self.embedder = embedder
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.recency_decay = recency_decay
        self.turns: list[Turn] = []
        self._step = 0

    def add(self, role: str, text: str) -> None:
        self._step += 1
        vec = self.embedder.embed([text])[0]
        self.turns.append(Turn(role, text, vec, self._step))

    def weights(self, query_vec: np.ndarray | None) -> list[float]:
        out = []
        for t in self.turns:
            rec = self.recency_decay ** (self._step - t.step)
            rel = 1.0 if query_vec is None else 0.5 + 0.5 * float(t.vec @ query_vec)
            out.append(rec * rel)
        return out

    def context(self, query: str | None = None) -> list[Turn]:
        """Return turns in chronological order after attention-weighted eviction."""
        qv = self.embedder.embed([query])[0] if query else None
        keep = list(range(len(self.turns)))
        w = self.weights(qv)

        def over_budget(idx: list[int]) -> bool:
            return (len(idx) > self.max_turns or
                    sum(_ntok(self.turns[i].text) for i in idx) > self.max_tokens)

        while keep and over_budget(keep):
            victim = min(keep, key=lambda i: w[i])
            keep.remove(victim)
        return [self.turns[i] for i in keep]

    def render(self, query: str | None = None) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.context(query))

    def clear(self) -> None:
        self.turns.clear()


# --------------------------------------------------------------------- LTM
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge TEXT NOT NULL,
    sources TEXT NOT NULL,
    reliability REAL NOT NULL,
    timestamp TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    domain TEXT NOT NULL DEFAULT '',
    quality REAL NOT NULL DEFAULT 0,
    stability REAL NOT NULL DEFAULT 0.5,
    learning_score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ltm',
    created_at TEXT NOT NULL,
    last_access TEXT NOT NULL,
    feedback REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status ON memory(status);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class LongTermMemory:
    def __init__(self, db_path: str, embedder: Embedder, decay_half_life_days: float = 30.0):
        self.embedder = embedder
        self.half_life = decay_half_life_days
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.index = VectorIndex(embedder.dim, str(Path(db_path).with_suffix(".index")))
        if len(self.index) == 0:
            self._rebuild_index()

    # ---- persistence helpers
    def _row_to_item(self, r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"], knowledge=r["knowledge"], sources=json.loads(r["sources"]),
            reliability=r["reliability"], timestamp=_dt(r["timestamp"]),
            access_count=r["access_count"], domain=r["domain"], quality=r["quality"],
            stability=r["stability"], learning_score=r["learning_score"], status=r["status"],
            created_at=_dt(r["created_at"]), last_access=_dt(r["last_access"]),
            feedback=r["feedback"],
        )

    def _rebuild_index(self) -> None:
        rows = self.db.execute("SELECT id, knowledge FROM memory").fetchall()
        if rows:
            self.index.add([r["id"] for r in rows], self.embedder.embed([r["knowledge"] for r in rows]))
            self.index.save()

    def log(self, kind: str, payload: dict) -> None:
        self.db.execute("INSERT INTO events(at, kind, payload) VALUES (?,?,?)",
                        (_iso(utcnow()), kind, json.dumps(payload, default=str)))
        self.db.commit()

    # ---- CRUD
    def add(self, item: MemoryItem, vec: np.ndarray | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO memory(knowledge, sources, reliability, timestamp, access_count,
               domain, quality, stability, learning_score, status, created_at, last_access, feedback)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (item.knowledge, json.dumps(item.sources), item.reliability, _iso(item.timestamp),
             item.access_count, item.domain, item.quality, item.stability, item.learning_score,
             item.status, _iso(item.created_at), _iso(item.last_access), item.feedback))
        self.db.commit()
        item.id = int(cur.lastrowid)
        if vec is None:
            vec = self.embedder.embed([item.knowledge])[0]
        self.index.add([item.id], vec[None, :])
        self.index.save()
        return item.id

    def get(self, item_id: int) -> MemoryItem | None:
        r = self.db.execute("SELECT * FROM memory WHERE id=?", (item_id,)).fetchone()
        return self._row_to_item(r) if r else None

    def update(self, item: MemoryItem) -> None:
        self.db.execute(
            """UPDATE memory SET knowledge=?, sources=?, reliability=?, timestamp=?, access_count=?,
               domain=?, quality=?, stability=?, learning_score=?, status=?, last_access=?, feedback=?
               WHERE id=?""",
            (item.knowledge, json.dumps(item.sources), item.reliability, _iso(item.timestamp),
             item.access_count, item.domain, item.quality, item.stability, item.learning_score,
             item.status, _iso(item.last_access), item.feedback, item.id))
        self.db.commit()

    def delete(self, item_id: int) -> None:
        self.db.execute("DELETE FROM memory WHERE id=?", (item_id,))
        self.db.commit()
        self.index.remove([item_id])
        self.index.save()

    def all(self, status: str | None = None) -> list[MemoryItem]:
        if status:
            rows = self.db.execute("SELECT * FROM memory WHERE status=?", (status,)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM memory").fetchall()
        return [self._row_to_item(r) for r in rows]

    def __len__(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    # ---- retrieval
    def decay_factor(self, item: MemoryItem, now: datetime | None = None) -> float:
        now = now or utcnow()
        days = max(0.0, (now - item.last_access).total_seconds() / 86400)
        return 0.5 ** (days / self.half_life)

    def search(self, query: str | np.ndarray, k: int = 5, touch: bool = True,
               min_sim: float = 0.0) -> list[tuple[MemoryItem, float]]:
        """Return (item, score) with score = cosine x decay-weighted reliability."""
        qv = self.embedder.embed([query])[0] if isinstance(query, str) else query
        hits = self.index.search(qv, k=k * 3)
        out: list[tuple[MemoryItem, float, float]] = []
        for item_id, sim in hits:
            if sim < min_sim:
                continue
            item = self.get(item_id)
            if item is None:
                continue
            weight = 0.7 + 0.3 * item.reliability * self.decay_factor(item)
            out.append((item, sim, sim * weight))
        out.sort(key=lambda x: -x[2])
        out = out[:k]
        if touch:
            now = utcnow()
            for item, _, _ in out:
                item.access_count += 1
                item.last_access = now
                self.update(item)
        return [(item, sim) for item, sim, _ in out]

    def nearest_similarity(self, vec: np.ndarray) -> float:
        hits = self.index.search(vec, k=1)
        return hits[0][1] if hits else 0.0

    # ---- maintenance
    def prune(self, min_strength: float = 0.05) -> int:
        """Evict stagnant items whose decayed strength falls below `min_strength`.
        Learned/queued items are kept (they are the replay buffer)."""
        removed = 0
        for item in self.all(status="ltm"):
            strength = item.reliability * self.decay_factor(item) * (1 + math.log1p(item.access_count))
            if strength < min_strength:
                self.delete(item.id)
                removed += 1
        return removed

    def stats(self) -> dict:
        rows = self.db.execute("SELECT status, COUNT(*) c FROM memory GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows} | {"total": len(self)}
