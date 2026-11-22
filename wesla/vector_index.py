"""Approximate nearest-neighbour index over normalised embeddings.

Uses FAISS (IndexFlatIP wrapped in IndexIDMap2) when installed, otherwise an
exact NumPy matrix search. Both expose the same add / remove / search API and
persist to disk.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("wesla.vector_index")

# On Windows, faiss-cpu and PyTorch each ship their own OpenMP runtime; loading both can
# abort the whole process ("OMP: Error #15"). Use the NumPy index there unless asked.
_want_faiss = os.environ.get("WESLA_USE_FAISS", "0" if sys.platform == "win32" else "1") == "1"
faiss = None
if _want_faiss:
    try:  # pragma: no cover - depends on environment
        import faiss  # type: ignore
    except Exception:  # pragma: no cover
        faiss = None


class VectorIndex:
    def __init__(self, dim: int, path: str | None = None):
        log.debug("vector index backend: %s", "faiss" if faiss is not None else "numpy")
        self.dim = dim
        self.path = Path(path) if path else None
        self._ids = np.zeros((0,), dtype=np.int64)
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        if self.path and self.path.with_suffix(".npz").exists():
            self.load()

    def __len__(self) -> int:
        return int(self._ids.shape[0])

    def add(self, ids: list[int], vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vecs = np.asarray(vecs, dtype=np.float32).reshape(len(ids), self.dim)
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.remove(ids)  # upsert semantics
        self._ids = np.concatenate([self._ids, ids_arr])
        self._vecs = np.vstack([self._vecs, vecs])
        if self._faiss is not None:
            self._faiss.add_with_ids(vecs, ids_arr)

    def remove(self, ids: list[int]) -> None:
        if len(self) == 0 or len(ids) == 0:
            return
        mask = ~np.isin(self._ids, np.asarray(ids, dtype=np.int64))
        self._ids, self._vecs = self._ids[mask], self._vecs[mask]
        if self._faiss is not None:
            self._faiss.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, self.dim)
        k = min(k, len(self))
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = self._vecs @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(self._ids[i]), float(sims[i])) for i in top]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path.with_suffix(".npz"), ids=self._ids, vecs=self._vecs)

    def load(self) -> None:
        data = np.load(self.path.with_suffix(".npz"))
        self._ids = np.zeros((0,), dtype=np.int64)
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        if self._faiss is not None:
            self._faiss.reset()
        if data["ids"].size:
            self.add(list(data["ids"]), data["vecs"])
