"""Semantic / near-duplicate removal with MinHash LSH (proposal §4.B).

Uses `datasketch` when installed; otherwise a NumPy MinHash + banded LSH with
the same behaviour. The `Deduplicator` is stateful, so it can also drop chunks
that duplicate text already seen in earlier queries.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .types import Chunk

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_TOKEN = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = 5) -> set[str]:
    toks = _TOKEN.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _h32(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], "little")


class NumpyMinHash:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**31 - 1, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.b = rng.randint(0, 2**31 - 1, size=num_perm, dtype=np.int64).astype(np.uint64)
        self.num_perm = num_perm

    def signature(self, sh: set[str]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        hv = np.array([_h32(s) for s in sh], dtype=np.uint64)[:, None]
        phv = ((hv * self.a + self.b) % _MERSENNE) & _MAX_HASH
        return phv.min(axis=0)


def _optimal_bands(threshold: float, num_perm: int) -> tuple[int, int]:
    """Pick (b, r) with b*r <= num_perm whose S-curve threshold (1/b)^(1/r) is closest."""
    best, best_err = (num_perm, 1), 1e9
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b == 0:
            break
        err = abs((1 / b) ** (1 / r) - threshold)
        if err < best_err:
            best, best_err = (b, r), err
    return best


class Deduplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 5):
        self.threshold = threshold
        self.k = shingle_size
        self.num_perm = num_perm
        self._seen: dict[str, np.ndarray] = {}
        try:
            from datasketch import MinHash, MinHashLSH  # type: ignore

            self._ds = (MinHash, MinHashLSH(threshold=threshold, num_perm=num_perm))
        except Exception:
            self._ds = None
            self._mh = NumpyMinHash(num_perm)
            self.bands, self.rows = _optimal_bands(threshold, num_perm)
            self._buckets: dict[tuple[int, bytes], list[str]] = {}

    # ---- numpy path
    def _np_candidates(self, sig: np.ndarray) -> set[str]:
        cands: set[str] = set()
        for b in range(self.bands):
            key = (b, sig[b * self.rows:(b + 1) * self.rows].tobytes())
            cands.update(self._buckets.get(key, []))
        return cands

    def _np_insert(self, key: str, sig: np.ndarray) -> None:
        self._seen[key] = sig
        for b in range(self.bands):
            k = (b, sig[b * self.rows:(b + 1) * self.rows].tobytes())
            self._buckets.setdefault(k, []).append(key)

    def find_duplicate_and_add(self, key: str, text: str) -> str | None:
        """Return the key of an already-seen near-duplicate, or None (and index this text)."""
        sh = shingles(text, self.k)
        if self._ds is not None:
            MinHash, lsh = self._ds
            m = MinHash(num_perm=self.num_perm)
            for s in sh:
                m.update(s.encode())
            for c in lsh.query(m):
                if m.jaccard(self._seen[c]) >= self.threshold:
                    return c
            if key not in self._seen:
                lsh.insert(key, m)
                self._seen[key] = m
            return None
        sig = self._mh.signature(sh)
        for c in self._np_candidates(sig):
            if float(np.mean(sig == self._seen[c])) >= self.threshold:
                return c
        if key not in self._seen:
            self._np_insert(key, sig)
        return None

    def is_duplicate_and_add(self, key: str, text: str) -> bool:
        return self.find_duplicate_and_add(key, text) is not None

    def filter(self, chunks: list[Chunk]) -> tuple[list[Chunk], int]:
        """Drop near-duplicates. The kept representative remembers the dropped
        copies' URLs in `mirrors`, so verification can still count them as sources."""
        kept: dict[str, Chunk] = {}
        dropped = 0
        for c in chunks:
            key = c.chunk_id or c.text[:40]
            dup_of = self.find_duplicate_and_add(key, c.text)
            if dup_of is not None:
                dropped += 1
                if dup_of in kept and c.url not in kept[dup_of].mirrors and c.url != kept[dup_of].url:
                    kept[dup_of].mirrors.append(c.url)
            else:
                kept[key] = c
        return list(kept.values()), dropped
