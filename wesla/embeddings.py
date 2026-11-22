"""Text embeddings.

`SentenceTransformerEmbedder` is the real backend (bge-small / MiniLM on CPU).
`HashingEmbedder` is a dependency-free fallback (feature hashing of word and
character n-grams) so the pipeline and tests run without model downloads.
All embedders return L2-normalised float32 vectors, so dot product == cosine.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .log import get_logger

log = get_logger("embeddings")

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, h % self.dim] += sign
        return _normalise(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def make_embedder(backend: str, model: str, dim: int) -> Embedder:
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(model)
        except Exception as exc:  # missing package or offline
            log.warning("sentence-transformers unavailable (%s); using hashing embedder", exc)
    return HashingEmbedder(dim)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a (n,d) and b (m,d) -> (n,m)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return _normalise(a) @ _normalise(b).T
