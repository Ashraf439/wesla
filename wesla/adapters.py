"""Registry of trained PNN columns (= LoRA adapters) as seen by the live app.

The offline trainer writes `columns.json`:
{
  "embedder": "BAAI/bge-small-en-v1.5",
  "theta_domain": 0.55,
  "columns": [
    {"id": "col_000", "name": "tech-news", "centroid": [...], "ollama_model": "wesla-col_000",
     "hf_repo": "user/wesla-col_000", "parents": [], "n_items": 120, "version": 3}
  ]
}
At query time the app embeds the query, picks the nearest column (if
similarity >= theta_domain) and generates with that column's Ollama model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .log import get_logger

log = get_logger("adapters")


@dataclass
class ColumnInfo:
    id: str
    name: str
    centroid: np.ndarray
    ollama_model: str
    parents: list[str]
    version: int = 1


class AdapterRegistry:
    def __init__(self, path: str, embedder: Embedder, base_model: str):
        self.path = Path(path)
        self.embedder = embedder
        self.base_model = base_model
        self.theta = 0.55
        self.columns: list[ColumnInfo] = []
        self.reload()

    def reload(self) -> None:
        self.columns = []
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.theta = float(data.get("theta_domain", self.theta))
        for c in data.get("columns", []):
            cen = np.asarray(c["centroid"], dtype=np.float32)
            if cen.shape[0] != self.embedder.dim:
                log.warning("column %s centroid dim %d != embedder dim %d; skipped "
                            "(use the same embedder as the trainer)", c["id"], cen.shape[0], self.embedder.dim)
                continue
            self.columns.append(ColumnInfo(c["id"], c.get("name", c["id"]), cen / (np.linalg.norm(cen) or 1),
                                           c.get("ollama_model", self.base_model), c.get("parents", []),
                                           int(c.get("version", 1))))

    def select(self, query: str) -> tuple[ColumnInfo | None, float]:
        if not self.columns:
            return None, 0.0
        q = self.embedder.embed([query])[0]
        sims = np.array([c.centroid @ q for c in self.columns])
        i = int(np.argmax(sims))
        return (self.columns[i], float(sims[i])) if sims[i] >= self.theta else (None, float(sims[i]))

    def model_for(self, query: str) -> tuple[str, str]:
        col, sim = self.select(query)
        if col is None:
            return self.base_model, f"base model (best column sim {sim:.2f} < {self.theta})"
        return col.ollama_model, f"column {col.id} '{col.name}' (sim {sim:.2f})"
