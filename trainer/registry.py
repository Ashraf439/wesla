"""Column registry: one PNN column == one LoRA adapter over a frozen base model.

Written by the trainer, read by the live app (wesla/adapters.py). Pure Python.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Column:
    id: str
    name: str
    centroid: list[float]
    n_items: int = 0
    version: int = 0                     # 0 = created but never trained successfully
    adapter_dir: str = ""                # local path of the current accepted adapter
    ollama_model: str = ""
    hf_repo: str = ""
    parents: list[str] = field(default_factory=list)   # lateral-transfer sources
    parent_weights: list[float] = field(default_factory=list)
    rank: int = 0                        # total LoRA rank (frozen lateral blocks + own block)
    frozen_rank: int = 0                 # leading rank rows copied from parents and kept frozen
    trained_ids: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # per-version metrics

    def centroid_vec(self) -> np.ndarray:
        v = np.asarray(self.centroid, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)


@dataclass
class Registry:
    embedder: str = "BAAI/bge-small-en-v1.5"
    base_model: str = "unsloth/Llama-3.2-1B-Instruct"
    theta_domain: float = 0.55
    columns: list[Column] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        cols = [Column(**c) for c in d.pop("columns", [])]
        known = {k: v for k, v in d.items() if k in {"embedder", "base_model", "theta_domain"}}
        return cls(columns=cols, **known)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1))

    def get(self, cid: str) -> Column:
        return next(c for c in self.columns if c.id == cid)

    def new_id(self) -> str:
        return f"col_{len(self.columns):03d}"

    def trained(self) -> list[Column]:
        return [c for c in self.columns if c.version > 0]

    def all_trained_ids(self) -> set[int]:
        return {i for c in self.columns for i in c.trained_ids}
