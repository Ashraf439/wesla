"""Adaptive PNN domain clustering (proposal §4.G). Pure NumPy.

For each queued item embedding x:
  s* = max_j cos(x, c_j) over existing column centroids
  if s* >= theta_domain  -> reuse column j (update its adapter)
  else                   -> leftover
Leftovers are grouped by leader clustering at the same theta. A leftover group
becomes a NEW column only if it has >= min_new_items members (and the column
cap allows); smaller groups stay in the queue until more evidence arrives.
When the cap is reached, leftovers are forced to their nearest column.

Lateral transfer: a new column is initialised from its most similar existing
columns, weighted by softmax(similarity / T) - the LoRA analogue of PNN lateral
connections to frozen earlier columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .registry import Column, Registry


@dataclass
class Assignment:
    reuse: dict[str, list[int]] = field(default_factory=dict)        # column id -> item indices
    new: list[list[int]] = field(default_factory=list)                # groups of item indices
    deferred: list[int] = field(default_factory=list)                 # too few to justify a column
    similarity: dict[int, float] = field(default_factory=dict)        # item index -> s*


def _norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1
    return m / n


def leader_cluster(vecs: np.ndarray, theta: float) -> list[list[int]]:
    """Greedy single-pass clustering: join the nearest group whose centroid sim >= theta."""
    groups: list[list[int]] = []
    cents: list[np.ndarray] = []
    for i, v in enumerate(vecs):
        if cents:
            sims = np.array([c @ v for c in cents])
            j = int(np.argmax(sims))
            if sims[j] >= theta:
                groups[j].append(i)
                cents[j] = _norm(vecs[groups[j]].mean(axis=0))
                continue
        groups.append([i])
        cents.append(v.copy())
    return groups


def assign(reg: Registry, vecs: np.ndarray, theta: float | None = None, min_new_items: int = 8,
           max_columns: int = 3) -> Assignment:
    theta = reg.theta_domain if theta is None else theta
    vecs = _norm(np.asarray(vecs, dtype=np.float32))
    out = Assignment()
    leftovers: list[int] = []
    cols = reg.columns
    cmat = np.stack([c.centroid_vec() for c in cols]) if cols else np.zeros((0, vecs.shape[1]))
    for i, v in enumerate(vecs):
        if len(cols):
            sims = cmat @ v
            j = int(np.argmax(sims))
            out.similarity[i] = float(sims[j])
            if sims[j] >= theta:
                out.reuse.setdefault(cols[j].id, []).append(i)
                continue
        leftovers.append(i)

    if leftovers:
        groups = leader_cluster(vecs[leftovers], theta)
        groups = [[leftovers[k] for k in g] for g in groups]
        groups.sort(key=len, reverse=True)
        room = max(0, max_columns - len(cols))
        for g in groups:
            if len(g) >= min_new_items and room > 0:
                out.new.append(g)
                room -= 1
            elif room == 0 and len(cols):
                for i in g:  # cap reached: force into the nearest existing column
                    j = int(np.argmax(cmat @ vecs[i]))
                    out.reuse.setdefault(cols[j].id, []).append(i)
            else:
                out.deferred.extend(g)
    return out


def update_centroid(col: Column, new_vecs: np.ndarray) -> None:
    """Running mean of all items the column has absorbed."""
    old = np.asarray(col.centroid, dtype=np.float32) * max(col.n_items, 0)
    tot = old + np.asarray(new_vecs, dtype=np.float32).sum(axis=0)
    n = col.n_items + len(new_vecs)
    col.centroid = [float(x) for x in _norm(tot / max(n, 1))]
    col.n_items = n


def lateral_parents(reg: Registry, centroid: np.ndarray, k: int = 2, temperature: float = 0.1,
                    min_sim: float = 0.2) -> tuple[list[str], list[float]]:
    trained = reg.trained()
    if not trained:
        return [], []
    sims = np.array([c.centroid_vec() @ _norm(centroid) for c in trained])
    order = np.argsort(-sims)[:k]
    order = [i for i in order if sims[i] >= min_sim]
    if not order:
        return [], []
    logits = sims[order] / temperature
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return [trained[i].id for i in order], [float(x) for x in w]
