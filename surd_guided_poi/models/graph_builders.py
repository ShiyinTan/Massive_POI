"""Count-based POI transition / user–POI interact graphs for Graph-Flashback.

When Massive-STEPS KGE ``.pkl`` graphs are unavailable, build sparse adjacency
from official-train ``NextPOISample`` sequences (same shapes Graph-Flashback
expects: transition ``[K,K]``, interact ``[U,K]``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..data.preprocessing import NextPOISample


def build_transition_graph_from_samples(
    samples: Iterable[NextPOISample],
    num_pois: int,
) -> coo_matrix:
    """Directed POI→POI counts from history prefixes + target."""
    rows, cols, data = [], [], []
    for sample in samples:
        seq = list(sample.history_poi_ids) + [int(sample.target_poi_id)]
        for a, b in zip(seq[:-1], seq[1:]):
            a_i, b_i = int(a), int(b)
            if 0 <= a_i < num_pois and 0 <= b_i < num_pois:
                rows.append(a_i)
                cols.append(b_i)
                data.append(1.0)
    if not rows:
        return coo_matrix((num_pois, num_pois), dtype=np.float32)
    mat = coo_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(num_pois, num_pois))
    return coo_matrix(mat)  # sum duplicates


def build_interact_graph_from_samples(
    samples: Iterable[NextPOISample],
    num_users: int,
    num_pois: int,
) -> csr_matrix:
    """User–POI visit counts ``[U, K]`` (Graph-Flashback interact graph)."""
    rows, cols, data = [], [], []
    for sample in samples:
        u = int(sample.user_id)
        if not (0 <= u < num_users):
            continue
        for p in list(sample.history_poi_ids) + [int(sample.target_poi_id)]:
            p_i = int(p)
            if 0 <= p_i < num_pois:
                rows.append(u)
                cols.append(p_i)
                data.append(1.0)
    if not rows:
        return csr_matrix((num_users, num_pois), dtype=np.float32)
    mat = coo_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(num_users, num_pois))
    return csr_matrix(mat)


def load_sparse_graph_pkl(path: Path, *, prefer_csr: bool = False):
    """Load a Graph-Flashback / KGE pickle graph."""
    with open(path, "rb") as f:
        graph = pickle.load(f)
    if prefer_csr:
        return csr_matrix(graph)
    return coo_matrix(graph)


def resolve_graph_flashback_graphs(
    *,
    num_pois: int,
    num_users: int,
    train_samples: Optional[Iterable[NextPOISample]] = None,
    transition_pkl: Optional[str | Path] = None,
    interact_pkl: Optional[str | Path] = None,
) -> Tuple[coo_matrix, csr_matrix]:
    """Return ``(transition [K,K], interact [U,K])`` from pkl paths or train counts."""
    if transition_pkl:
        transition = load_sparse_graph_pkl(Path(transition_pkl))
        if transition.shape != (num_pois, num_pois):
            raise ValueError(
                f"transition graph shape {transition.shape} != ({num_pois}, {num_pois})"
            )
    else:
        if train_samples is None:
            raise ValueError("graph_flashback needs train_samples or model.transition_pkl")
        transition = build_transition_graph_from_samples(train_samples, num_pois)

    if interact_pkl:
        interact = load_sparse_graph_pkl(Path(interact_pkl), prefer_csr=True)
        if interact.shape != (num_users, num_pois):
            raise ValueError(
                f"interact graph shape {interact.shape} != ({num_users}, {num_pois})"
            )
    else:
        if train_samples is None:
            raise ValueError("graph_flashback needs train_samples or model.interact_pkl")
        interact = build_interact_graph_from_samples(train_samples, num_users, num_pois)

    return transition, interact
