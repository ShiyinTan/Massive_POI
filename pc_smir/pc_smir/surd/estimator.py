"""Named discrete SURD / MMI-PID estimator for predictive atoms.

Estimator name: ``mmi_pid_discrete_v1``

Uses a non-negative Partial Information Decomposition with the
minimum-mutual-information (MMI) redundancy heuristic on discrete
variables and a binary target, with Dirichlet-α smoothed empirical
distributions.

For sources Qi, Qj and binary Y:

  I_i  = I(Qi; Y)
  I_j  = I(Qj; Y)
  I_ij = I((Qi, Qj); Y)
  R_ij = min(I_i, I_j)
  U_i  = I_i - R_ij
  U_j  = I_j - R_ij
  S_ij = I_ij - I_i - I_j + R_ij

Single-source unique atoms reported in the multi-source profile use
marginal MI I(Qi; Y) as a relevance cue (heuristic mapping; not a
causal unique-information claim). Pair synergy maps only to S_ij.
Redundancy R_ij is stored for diagnostics and is NEVER fed to
interaction experts as synergy.

Synthetic qualitative checks (see tests):
  1) Y = Q1           -> unique Q1 high
  2) Y = XOR(Q1, Q2)  -> synergy high
  3) Q2 = Q1, Y = Q1  -> redundancy high
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

ESTIMATOR_NAME = "mmi_pid_discrete_v1"
ESTIMATOR_VERSION = "1.0.0"


def _joint_counts(arrays: list[np.ndarray], n_states: list[int], weights: np.ndarray | None = None) -> np.ndarray:
    n = len(arrays[0])
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    idx = np.zeros(n, dtype=np.int64)
    mul = 1
    for arr, ns in zip(reversed(arrays), reversed(n_states)):
        idx = idx + arr.astype(np.int64) * mul
        mul *= ns
    flat = np.bincount(idx, weights=weights, minlength=mul).astype(np.float64)
    return flat.reshape(n_states)


def _entropy(p: np.ndarray) -> float:
    p = p.ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mi_from_counts(counts: np.ndarray, alpha: float) -> float:
    """I(X; Y) for last axis = Y (binary or categorical)."""
    smoothed = counts + alpha
    joint = smoothed / smoothed.sum()
    px = joint.sum(axis=-1, keepdims=False)
    # Flatten X axes.
    y_dim = joint.shape[-1]
    joint_xy = joint.reshape(-1, y_dim)
    px = px.reshape(-1)
    py = joint_xy.sum(axis=0)
    hx = _entropy(px)
    hy = _entropy(py)
    hxy = _entropy(joint_xy)
    return max(0.0, hx + hy - hxy)


def mutual_information(
    x: np.ndarray | list[np.ndarray],
    y: np.ndarray,
    x_card: int | list[int],
    y_card: int = 2,
    alpha: float = 0.5,
    weights: np.ndarray | None = None,
) -> float:
    if isinstance(x, np.ndarray) and x.ndim == 1:
        xs = [x]
        cards = [int(x_card) if isinstance(x_card, int) else int(x_card[0])]
    else:
        xs = list(x)
        cards = list(x_card) if not isinstance(x_card, int) else [x_card] * len(xs)
    ys = y.astype(np.int64)
    counts = _joint_counts(xs + [ys], cards + [y_card], weights=weights)
    return _mi_from_counts(counts, alpha=alpha)


def estimate_pair_pid(
    q_i: np.ndarray,
    q_j: np.ndarray,
    y: np.ndarray,
    card_i: int,
    card_j: int,
    alpha: float = 0.5,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    y = y.astype(np.int64)
    i_i = mutual_information(q_i, y, card_i, alpha=alpha, weights=weights)
    i_j = mutual_information(q_j, y, card_j, alpha=alpha, weights=weights)
    i_ij = mutual_information([q_i, q_j], y, [card_i, card_j], alpha=alpha, weights=weights)
    redundancy = min(i_i, i_j)
    unique_i = max(0.0, i_i - redundancy)
    unique_j = max(0.0, i_j - redundancy)
    synergy = max(0.0, i_ij - i_i - i_j + redundancy)
    return {
        "unique_i": unique_i,
        "unique_j": unique_j,
        "redundancy": redundancy,
        "synergy": synergy,
        "mi_i": i_i,
        "mi_j": i_j,
        "mi_ij": i_ij,
    }


def estimate_triple_synergy(
    q_i: np.ndarray,
    q_j: np.ndarray,
    q_k: np.ndarray,
    y: np.ndarray,
    cards: Sequence[int],
    alpha: float = 0.5,
    weights: np.ndarray | None = None,
) -> float:
    """Higher-order synergy heuristic: joint MI beyond best pair MI."""
    y = y.astype(np.int64)
    i_ijk = mutual_information([q_i, q_j, q_k], y, list(cards), alpha=alpha, weights=weights)
    pair_mis = [
        mutual_information([q_i, q_j], y, [cards[0], cards[1]], alpha=alpha, weights=weights),
        mutual_information([q_i, q_k], y, [cards[0], cards[2]], alpha=alpha, weights=weights),
        mutual_information([q_j, q_k], y, [cards[1], cards[2]], alpha=alpha, weights=weights),
    ]
    singles = [
        mutual_information(q_i, y, cards[0], alpha=alpha, weights=weights),
        mutual_information(q_j, y, cards[1], alpha=alpha, weights=weights),
        mutual_information(q_k, y, cards[2], alpha=alpha, weights=weights),
    ]
    # Residual above the strongest lower-order explanatory MI.
    return max(0.0, i_ijk - max(pair_mis + singles))


def estimate_surd(
    y_binary: np.ndarray,
    source_states: dict[str, np.ndarray],
    source_names: Sequence[str],
    config: dict | None = None,
) -> dict:
    """Adapter API required by the implementation spec.

    Returns unique / synergy / redundancy atoms plus metadata.
    """
    cfg = {
        "alpha": 0.5,
        "pairs": None,
        "triples": None,
        "estimator": ESTIMATOR_NAME,
        "version": ESTIMATOR_VERSION,
    }
    if config:
        cfg.update(config)

    names = list(source_names)
    for n in names:
        if n not in source_states:
            raise KeyError(f"Missing source state for {n}")
    y = np.asarray(y_binary).astype(np.int64)
    if y.ndim != 1 or set(np.unique(y)).difference({0, 1}):
        raise ValueError("y_binary must be a 1-D binary array")

    states = {n: np.asarray(source_states[n]).astype(np.int64) for n in names}
    cards = {n: int(states[n].max()) + 1 if len(states[n]) else 1 for n in names}
    # Ensure non-negative labels.
    for n in names:
        if states[n].min() < 0:
            raise ValueError(f"Source {n} has negative state ids")

    alpha = float(cfg["alpha"])
    unique = {n: mutual_information(states[n], y, cards[n], alpha=alpha) for n in names}

    pair_list = cfg["pairs"] or list(combinations(names, 2))
    triple_list = cfg["triples"] or []

    synergy: dict[tuple, float] = {}
    redundancy: dict[tuple, float] = {}
    for a, b in pair_list:
        pid = estimate_pair_pid(states[a], states[b], y, cards[a], cards[b], alpha=alpha)
        synergy[(a, b)] = pid["synergy"]
        redundancy[(a, b)] = pid["redundancy"]

    for trip in triple_list:
        a, b, c = trip
        synergy[(a, b, c)] = estimate_triple_synergy(
            states[a], states[b], states[c], y, [cards[a], cards[b], cards[c]], alpha=alpha
        )

    return {
        "unique": unique,
        "synergy": {k: float(v) for k, v in synergy.items()},
        "redundancy": {k: float(v) for k, v in redundancy.items()},
        "metadata": {
            "estimator": ESTIMATOR_NAME,
            "version": ESTIMATOR_VERSION,
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "alpha": alpha,
            "cards": cards,
            "identifiable_atoms": {
                "unique": names,
                "synergy_pairs": [list(p) for p in pair_list],
                "synergy_triples": [list(t) for t in triple_list],
                "redundancy_pairs": [list(p) for p in pair_list],
            },
        },
    }


def atom_key(parts: Iterable[str]) -> str:
    return "_".join(parts)