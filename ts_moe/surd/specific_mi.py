"""Specific mutual information and the routing-scale normalization.

Specific MI of target state ``y`` (spec §4.2), estimated on samples whose
next POI is ``y``::

    i(y; Q_A) = E[ log P(Y=y | Q_A) - log P(Y=y) | Y=y ]

The neural conditional estimator and the smoothed discrete estimator both
reduce to this average. Monotone projection matches ALD-Lab ``surd``: a
superset cannot carry less specific information than one of its subsets.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .atoms import ALL_SUBSETS, subset_key


def aggregate_specific_mi(
    log_ratio: np.ndarray,
    labels: np.ndarray,
    n_pois: int,
) -> np.ndarray:
    """Average ``log P(y|Q) - log P(y)`` over samples with target ``y``.

    POIs that never appear in ``labels`` stay 0. Index 0 (padding) stays 0.
    """
    labels = np.asarray(labels, dtype=np.int64)
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    sums = np.bincount(labels, weights=log_ratio, minlength=n_pois)
    counts = np.bincount(labels, minlength=n_pois)
    mi = np.zeros(n_pois, dtype=np.float64)
    seen = counts > 0
    mi[seen] = sums[seen] / counts[seen]
    mi[0] = 0.0
    return mi


def marginal_prior(labels: np.ndarray, n_pois: int, eps: float = 1e-3) -> np.ndarray:
    """Smoothed ``P(Y=y)`` on real POI ids. Padding slot stays a tiny constant."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_pois).astype(np.float64)
    counts[0] = 0.0
    prior = (counts + eps) / (counts.sum() + eps * max(n_pois - 1, 1))
    prior[0] = eps
    return prior


def enforce_subset_monotonicity(
    specific_mi: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Project ``i(y; Q_A) = max(0, max_{B ⊆ A} i(y; Q_B))`` with ``i(empty)=0``."""
    running: dict[str, np.ndarray] = {}
    for subset in sorted(ALL_SUBSETS, key=len):
        key = subset_key(subset)
        if key not in specific_mi:
            raise KeyError(f"missing specific MI for {key}")
        value = np.maximum(np.asarray(specific_mi[key], dtype=np.float64), 0.0)
        members = set(subset)
        for smaller_key, smaller_value in running.items():
            if set(smaller_key.split("_")) < members:
                value = np.maximum(value, smaller_value)
        running[key] = value
    return {subset_key(subset): running[subset_key(subset)] for subset in ALL_SUBSETS}


def routing_distribution(
    raw_mass: np.ndarray,
    counts: np.ndarray,
    kappa: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Turn raw expert masses into the cached per-POI routing prior.

    Steps, in order:

    1. Negative atoms are dropped (``max(0, ·)`` from spec §4.3).
    2. A POI whose positive mass is 0 is treated as *no expert preference*
       (uniform). Using ``0 / (0+ε) = 0`` and then normalizing would turn
       "no information" into whatever vector happens to be nonzero.
    3. Long-tail shrink (spec §11.2)::

           c_y = n_y / (n_y + κ)
           R_y = c_y R_y + (1 - c_y) R_global

       ``n_y`` is the number of samples that entered the specific-MI average.
       ``R_global`` is the count-weighted mean of those per-POI distributions.
       Unseen POIs have ``c_y = 0`` and use ``R_global``.
    4. The mixture is renormalized. That vector is both the router feature
       ``r_y`` and the KL target ``π*`` (spec §8.3 ``Normalize``).
    """
    raw = np.asarray(raw_mass, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    n_pois, n_exp = raw.shape
    positive = np.maximum(raw, 0.0)
    denom = positive.sum(axis=-1, keepdims=True)
    own = np.full_like(positive, 1.0 / n_exp)
    seen_mass = denom.squeeze(-1) > 0
    own[seen_mass] = positive[seen_mass] / denom[seen_mass]

    support = counts > 0
    if not np.any(support):
        global_prior = np.full(n_exp, 1.0 / n_exp)
    else:
        weight = counts.copy()
        weight[~support] = 0.0
        global_prior = (own * weight[:, None]).sum(axis=0) / weight.sum()
        gsum = global_prior.sum()
        global_prior = global_prior / gsum if gsum > 0 else np.full(n_exp, 1.0 / n_exp)

    confidence = counts / (counts + float(kappa))
    mixed = confidence[:, None] * own + (1.0 - confidence[:, None]) * global_prior
    mixed = np.maximum(mixed, 0.0)
    mixed_sum = mixed.sum(axis=-1, keepdims=True)
    mixed = mixed / np.maximum(mixed_sum, eps)
    mixed[0] = global_prior  # padding row is never a candidate; keep it usable
    return mixed


def observation_counts(labels: np.ndarray, n_pois: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_pois).astype(np.float64)
    counts[0] = 0.0
    return counts
