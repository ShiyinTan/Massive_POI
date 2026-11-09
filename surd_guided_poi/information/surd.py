"""Official SURD assignment of specific mutual information into R / U / S.

Algorithm source (adapted for POI-specific / per-state outputs):
  Martínez-Sánchez, Arranz & Lozano-Durán, Nature Communications 15, 9296 (2024)
  Official package: https://github.com/ALD-Lab/SURD  (`surd.assign`)

We keep the sorting, monotone clipping of supersets, differencing, and
redundancy-bookkeeping rules of ``assign``. Two intentional adaptations:

1. Per-POI (specific) contributions are returned *without* multiplying by
   P(Y=y) inside the loop, so each target state keeps its own R/U/S.
2. Global SURD is obtained afterwards by population-weighted aggregation
   ``sum_y P(y) * specific_term(y)`` as required by the project spec.

Observational language: these are predictive information contributions,
not causal effects.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..models.subset_estimators import SOURCE_NAMES, SUBSET_NAMES, subset_key


def _order(values: np.ndarray, sizes: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Stable ascending sort; near-ties broken by subset size then position.

    Copied from ALD-Lab/SURD ``core._order`` (same rationale: a subset and
    its superset often share equal specific MI; the subset must come first
    so information is not booked as synergy).
    """
    values = np.asarray(values, dtype=float)
    sizes = np.asarray(sizes)
    if values.size == 0:
        return np.arange(0)
    tol = rtol * max(float(np.abs(values).max()), 1.0)
    i0 = np.argsort(values, kind="stable")
    group = np.empty(values.size, dtype=np.int64)
    group[i0] = np.concatenate(([0], np.cumsum(np.diff(values[i0]) > tol)))
    return np.lexsort((np.arange(values.size), sizes, group))


def _subset_tuples(source_names: Sequence[str] = SOURCE_NAMES) -> List[Tuple[int, ...]]:
    """1-based index tuples ordered like itertools.combinations (SURD keys)."""
    n = len(source_names)
    inds = range(1, n + 1)
    return [c for k in inds for c in combinations(inds, k)]


def _name_tuple_to_index_tuple(name_tuple: Tuple[str, ...], source_names: Sequence[str]) -> Tuple[int, ...]:
    mapping = {n: i + 1 for i, n in enumerate(source_names)}
    return tuple(mapping[n] for n in name_tuple)


def assign_surd_from_specific_mi(
    specific_mi: Mapping[str, np.ndarray],
    source_names: Sequence[str] = SOURCE_NAMES,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Run official SURD assignment independently for each target POI state.

    Parameters
    ----------
    specific_mi:
        键为 subset_key（如 'Q1'、'Q1_Q2'、'Q1_Q2_Q3'）→ [num_pois] 的
        specific information（宜先做过单调投影）。

    Returns
    -------
    poi_surd[poi_id] = {
        "unique": {"Q1": ..., ...},          # I_R[(i,)] terms
        "redundant": {"Q1_Q2": ..., ...},    # I_R multi-source terms
        "synergistic": {"Q1_Q2": ..., ...},  # I_S terms
    }
    """
    source_names = tuple(source_names)
    nvars = len(source_names)
    index_combs = _subset_tuples(source_names)
    # Align specific MI arrays to index_combs order.
    name_combs = SUBSET_NAMES if source_names == SOURCE_NAMES else [
        tuple(source_names[i - 1] for i in c) for c in index_combs
    ]
    mats = []
    for names in name_combs:
        key = subset_key(names)
        if key not in specific_mi:
            raise KeyError(f"Missing specific MI for subset {key}")
        mats.append(np.asarray(specific_mi[key], dtype=np.float64))
    stacked = np.stack(mats, axis=0)  # [num_combs, K]
    K = stacked.shape[1]

    poi_surd: Dict[int, Dict[str, Dict[str, float]]] = {}
    inds = list(range(1, nvars + 1))

    for y in range(K):
        I1 = stacked[:, y].copy()
        sizes = np.array([len(c) for c in index_combs])
        i1 = _order(I1, sizes)
        lab = [index_combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])
        I1 = I1[i1]

        # Clip supersets that fall below the max of size-l subsets (official rule).
        if lens.max() >= 2:
            for l in range(1, int(lens.max())):
                inds_l2 = np.where(lens == l + 1)[0]
                if inds_l2.size == 0 or not np.any(lens == l):
                    continue
                Il1max = I1[lens == l].max()
                weak = inds_l2[I1[inds_l2] < Il1max]
                I1[weak] = 0.0

        i1 = _order(I1, lens)
        lab = [lab[i_] for i_ in i1]
        Di = np.diff(I1[i1], prepend=0.0)
        red_vars = list(inds)

        unique: Dict[str, float] = {n: 0.0 for n in source_names}
        redundant: Dict[str, float] = {}
        synergistic: Dict[str, float] = {}
        for i_, ll in enumerate(lab):
            info = float(Di[i_])  # POI-specific: do NOT multiply by P(y) here
            if len(ll) == 1:
                # Book to current redundancy pool; singleton pool => unique.
                key = tuple(red_vars)
                if len(key) == 1:
                    unique[source_names[key[0] - 1]] += info
                else:
                    rname = subset_key(tuple(source_names[j - 1] for j in key))
                    redundant[rname] = redundant.get(rname, 0.0) + info
                red_vars.remove(ll[0])
            else:
                sname = subset_key(tuple(source_names[j - 1] for j in ll))
                synergistic[sname] = synergistic.get(sname, 0.0) + info

        poi_surd[y] = {
            "unique": unique,
            "redundant": redundant,
            "synergistic": synergistic,
        }
    return poi_surd


def aggregate_global_surd(
    poi_surd: Mapping[int, Dict[str, Dict[str, float]]],
    poi_prior: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Population-weighted global SURD: sum_y P(y) * specific_term(y)."""
    prior = np.asarray(poi_prior, dtype=np.float64)
    unique = {n: 0.0 for n in SOURCE_NAMES}
    redundant: Dict[str, float] = {}
    synergistic: Dict[str, float] = {}
    for y, terms in poi_surd.items():
        w = float(prior[y]) if y < len(prior) else 0.0
        for k, v in terms["unique"].items():
            unique[k] = unique.get(k, 0.0) + w * v
        for k, v in terms["redundant"].items():
            redundant[k] = redundant.get(k, 0.0) + w * v
        for k, v in terms["synergistic"].items():
            synergistic[k] = synergistic.get(k, 0.0) + w * v
    return {"unique": unique, "redundant": redundant, "synergistic": synergistic}


def balanced_surd_summary(
    poi_surd: Mapping[int, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Macro-average over POIs that appear in poi_surd (not population MI)."""
    if not poi_surd:
        return {"unique": {}, "redundant": {}, "synergistic": {}}
    uniform = np.ones(max(poi_surd) + 1, dtype=np.float64)
    for y in range(len(uniform)):
        uniform[y] = 1.0 / len(poi_surd) if y in poi_surd else 0.0
    # Renormalize over present POIs only via explicit average.
    unique = {n: 0.0 for n in SOURCE_NAMES}
    redundant: Dict[str, float] = {}
    synergistic: Dict[str, float] = {}
    inv = 1.0 / len(poi_surd)
    for terms in poi_surd.values():
        for k, v in terms["unique"].items():
            unique[k] = unique.get(k, 0.0) + inv * v
        for k, v in terms["redundant"].items():
            redundant[k] = redundant.get(k, 0.0) + inv * v
        for k, v in terms["synergistic"].items():
            synergistic[k] = synergistic.get(k, 0.0) + inv * v
    return {"unique": unique, "redundant": redundant, "synergistic": synergistic}