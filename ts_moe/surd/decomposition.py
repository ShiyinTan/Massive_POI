"""Per-POI SURD assignment from specific mutual information.

Atom definition follows Martínez-Sánchez, Arranz, Lozano-Durán,
*Decomposing causality into its synergistic, unique, and redundant components*,
Nature Communications 15, 9296 (2024), and the reference implementation
``surd.assign`` (https://github.com/ALD-Lab/SURD).

Adaptation for TS-MoE: increments are **not** multiplied by ``P(Y=y)``.
The spec caches a prior per target POI, so each state keeps its own
unique / redundant / synergistic atoms. Population aggregation would mix
those states together and cannot route one candidate.

This is not inclusion-exclusion on the 15 subset MIs. Subsets are ordered by
specific information, weak supersets are zeroed, and each nonnegative
increment is booked either to the current redundancy pool (singleton step)
or to synergy of that subset (multi-source step).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Sequence

import numpy as np

from .atoms import SOURCES, subset_key


def _order(values: np.ndarray, sizes: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Ascending specific-MI order. Near-ties break toward the smaller subset.

    Adapted from ALD-Lab/SURD ``core._order``. A subset must be booked before
    an equal-MI superset, otherwise the shared mass is mislabeled as synergy.
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


def _subset_tuples(nvars: int) -> list[tuple[int, ...]]:
    inds = range(1, nvars + 1)
    return [comb for k in inds for comb in combinations(inds, k)]


def assign_surd_from_specific_mi(
    specific_mi: Mapping[str, np.ndarray],
    source_names: Sequence[str] = SOURCES,
) -> list[dict[str, Dict[str, float]]]:
    """Assign unique / redundant / synergistic atoms for every target state.

    Parameters
    ----------
    specific_mi:
        ``subset_key → array [K]`` of specific mutual information
        ``i(Y=y; Q_A)``. Keys cover every non-empty source subset.
        Callers should pass monotone-projected nonnegative values; this
        function still applies the official weak-superset clip.

    Returns
    -------
    Length-K list. Each item is ``{"unique", "redundant", "synergistic"}``
    mapping atom names to nonnegative masses. Sum of atoms for a state equals
    the largest subset specific MI after clipping (information conservation
    of the sorted increments).
    """
    source_names = tuple(source_names)
    nvars = len(source_names)
    index_combs = _subset_tuples(nvars)
    name_combs = [tuple(source_names[i - 1] for i in comb) for comb in index_combs]
    mats = []
    for names in name_combs:
        key = subset_key(names)
        if key not in specific_mi:
            raise KeyError(f"missing specific MI for subset {key}")
        mats.append(np.asarray(specific_mi[key], dtype=np.float64))
    stacked = np.stack(mats, axis=0)  # [n_combs, K]
    n_states = stacked.shape[1]
    inds = list(range(1, nvars + 1))
    poi_surd: list[dict[str, Dict[str, float]]] = []

    for y in range(n_states):
        info = stacked[:, y].copy()
        sizes = np.array([len(comb) for comb in index_combs])
        order = _order(info, sizes)
        lab = [index_combs[i] for i in order]
        lens = np.array([len(item) for item in lab])
        info = info[order]

        if lens.max() >= 2:
            for level in range(1, int(lens.max())):
                higher = np.where(lens == level + 1)[0]
                if higher.size == 0 or not np.any(lens == level):
                    continue
                cap = info[lens == level].max()
                weak = higher[info[higher] < cap]
                info[weak] = 0.0

        order = _order(info, lens)
        lab = [lab[i] for i in order]
        increments = np.diff(info[order], prepend=0.0)
        red_vars = list(inds)

        unique = {name: 0.0 for name in source_names}
        redundant: Dict[str, float] = {}
        synergistic: Dict[str, float] = {}
        for inc, members in zip(increments, lab):
            mass = float(inc)
            if len(members) == 1:
                pool = tuple(red_vars)
                if len(pool) == 1:
                    unique[source_names[pool[0] - 1]] += mass
                else:
                    rname = subset_key(tuple(source_names[j - 1] for j in pool))
                    redundant[rname] = redundant.get(rname, 0.0) + mass
                red_vars.remove(members[0])
            else:
                sname = subset_key(tuple(source_names[j - 1] for j in members))
                synergistic[sname] = synergistic.get(sname, 0.0) + mass

        poi_surd.append(
            {"unique": unique, "redundant": redundant, "synergistic": synergistic}
        )
    return poi_surd
