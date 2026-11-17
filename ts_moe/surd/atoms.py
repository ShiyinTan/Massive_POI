"""Four-source subset keys and the 21 TS-MoE expert slots.

Sources are User, POI, Time, Location. Expert order matches the spec table:

    unique U, P, T, L
    redundancy UP, UT, UL, PT, PL, TL
    synergy    UP, UT, UL, PT, PL, TL, UPT, UPL, UTL, PTL, UPTL
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

SOURCES: tuple[str, ...] = ("U", "P", "T", "L")
SOURCE_INDEX = {name: i for i, name in enumerate(SOURCES)}

PAIRS: tuple[tuple[str, ...], ...] = tuple(combinations(SOURCES, 2))
TRIPLES: tuple[tuple[str, ...], ...] = tuple(combinations(SOURCES, 3))
QUAD: tuple[tuple[str, ...], ...] = tuple(combinations(SOURCES, 4))
SYNERGY_SUBSETS: tuple[tuple[str, ...], ...] = PAIRS + TRIPLES + QUAD

# Non-empty subsets in SURD combination order (size, then source order).
ALL_SUBSETS: tuple[tuple[str, ...], ...] = tuple(
    comb for k in range(1, len(SOURCES) + 1) for comb in combinations(SOURCES, k)
)


def subset_key(subset: Iterable[str]) -> str:
    """``("U", "P")`` → ``"U_P"``."""
    return "_".join(subset)


SUBSET_KEYS: tuple[str, ...] = tuple(subset_key(s) for s in ALL_SUBSETS)

# (kind, member names) in the 21-column prior order.
EXPERTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    tuple(("unique", (name,)) for name in SOURCES)
    + tuple(("redundancy", pair) for pair in PAIRS)
    + tuple(("synergy", subset) for subset in SYNERGY_SUBSETS)
)

assert len(EXPERTS) == 21

VARIANT_GROUPS = {
    4: ("unique",),
    10: ("unique", "redundancy"),
    21: ("unique", "redundancy", "synergy"),
}


def expert_names(experts: Sequence[tuple[str, tuple[str, ...]]] | None = None) -> list[str]:
    """Unique experts are ``U``/``P``/``T``/``L``. Redundancy and synergy keep a prefix.

    A prefix is required so pairwise redundancy ``R_UP`` is not confused with
    unique ``P`` or synergy ``S_UP``.
    """
    catalog = EXPERTS if experts is None else experts
    out = []
    for kind, members in catalog:
        body = "".join(members)
        if kind == "unique":
            out.append(body)
        elif kind == "redundancy":
            out.append("R_" + body)
        else:
            out.append("S_" + body)
    return out


def select_experts(
    variant: int = 21,
    groups: Sequence[str] | None = None,
) -> tuple[list[int], list[tuple[str, tuple[str, ...]]]]:
    """Return column indices into the 21-d prior and the matching expert specs.

    ``groups`` overrides ``variant`` and enables the section-12 ablations
    (drop unique, redundancy, or synergy). Named variants are TS-MoE-4 / 10 / 21.
    """
    if groups is None:
        if variant not in VARIANT_GROUPS:
            raise ValueError(f"expert variant must be 4, 10, or 21, got {variant}")
        chosen = set(VARIANT_GROUPS[variant])
    else:
        chosen = set(groups)
        unknown = chosen.difference({"unique", "redundancy", "synergy"})
        if unknown:
            raise ValueError(f"unknown expert groups: {sorted(unknown)}")
        if not chosen:
            raise ValueError("at least one expert group is required")
    indices: list[int] = []
    specs: list[tuple[str, tuple[str, ...]]] = []
    for i, spec in enumerate(EXPERTS):
        if spec[0] in chosen:
            indices.append(i)
            specs.append(spec)
    return indices, specs


def atoms_to_expert_mass(atoms: dict) -> np.ndarray:
    """Map one POI's SURD atoms onto the 21 expert columns.

    Unique atoms fill the four unique experts. Synergy atoms fill the eleven
    synergy experts (missing atoms stay 0).

    Official SURD redundancy is booked to the *current redundancy pool*, which
    can be a pair, a triple, or the full set — not only the six pairs that have
    experts. A pool ``A`` with ``|A| > 2`` is split uniformly across the pairs
    inside ``A``, so total redundant mass is conserved and every redundancy
    expert has a defined prior. A pairwise atom is copied onto that pair.
    """
    mass = np.zeros(len(EXPERTS), dtype=np.float64)
    unique = atoms.get("unique", {})
    for i, name in enumerate(SOURCES):
        mass[i] = float(unique.get(name, 0.0))

    pair_index = {pair: 4 + i for i, pair in enumerate(PAIRS)}
    redundant = atoms.get("redundant", {})
    for key, value in redundant.items():
        members = tuple(key.split("_"))
        pairs = list(combinations(members, 2))
        if not pairs:
            continue
        share = float(value) / len(pairs)
        for pair in pairs:
            mass[pair_index[pair]] += share

    synergistic = atoms.get("synergistic", {})
    for i, subset in enumerate(SYNERGY_SUBSETS):
        mass[10 + i] = float(synergistic.get(subset_key(subset), 0.0))
    return mass
