"""Optional cold user / POI filtering (``remove_unseen_user_poi``-style).

Flashback / Graph-Flashback keep transductive vocab and do **not** drop cold
users/POIs at eval by default. Set ``eval.drop_unseen_user_poi: true``
(or ``data.drop_unseen_user_poi``) to drop val/test rows whose user or target
POI never appears in the train split.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Set, Tuple

import pandas as pd

from .preprocessing import NextPOISample, Vocab


def seen_user_poi_ids_from_train_frame(
    train_frame: pd.DataFrame,
    vocab: Vocab,
) -> Tuple[Set[int], Set[int]]:
    """Users / POIs that appear as check-ins in official train CSV (vocab indices)."""
    users: Set[int] = set()
    for u in train_frame["user_id"].unique():
        if u in vocab.user2idx:
            users.add(vocab.user2idx[u])
    pois: Set[int] = set()
    for v in train_frame["venue_id"].unique():
        vi = int(v)
        if vi in vocab.poi2idx:
            pois.add(vocab.poi2idx[vi])
    return users, pois


def seen_user_poi_ids_from_samples(
    samples: Iterable[NextPOISample],
) -> Tuple[Set[int], Set[int]]:
    """Fallback: union of users and POIs touched by train samples."""
    users: Set[int] = set()
    pois: Set[int] = set()
    for s in samples:
        users.add(int(s.user_id))
        pois.update(int(p) for p in s.history_poi_ids)
        pois.add(int(s.target_poi_id))
    return users, pois


def filter_unseen_user_poi(
    samples: Sequence[NextPOISample],
    train_user_ids: Set[int],
    train_poi_ids: Set[int],
) -> list[NextPOISample]:
    """Keep samples whose ``user_id`` and ``target_poi_id`` both appeared in train.

    Mirrors ``Massive-STEPS/.../preprocess_fn.py::remove_unseen_user_poi``.
    """
    return [
        s
        for s in samples
        if int(s.user_id) in train_user_ids and int(s.target_poi_id) in train_poi_ids
    ]


def maybe_drop_unseen_user_poi(
    samples: Sequence[NextPOISample],
    *,
    train_user_ids: Set[int],
    train_poi_ids: Set[int],
    enabled: bool,
) -> list[NextPOISample]:
    if not enabled:
        return list(samples)
    return filter_unseen_user_poi(samples, train_user_ids, train_poi_ids)
