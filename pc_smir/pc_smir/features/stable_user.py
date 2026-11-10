"""Q4 stable-user behavioral profile from available prefix only."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from pc_smir.data.vocab import VocabMaps, haversine_m


def stable_user_features(
    poi_ids: Sequence[int],
    cat_ids: Sequence[int],
    coords: Sequence[tuple[float, float]],
    vocab: VocabMaps,
) -> np.ndarray:
    """Return fixed-length Q4 vector from history prefix (no target)."""
    n = len(poi_ids)
    if n == 0:
        return np.zeros(8, dtype=np.float32)

    visit_count = float(n)
    distinct_poi = float(len(set(poi_ids)))
    cat_counts = Counter(cat_ids)
    probs = np.asarray(list(cat_counts.values()), dtype=np.float64)
    probs = probs / probs.sum()
    cat_entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    distances = []
    for i in range(1, len(coords)):
        distances.append(haversine_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
    median_move = float(np.median(distances)) if distances else 0.0

    # Activity radius: max distance from centroid of observed coords.
    lats = np.asarray([c[0] for c in coords], dtype=np.float64)
    lons = np.asarray([c[1] for c in coords], dtype=np.float64)
    clat, clon = float(lats.mean()), float(lons.mean())
    radius = max(haversine_m(clat, clon, float(a), float(b)) for a, b in coords)

    # Popularity affinity of visited POIs (train-only counts).
    pop = [vocab.train_poi_counts.get(int(p), 0) for p in poi_ids]
    mean_log_pop = float(np.mean(np.log1p(pop))) if pop else 0.0

    return np.asarray(
        [
            np.log1p(visit_count),
            np.log1p(distinct_poi),
            distinct_poi / visit_count,
            cat_entropy,
            np.log1p(median_move),
            np.log1p(radius),
            mean_log_pop,
            float(len(cat_counts)),
        ],
        dtype=np.float32,
    )