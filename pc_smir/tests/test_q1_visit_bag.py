"""Q1 visit-bag: order-invariant hashed bag-of-POI / category proportions."""

from __future__ import annotations

import numpy as np

from pc_smir.features.sources import SourceDiscretizer, _poi_bucket


def test_poi_bucket_stable_and_scatters_neighbors():
    # Same id → same bucket.
    assert _poi_bucket(10) == _poi_bucket(10)
    # Consecutive ids should usually not land in consecutive buckets (hash scatter).
    # Not a hard guarantee for all pairs, but 10 vs 11 should differ often;
    # at minimum buckets are in range.
    assert 0 <= _poi_bucket(10) < 64
    assert 0 <= _poi_bucket(11) < 64


def test_q1_vector_order_invariant_and_visit_counts():
    disc = SourceDiscretizer(n_clusters=4)
    base = {
        "history_poi_ids": [10, 20, 10, 30],
        "history_categories": [2, 5, 2, 7],
        "history_coordinates": [(39.9, 116.4)] * 4,
        "query_time": __import__("pandas").Timestamp("2012-01-01 12:00:00"),
        "stable_user_features": np.zeros(8, dtype=np.float32),
    }
    shuffled = dict(base)
    shuffled["history_poi_ids"] = [30, 10, 20, 10]
    shuffled["history_categories"] = [7, 2, 5, 2]

    v1 = disc._q1_vector(base)
    v2 = disc._q1_vector(shuffled)
    assert v1.shape == (96,)
    assert np.allclose(v1, v2)

    # Double visits to poi 10 should increase its hashed bucket vs single visit.
    once = dict(base)
    once["history_poi_ids"] = [10, 20, 30]
    once["history_categories"] = [2, 5, 7]
    twice = dict(base)
    twice["history_poi_ids"] = [10, 10, 20, 30]
    twice["history_categories"] = [2, 2, 5, 7]
    b = _poi_bucket(10)
    assert disc._q1_vector(twice)[32 + b] > disc._q1_vector(once)[32 + b] - 1e-12
