"""Q1 MinHash + KMeans: visit-set signatures, train-only seed, leakage-safe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pc_smir.features.sources import (
    MINHASH_PRIME,
    SourceDiscretizer,
    build_visit_set,
    minhash_signature,
)


def _vocab():
    return type(
        "V",
        (),
        {"spatial_cell_size_m": 500.0, "lat_origin": 39.0, "lon_origin": 116.0},
    )()


def _ex(pois, cats=None, **extra):
    d = {
        "history_poi_ids": list(pois),
        "history_categories": list(cats or []),
        "history_coordinates": [(39.9, 116.4)] * max(1, len(pois) or 1),
        "query_time": pd.Timestamp("2020-06-15 14:30:00"),
        "stable_user_features": np.zeros(8, dtype=np.float64),
        "vocab": _vocab(),
    }
    d.update(extra)
    return d


def test_build_visit_set_namespaces_and_dedup():
    tokens = build_visit_set(_ex([10, 10, 20], cats=[3, 3]))
    assert tokens == {"p:10", "p:20", "c:3"}


def test_minhash_order_invariant_and_seed_freeze():
    disc = SourceDiscretizer(n_minhash=16, minhash_seed=7)
    disc._ensure_minhash_params()
    a, b = disc.minhash_a.copy(), disc.minhash_b.copy()

    base = _ex([5, 9, 12, 5], cats=[1, 2])
    shuffled = _ex([12, 5, 9], cats=[2, 1])
    s1 = disc._q1_vector(base)
    s2 = disc._q1_vector(shuffled)
    assert s1.shape == (16,)
    assert np.allclose(s1, s2)

    # Same seed → same (a,b); second ensure must not resample
    disc2 = SourceDiscretizer(n_minhash=16, minhash_seed=7)
    disc2._ensure_minhash_params()
    assert np.array_equal(disc2.minhash_a, a)
    assert np.array_equal(disc2.minhash_b, b)
    assert np.allclose(disc2._q1_vector(base), s1)

    # Different seed → different params (almost surely)
    disc3 = SourceDiscretizer(n_minhash=16, minhash_seed=99)
    disc3._ensure_minhash_params()
    assert not np.array_equal(disc3.minhash_a, a)


def test_minhash_jaccard_monotonicity_rough():
    """Higher Jaccard → typically closer L1 signature distance (probabilistic)."""
    rng_a = np.array([3, 5, 7, 11, 13, 17, 19, 23], dtype=np.int64)
    rng_b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    a = {"p:1", "p:2", "p:3", "p:4"}
    b_close = {"p:1", "p:2", "p:3", "p:5"}  # Jaccard 3/5
    b_far = {"p:10", "p:11", "p:12", "p:13"}  # Jaccard 0
    sa = minhash_signature(a, rng_a, rng_b)
    sc = minhash_signature(b_close, rng_a, rng_b)
    sf = minhash_signature(b_far, rng_a, rng_b)
    assert np.sum(np.abs(sa - sc)) < np.sum(np.abs(sa - sf))


def test_empty_history_ones_and_no_target_leak():
    disc = SourceDiscretizer(n_minhash=8, minhash_seed=1)
    empty = _ex([], cats=[])
    sig = disc._q1_vector(empty)
    assert np.allclose(sig, np.ones(8))

    # target_poi_id must not enter the visit set even if present on the example
    leaky = _ex([1, 2], cats=[9])
    leaky["target_poi_id"] = 999
    tokens = build_visit_set(leaky)
    assert "p:999" not in tokens
    assert tokens == {"p:1", "p:2", "c:9"}


def test_fit_transform_interface_unchanged():
    examples = [
        _ex([1, 2, 3], cats=[1]),
        _ex([1, 2, 4], cats=[1]),
        _ex([10, 11], cats=[2]),
        _ex([10, 12], cats=[2]),
        _ex([20], cats=[3]),
        _ex([21], cats=[3]),
        _ex([1, 10], cats=[1, 2]),
        _ex([2, 11], cats=[1, 2]),
    ]
    # pad coords + vocab for Q3
    for ex in examples:
        ex["history_coordinates"] = [
            (39.9 + i * 0.01, 116.4) for i in range(max(2, len(ex["history_poi_ids"])))
        ]
        ex["vocab"] = _vocab()

    disc = SourceDiscretizer(n_clusters=3, n_minhash=16, minhash_seed=42)
    disc.fit(examples)
    assert disc.minhash_a is not None and disc.minhash_a.shape == (16,)
    out = disc.transform_example(examples[0])
    assert set(out) == {"Q1", "Q2", "Q3", "Q4"}
    assert all(isinstance(out[k], int) for k in out)
    assert 0 <= out["Q1"] < disc.q1_model.n_clusters


def test_signature_in_unit_interval():
    disc = SourceDiscretizer(n_minhash=32, minhash_seed=0)
    sig = disc._q1_vector(_ex([7, 8, 9], cats=[4]))
    assert sig.shape == (32,)
    assert np.all(sig >= 0.0) and np.all(sig < 1.0)
    assert disc.minhash_a.min() >= 1
    assert disc.minhash_a.max() < MINHASH_PRIME
