"""Optional cold user/POI filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surd_guided_poi.data.cold_filter import (
    filter_unseen_user_poi,
    maybe_drop_unseen_user_poi,
    seen_user_poi_ids_from_train_frame,
)
from surd_guided_poi.data.preprocessing import NextPOISample, Vocab


def _sample(sid, user_id, target_poi, hist=(0,)):
    return NextPOISample(
        sample_id=sid,
        user_id=user_id,
        trail_id=f"t{sid}",
        history_poi_ids=list(hist),
        history_timestamps=[pd.Timestamp("2020-01-01")],
        history_lats=[39.9],
        history_lons=[116.4],
        history_unix=[1e9],
        history_time_features=np.zeros((1, 10), dtype=np.float32),
        history_spatial_features=np.zeros((1, 6), dtype=np.float32),
        target_poi_id=target_poi,
        target_timestamp=pd.Timestamp("2020-01-02"),
        target_time_features=np.zeros(8, dtype=np.float32),
        spatial_features=np.zeros(12, dtype=np.float32),
        user_features=np.zeros(6, dtype=np.float32),
        split="test",
    )


def test_filter_unseen_user_poi_keeps_seen_only():
    samples = [
        _sample(0, user_id=0, target_poi=1),
        _sample(1, user_id=9, target_poi=1),  # cold user
        _sample(2, user_id=0, target_poi=8),  # cold poi
    ]
    kept = filter_unseen_user_poi(samples, train_user_ids={0, 1}, train_poi_ids={0, 1, 2})
    assert [s.sample_id for s in kept] == [0]


def test_maybe_drop_disabled_is_noop():
    samples = [_sample(0, 9, 8)]
    out = maybe_drop_unseen_user_poi(
        samples, train_user_ids={0}, train_poi_ids={0}, enabled=False
    )
    assert len(out) == 1


def test_seen_ids_from_train_frame():
    vocab = Vocab(
        poi2idx={100: 0, 200: 1, 300: 2},
        idx2poi=[100, 200, 300],
        user2idx={"u1": 0, "u2": 1},
        category2idx={"c": 0},
        poi_coords=np.zeros((3, 2)),
        poi_category_idx=np.zeros(3, dtype=int),
    )
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "venue_id": [100, 200],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    users, pois = seen_user_poi_ids_from_train_frame(train, vocab)
    assert users == {0}
    assert pois == {0, 1}
