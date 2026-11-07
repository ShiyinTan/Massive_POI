"""Official-split and estimator-holdout checks."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from surd_guided_poi.data.preprocessing import NextPOISample
from surd_guided_poi.data.splits import assert_official_split_disjoint_trails, carve_estimator_holdout


def _s(i: int, day: int) -> NextPOISample:
    ts = pd.Timestamp(datetime(2013, 1, 1) + timedelta(days=day))
    return NextPOISample(
        sample_id=i,
        user_id=0,
        trail_id="t",
        history_poi_ids=[0],
        history_timestamps=[ts - timedelta(hours=1)],
        history_lats=[0.0],
        history_lons=[0.0],
        history_unix=[float(ts.value // 10**9) - 3600],
        target_poi_id=1,
        target_timestamp=ts,
        target_time_features=np.zeros(8, dtype=np.float32),
        spatial_features=np.zeros(12, dtype=np.float32),
        user_features=np.zeros(6, dtype=np.float32),
        split="train",
    )


def test_estimator_holdout_inside_train_only():
    # estimation 应整体晚于 model_train（按 target_timestamp）
    samples = [_s(i, i) for i in range(20)]
    model_train, estimation = carve_estimator_holdout(samples, 0.2)
    assert model_train and estimation
    assert max(s.target_timestamp for s in model_train) <= min(
        s.target_timestamp for s in estimation
    )


def test_official_trails_disjoint():
    frames = {
        "train": pd.DataFrame({"trail_id": ["a", "b"]}),
        "val": pd.DataFrame({"trail_id": ["c"]}),
        "test": pd.DataFrame({"trail_id": ["d", "e"]}),
    }
    assert_official_split_disjoint_trails(frames)


def test_official_trails_overlap_raises():
    frames = {
        "train": pd.DataFrame({"trail_id": ["a", "b"]}),
        "val": pd.DataFrame({"trail_id": ["b"]}),
        "test": pd.DataFrame({"trail_id": ["c"]}),
    }
    try:
        assert_official_split_disjoint_trails(frames)
        assert False, "expected assertion"
    except AssertionError:
        pass