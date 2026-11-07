"""Temporal split leakage checks."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from surd_guided_poi.data.preprocessing import NextPOISample
import numpy as np
from surd_guided_poi.data.splits import carve_estimator_holdout, temporal_split_samples


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
        target_poi_id=1,
        target_timestamp=ts,
        target_time_features=np.zeros(8, dtype=np.float32),
        spatial_features=np.zeros(12, dtype=np.float32),
        user_features=np.zeros(6, dtype=np.float32),
    )


def test_temporal_split_ordering():
    samples = [_s(i, i) for i in range(20)]
    splits = temporal_split_samples(samples, 0.7, 0.1, 0.2)
    train_max = max(s.target_timestamp for s in splits["train"])
    val_min = min(s.target_timestamp for s in splits["val"])
    val_max = max(s.target_timestamp for s in splits["val"])
    test_min = min(s.target_timestamp for s in splits["test"])
    assert train_max <= val_min
    assert val_max <= test_min


def test_estimator_holdout_inside_train_period():
    samples = [_s(i, i) for i in range(20)]
    splits = temporal_split_samples(samples, 0.7, 0.1, 0.2)
    model_train, estimation = carve_estimator_holdout(splits["train"], 0.2)
    assert model_train and estimation
    assert max(s.target_timestamp for s in model_train) <= min(
        s.target_timestamp for s in estimation
    )
    assert max(s.target_timestamp for s in estimation) <= max(
        s.target_timestamp for s in splits["train"]
    )