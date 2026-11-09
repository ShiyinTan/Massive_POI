"""Q3 SpatialEncoder: history location sequence + 12-d summary → fused vector."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from surd_guided_poi.data.dataset import collate_next_poi
from surd_guided_poi.data.feature_builder import (
    HISTORY_SPATIAL_FEATURE_DIM,
    SPATIAL_FEATURE_DIM,
)
from surd_guided_poi.data.preprocessing import (
    NextPOISample,
    _history_spatial_features,
    _history_time_features,
)
from surd_guided_poi.models.encoders import SpatialEncoder


def test_history_spatial_features_shape_and_step_dist():
    t0 = pd.Timestamp(datetime(2013, 1, 1, 10, 0, 0))
    t1 = pd.Timestamp(datetime(2013, 1, 1, 12, 0, 0))
    # ~111 km north (1 degree lat)
    feats = _history_spatial_features([39.0, 40.0], [116.0, 116.0], [t0, t1])
    assert feats.shape == (2, HISTORY_SPATIAL_FEATURE_DIM)
    assert feats[0, 3] == np.float32(0.0)  # first step: no prev dist
    assert feats[1, 3] > 0  # second step has distance
    assert feats[1, 4] == np.float32(np.log1p(2.0))  # 2 hours
    assert feats[0, 5] == np.float32(0.0)
    assert feats[1, 5] == np.float32(1.0)


def test_spatial_encoder_shapes():
    B, L, d = 4, 5, 16
    enc = SpatialEncoder(
        hist_in_dim=HISTORY_SPATIAL_FEATURE_DIM,
        summary_in_dim=SPATIAL_FEATURE_DIM,
        hidden_dim=d,
    )
    hist = torch.randn(B, L, HISTORY_SPATIAL_FEATURE_DIM)
    lengths = torch.tensor([5, 3, 1, 4])
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, n in enumerate(lengths.tolist()):
        mask[i, :n] = True
    summary = torch.randn(B, SPATIAL_FEATURE_DIM)
    out = enc(hist, lengths, mask, summary)
    assert out.shape == (B, d)


def test_collate_pads_history_spatial_features():
    ts = pd.Timestamp(datetime(2013, 1, 1, 12, 0, 0))
    samples = []
    for i, L in enumerate([2, 1]):
        hist_ts = [ts - timedelta(hours=L - j) for j in range(L)]
        hist_lats = [39.0 + 0.01 * j for j in range(L)]
        hist_lons = [116.0] * L
        samples.append(
            NextPOISample(
                sample_id=i,
                user_id=0,
                trail_id="t",
                history_poi_ids=list(range(L)),
                history_timestamps=hist_ts,
                history_lats=hist_lats,
                history_lons=hist_lons,
                history_unix=[float(t.value // 10**9) for t in hist_ts],
                history_time_features=_history_time_features(hist_ts, ts),
                history_spatial_features=_history_spatial_features(
                    hist_lats, hist_lons, hist_ts
                ),
                target_poi_id=1,
                target_timestamp=ts,
                target_time_features=np.zeros(8, dtype=np.float32),
                spatial_features=np.zeros(12, dtype=np.float32),
                user_features=np.zeros(6, dtype=np.float32),
            )
        )
    batch = collate_next_poi(samples)
    assert batch.history_spatial_features.shape == (2, 2, HISTORY_SPATIAL_FEATURE_DIM)
    assert batch.spatial_features.shape == (2, SPATIAL_FEATURE_DIM)
