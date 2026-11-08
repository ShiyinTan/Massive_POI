"""Q2 TemporalEncoder: history time sequence + target time → fused vector."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from surd_guided_poi.data.dataset import collate_next_poi
from surd_guided_poi.data.feature_builder import HISTORY_TIME_FEATURE_DIM, TIME_FEATURE_DIM
from surd_guided_poi.data.preprocessing import NextPOISample, _history_time_features
from surd_guided_poi.models.encoders import TemporalEncoder


def test_history_time_features_shape_and_delta():
    t0 = pd.Timestamp(datetime(2013, 1, 1, 10, 0, 0))
    t1 = pd.Timestamp(datetime(2013, 1, 1, 12, 0, 0))
    target = pd.Timestamp(datetime(2013, 1, 1, 15, 0, 0))
    feats = _history_time_features([t0, t1], target)
    assert feats.shape == (2, HISTORY_TIME_FEATURE_DIM)
    # Δt hours for t1 → target = 3 → log1p(3) in last-but-one dim
    assert feats[1, 8] == np.float32(np.log1p(3.0))
    assert feats[1, 9] == np.float32(3.0 / 24.0)


def test_temporal_encoder_shapes():
    B, L, d = 4, 5, 16
    enc = TemporalEncoder(
        hist_in_dim=HISTORY_TIME_FEATURE_DIM,
        tgt_in_dim=TIME_FEATURE_DIM,
        hidden_dim=d,
    )
    hist = torch.randn(B, L, HISTORY_TIME_FEATURE_DIM)
    lengths = torch.tensor([5, 3, 1, 4])
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, n in enumerate(lengths.tolist()):
        mask[i, :n] = True
    tgt = torch.randn(B, TIME_FEATURE_DIM)
    out = enc(hist, lengths, mask, tgt)
    assert out.shape == (B, d)


def test_collate_pads_history_time_features():
    ts = pd.Timestamp(datetime(2013, 1, 1, 12, 0, 0))
    samples = []
    for i, L in enumerate([2, 1]):
        hist_ts = [ts - timedelta(hours=L - j) for j in range(L)]
        samples.append(
            NextPOISample(
                sample_id=i,
                user_id=0,
                trail_id="t",
                history_poi_ids=list(range(L)),
                history_timestamps=hist_ts,
                history_lats=[0.0] * L,
                history_lons=[0.0] * L,
                history_unix=[float(t.value // 10**9) for t in hist_ts],
                history_time_features=_history_time_features(hist_ts, ts),
                target_poi_id=1,
                target_timestamp=ts,
                target_time_features=np.zeros(8, dtype=np.float32),
                spatial_features=np.zeros(12, dtype=np.float32),
                user_features=np.zeros(6, dtype=np.float32),
            )
        )
    batch = collate_next_poi(samples)
    assert batch.history_time_features.shape == (2, 2, HISTORY_TIME_FEATURE_DIM)
    assert batch.target_time_features.shape == (2, TIME_FEATURE_DIM)
