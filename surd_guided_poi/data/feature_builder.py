"""Shared helpers for tensorizing sample fields."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from .preprocessing import NextPOISample


# ---------------------------------------------------------------------------
# 特征维数约定（与 preprocessing / TemporalEncoder / SpatialEncoder 一致）
# ---------------------------------------------------------------------------
# 目标时刻 8 维（_time_features）:
#   0: sin(2π·hour/24)   1: cos(2π·hour/24)
#   2: sin(2π·dow/7)     3: cos(2π·dow/7)
#   4: weekend(0/1)      5: sin(2π·(month-1)/12)
#   6: cos(2π·(month-1)/12)  7: hour/24
TIME_FEATURE_DIM = 8

# 历史每一步 10 维（_history_time_features）= 上面 8 维日历 + Δt 相对「预测目标时刻」:
#   8: log1p(Δ小时)      9: Δ小时/24
HISTORY_TIME_FEATURE_DIM = 10

# 空间 12 维（_spatial_features，由历史轨迹统计，不含目标点）:
#   0: last_lat/90       1: last_lon/180
#   2: mean_lat/90       3: mean_lon/180
#   4: std_lat           5: std_lon
#   6: log1p(mean_step_km)  7: log1p(max_step_km)
#   8: log1p(disp_km)    9: log1p(span_hours)
#  10: history_len       11: missing_last_coord(0/1)
SPATIAL_FEATURE_DIM = 12

USER_FEATURE_DIM = 6


def pad_history(
    samples: Sequence[NextPOISample],
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return history_poi_ids [B,L], lengths [B], mask [B,L]."""
    lengths = [len(s.history_poi_ids) for s in samples]
    L = max(lengths) if max_len is None else max_len
    B = len(samples)
    ids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, s in enumerate(samples):
        h = s.history_poi_ids[-L:]
        ids[i, : len(h)] = torch.tensor(h, dtype=torch.long)
        mask[i, : len(h)] = True
    length_t = torch.tensor([min(l, L) for l in lengths], dtype=torch.long)
    return ids, length_t, mask