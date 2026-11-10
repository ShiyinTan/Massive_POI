"""Temporal derived variables (leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hour_bin(ts: pd.Timestamp, n_bins: int = 8) -> int:
    hour = int(ts.hour)
    width = 24 // n_bins
    return hour // max(width, 1)


def day_type(ts: pd.Timestamp) -> int:
    # 0 weekday, 1 weekend
    return 1 if ts.dayofweek >= 5 else 0


def cyclical_hour(ts: pd.Timestamp) -> tuple[float, float]:
    ang = 2 * np.pi * ts.hour / 24.0
    return float(np.sin(ang)), float(np.cos(ang))


def cyclical_dow(ts: pd.Timestamp) -> tuple[float, float]:
    ang = 2 * np.pi * ts.dayofweek / 7.0
    return float(np.sin(ang)), float(np.cos(ang))


def query_time_features(ts: pd.Timestamp) -> np.ndarray:
    """Prediction-time context from last observed check-in timestamp (not target)."""
    hs, hc = cyclical_hour(ts)
    ds, dc = cyclical_dow(ts)
    return np.asarray(
        [hs, hc, ds, dc, float(ts.hour) / 23.0, float(ts.dayofweek) / 6.0, float(day_type(ts))],
        dtype=np.float32,
    )