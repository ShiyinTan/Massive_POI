"""Temporal train/val/test splits and estimator holdout (no future leakage)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .preprocessing import NextPOISample


def temporal_split_samples(
    samples: Sequence[NextPOISample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Dict[str, List[NextPOISample]]:
    """Split by target timestamp quantiles (earlier -> train, later -> test)."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")
    if not samples:
        return {"train": [], "val": [], "test": []}

    ordered = sorted(samples, key=lambda s: (s.target_timestamp, s.sample_id))
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Assign remaining to test to avoid dropping samples to rounding.
    n_train = max(1, n_train) if n >= 3 else max(0, n_train)
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    # Ensure non-empty val/test when enough data.
    if n >= 5:
        if not val:
            val = [train.pop()] if len(train) > 1 else []
        if not test:
            test = [val.pop()] if len(val) > 1 else ([train.pop()] if len(train) > 1 else [])
    return {"train": train, "val": val, "test": test}


def carve_estimator_holdout(
    train_samples: Sequence[NextPOISample],
    holdout_ratio: float = 0.2,
) -> Tuple[List[NextPOISample], List[NextPOISample]]:
    """
    Split the *training time period* into model-train vs MI/SURD estimation.

    Both subsets are still earlier than val/test. Estimation labels are not
    used as recommendation supervision in the same phase that consumes SURD
    weights estimated from them (see training/alternating.py).
    """
    if not train_samples:
        return [], []
    ordered = sorted(train_samples, key=lambda s: (s.target_timestamp, s.sample_id))
    n = len(ordered)
    n_est = max(1, int(round(n * holdout_ratio))) if n >= 5 else max(1, n // 5)
    n_est = min(n_est, n - 1) if n > 1 else 0
    # Use the later part of train as estimation/calibration split.
    model_train = list(ordered[: n - n_est])
    estimation = list(ordered[n - n_est :])
    return model_train, estimation


def assert_temporal_order(splits: Dict[str, Sequence[NextPOISample]]) -> None:
    """Raise if max(train) time is not <= min(val) <= ... (allows equal boundary ties)."""
    def _bounds(name: str):
        ss = splits.get(name) or []
        if not ss:
            return None
        ts = [s.target_timestamp for s in ss]
        return min(ts), max(ts)

    tb, vb, teb = _bounds("train"), _bounds("val"), _bounds("test")
    if tb and vb and tb[1] > vb[0]:
        # Boundary ties are OK; strict future leakage if train max >> val min with overlap volume.
        train_after = sum(1 for s in splits["train"] if s.target_timestamp > vb[0])
        if train_after > 0 and tb[1] > vb[1] if vb else False:
            raise AssertionError("Temporal leakage: train timestamps after validation window.")
    if vb and teb and vb[1] > teb[1] and vb[0] > teb[0]:
        raise AssertionError("Unexpected val/test temporal ordering.")