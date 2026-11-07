"""Unit tests for POI-specific mutual information."""

from __future__ import annotations

import numpy as np

from surd_guided_poi.information.specific_mi import enforce_subset_monotonicity, estimate_specific_mi


def test_specific_mi_uses_prior_not_uniform():
    # Two classes; all labels are class 0. Prior is empirical 1.0 / 0.0 clipped.
    N, K = 10, 3
    labels = np.zeros(N, dtype=np.int64)
    prior = np.array([1.0, 0.0, 0.0])  # will be clipped inside
    # Perfect predictions for class 0.
    probs = np.zeros((N, K), dtype=np.float64)
    probs[:, 0] = 1.0
    specific, counts = estimate_specific_mi({"Q1": probs}, labels, prior, eps=1e-12, units="nats")
    assert counts[0] == N
    assert counts[1] == 0
    assert np.isfinite(specific["Q1"][0])
    # log(1 / 1) = 0 when prior[0]=1
    assert abs(specific["Q1"][0]) < 1e-9


def test_specific_mi_log_ratio_uses_true_class_prob():
    N, K = 4, 2
    labels = np.array([0, 0, 1, 1])
    prior = np.array([0.5, 0.5])
    probs = np.array(
        [
            [0.8, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    specific, counts = estimate_specific_mi({"Q1": probs}, labels, prior, units="nats")
    expected = np.log(0.8 / 0.5)
    assert abs(specific["Q1"][0] - expected) < 1e-9
    assert abs(specific["Q1"][1] - expected) < 1e-9
    assert counts.tolist() == [2, 2]


def test_zero_prob_does_not_nan():
    N, K = 3, 2
    labels = np.array([0, 0, 0])
    prior = np.array([1.0, 0.0])
    probs = np.zeros((N, K))
    probs[:, 1] = 1.0  # never predicts class 0
    specific, _ = estimate_specific_mi({"Q1": probs}, labels, prior, eps=1e-12, units="nats")
    assert np.isfinite(specific["Q1"][0])


def test_monotonicity_projection():
    K = 2
    specific = {
        "Q1": np.array([0.5, 0.1]),
        "Q2": np.array([0.2, 0.4]),
        "Q3": np.array([0.1, 0.1]),
        "Q1_Q2": np.array([0.3, 0.2]),  # below max(Q1,Q2) for y=0
        "Q1_Q3": np.array([0.6, 0.2]),
        "Q2_Q3": np.array([0.2, 0.5]),
        "Q1_Q2_Q3": np.array([0.4, 0.3]),
    }
    out = enforce_subset_monotonicity(specific)
    assert out["Q1_Q2"][0] >= out["Q1"][0]
    assert out["Q1_Q2"][0] >= out["Q2"][0]
    assert out["Q1_Q2_Q3"][0] >= out["Q1_Q2"][0]