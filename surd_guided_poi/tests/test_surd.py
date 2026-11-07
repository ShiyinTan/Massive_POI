"""Unit tests for official SURD assignment (POI-specific + global)."""

from __future__ import annotations

import numpy as np

from surd_guided_poi.information.surd import aggregate_global_surd, assign_surd_from_specific_mi
from surd_guided_poi.models.subset_estimators import SUBSET_NAMES, subset_key


def _toy_specific_mi_unique_q1():
    """Only Q1 carries information for the single target state."""
    K = 1
    values = {
        "Q1": np.array([1.0]),
        "Q2": np.array([0.0]),
        "Q3": np.array([0.0]),
        "Q1_Q2": np.array([1.0]),
        "Q1_Q3": np.array([1.0]),
        "Q2_Q3": np.array([0.0]),
        "Q1_Q2_Q3": np.array([1.0]),
    }
    return values


def test_unique_information_booked_to_singleton():
    specific = _toy_specific_mi_unique_q1()
    poi_surd = assign_surd_from_specific_mi(specific)
    u = poi_surd[0]["unique"]
    # Nearly all predictive information should be unique to Q1.
    assert u["Q1"] > 0.9
    assert u["Q2"] < 0.1
    assert u["Q3"] < 0.1


def test_synergy_when_only_joint_helps():
    # Singletons uninformative; full set informative.
    specific = {
        "Q1": np.array([0.0]),
        "Q2": np.array([0.0]),
        "Q3": np.array([0.0]),
        "Q1_Q2": np.array([0.0]),
        "Q1_Q3": np.array([0.0]),
        "Q2_Q3": np.array([0.0]),
        "Q1_Q2_Q3": np.array([1.0]),
    }
    poi_surd = assign_surd_from_specific_mi(specific)
    s = poi_surd[0]["synergistic"]
    assert s.get("Q1_Q2_Q3", 0.0) > 0.9


def test_global_aggregation_weights_by_prior():
    # Two POIs with clearly unique single-source information; supersets match.
    specific = {
        "Q1": np.array([1.0, 0.0]),
        "Q2": np.array([0.0, 1.0]),
        "Q3": np.array([0.0, 0.0]),
        "Q1_Q2": np.array([1.0, 1.0]),
        "Q1_Q3": np.array([1.0, 0.0]),
        "Q2_Q3": np.array([0.0, 1.0]),
        "Q1_Q2_Q3": np.array([1.0, 1.0]),
    }
    poi_surd = assign_surd_from_specific_mi(specific)
    prior = np.array([0.75, 0.25])
    global_surd = aggregate_global_surd(poi_surd, prior)
    # Population-weighted unique should lean toward Q1 (more prior mass on y=0).
    assert global_surd["unique"]["Q1"] > global_surd["unique"]["Q2"]
    assert abs(global_surd["unique"]["Q1"] - 0.75) < 1e-6
    assert abs(global_surd["unique"]["Q2"] - 0.25) < 1e-6


def test_permutation_of_equal_sources_stable_total():
    specific = {
        "Q1": np.array([0.5]),
        "Q2": np.array([0.5]),
        "Q3": np.array([0.5]),
        "Q1_Q2": np.array([0.5]),
        "Q1_Q3": np.array([0.5]),
        "Q2_Q3": np.array([0.5]),
        "Q1_Q2_Q3": np.array([0.5]),
    }
    poi_surd = assign_surd_from_specific_mi(specific)
    terms = poi_surd[0]
    total = sum(terms["unique"].values()) + sum(terms["redundant"].values()) + sum(
        terms["synergistic"].values()
    )
    assert abs(total - 0.5) < 1e-9