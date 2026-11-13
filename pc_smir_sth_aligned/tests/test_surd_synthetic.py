"""Synthetic qualitative checks for mmi_pid_discrete_v1 (spec §7.6)."""

from __future__ import annotations

import numpy as np

from pc_smir_sth_aligned.surd.estimator import estimate_surd


def test_unique_from_q1_alone():
    rng = np.random.default_rng(0)
    n = 2000
    q1 = rng.integers(0, 2, size=n)
    q2 = rng.integers(0, 2, size=n)
    y = q1.copy()
    out = estimate_surd(y, {"Q1": q1, "Q2": q2}, ["Q1", "Q2"], {"pairs": [("Q1", "Q2")], "triples": []})
    assert out["unique"]["Q1"] > out["unique"]["Q2"]
    assert out["synergy"][("Q1", "Q2")] < out["unique"]["Q1"]


def test_synergy_from_xor():
    rng = np.random.default_rng(1)
    n = 4000
    q1 = rng.integers(0, 2, size=n)
    q2 = rng.integers(0, 2, size=n)
    y = np.bitwise_xor(q1, q2)
    out = estimate_surd(y, {"Q1": q1, "Q2": q2}, ["Q1", "Q2"], {"pairs": [("Q1", "Q2")], "triples": [], "alpha": 0.1})
    assert out["synergy"][("Q1", "Q2")] > out["unique"]["Q1"]
    assert out["synergy"][("Q1", "Q2")] > out["unique"]["Q2"]
    assert out["redundancy"][("Q1", "Q2")] <= out["synergy"][("Q1", "Q2")]


def test_redundancy_from_duplicate_sources():
    rng = np.random.default_rng(2)
    n = 2000
    q1 = rng.integers(0, 2, size=n)
    q2 = q1.copy()
    y = q1.copy()
    out = estimate_surd(y, {"Q1": q1, "Q2": q2}, ["Q1", "Q2"], {"pairs": [("Q1", "Q2")], "triples": []})
    assert out["redundancy"][("Q1", "Q2")] > out["synergy"][("Q1", "Q2")]
    assert out["redundancy"][("Q1", "Q2")] >= out["unique"]["Q1"]
    assert out["redundancy"][("Q1", "Q2")] >= out["unique"]["Q2"]


def test_adapter_metadata_declares_atoms():
    y = np.array([0, 1, 0, 1])
    q1 = np.array([0, 1, 0, 1])
    q2 = np.array([1, 0, 1, 0])
    out = estimate_surd(y, {"Q1": q1, "Q2": q2}, ["Q1", "Q2"], {"pairs": [("Q1", "Q2")]})
    assert out["metadata"]["estimator"] == "mmi_pid_discrete_v1"
    assert "Q1" in out["metadata"]["identifiable_atoms"]["unique"]
    assert ["Q1", "Q2"] in out["metadata"]["identifiable_atoms"]["synergy_pairs"]