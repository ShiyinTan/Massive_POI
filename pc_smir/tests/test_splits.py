from __future__ import annotations

from pathlib import Path

from pc_smir.data.splits import load_official_splits, split_manifest


def test_official_beijing_splits_disjoint():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "beijing"
    splits = load_official_splits("beijing", data_dir, download=True)
    trains = set(splits["train"]["trail_id"].astype(str))
    vals = set(splits["validation"]["trail_id"].astype(str))
    tests = set(splits["test"]["trail_id"].astype(str))
    assert not (trains & vals)
    assert not (trains & tests)
    assert not (vals & tests)
    man = split_manifest(splits)
    assert man["train"]["protocol"] == "massive_steps_official_trail_split"
    assert man["train"]["n_trails"] > 0