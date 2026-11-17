#!/usr/bin/env python3
"""Check check_ins_id densify for graph_scope=train (no HF / PyG needed)."""
from __future__ import annotations

import os.path as osp
import sys

import numpy as np
import pandas as pd

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def reindex_check_ins_id_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Inline copy of preprocess_fn helper (avoid importing preprocess → torch_sparse)."""
    out = df.copy()
    order = np.argsort(out["check_ins_id"].to_numpy(), kind="mergesort")
    new_ids = np.empty(len(out), dtype=np.int64)
    new_ids[order] = np.arange(len(out), dtype=np.int64)
    out["check_ins_id"] = new_ids
    return out


def ensure_dense_check_ins_id(data: pd.DataFrame) -> pd.DataFrame:
    """Mirror generate_hypergraph.ensure_dense_check_ins_id without torch/PyG imports."""
    n_unique = int(data["check_ins_id"].nunique())
    id_max = int(data["check_ins_id"].max()) if n_unique else -1
    if n_unique == id_max + 1:
        return data
    return reindex_check_ins_id_chronological(data)


def assert_dense_chronological(data: pd.DataFrame) -> None:
    """Same check as generate_ci2traj_pyg_data line ~150."""
    assert data.check_ins_id.unique().shape[0] == data.check_ins_id.max() + 1, (
        "check_ins_id is not chronological order in raw data"
    )


def main() -> int:
    # Holes like cached graph_sample.csv after trail filter (e.g. 0,1,5).
    df = pd.DataFrame(
        {
            "check_ins_id": [0, 1, 5],
            "UTCTimeOffsetEpoch": [100, 110, 150],
        }
    )
    assert df.check_ins_id.nunique() != df.check_ins_id.max() + 1
    try:
        assert_dense_chronological(df)
        raise AssertionError("expected assert to fail on holes")
    except AssertionError as exc:
        if "chronological" not in str(exc):
            raise

    out = ensure_dense_check_ins_id(df)
    assert_dense_chronological(out)
    assert list(out.check_ins_id) == [0, 1, 2]

    # Already dense (graph_scope=full) → no-op identity.
    dense = pd.DataFrame({"check_ins_id": [0, 1, 2], "UTCTimeOffsetEpoch": [1, 2, 3]})
    same = ensure_dense_check_ins_id(dense)
    assert list(same.check_ins_id) == [0, 1, 2]
    assert_dense_chronological(same)

    # Larger hole set via reindex helper
    wide = pd.DataFrame(
        {"check_ins_id": [0, 3, 4, 8, 10], "UTCTimeOffsetEpoch": [100, 130, 140, 180, 200]}
    )
    wide_out = reindex_check_ins_id_chronological(wide)
    assert_dense_chronological(wide_out)
    assert list(wide_out.check_ins_id) == [0, 1, 2, 3, 4]

    print("SCOPE_TRAIN_REINDEX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
