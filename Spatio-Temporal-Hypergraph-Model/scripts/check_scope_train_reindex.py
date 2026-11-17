#!/usr/bin/env python3
"""Check densify helpers for graph_scope=train (no HF / PyG needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reindex_check_ins_id_chronological(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    order = np.argsort(out["check_ins_id"].to_numpy(), kind="mergesort")
    new_ids = np.empty(len(out), dtype=np.int64)
    new_ids[order] = np.arange(len(out), dtype=np.int64)
    out["check_ins_id"] = new_ids
    return out


def ensure_dense_check_ins_id(data: pd.DataFrame) -> pd.DataFrame:
    n_unique = int(data["check_ins_id"].nunique())
    id_max = int(data["check_ins_id"].max()) if n_unique else -1
    if n_unique == id_max + 1:
        return data
    return reindex_check_ins_id_chronological(data)


def ensure_dense_traj_id(data: pd.DataFrame, traj_column: str = "pseudo_session_trajectory_id"):
    ids = data[traj_column]
    n_unique = int(ids.nunique())
    id_max = int(ids.max()) if n_unique else -1
    if n_unique == id_max + 1:
        identity = {int(i): int(i) for i in ids.unique()}
        return data, identity, False
    sorted_ids = np.sort(ids.unique().astype(np.int64))
    old_to_new = {int(old): int(new) for new, old in enumerate(sorted_ids)}
    out = data.copy()
    out[traj_column] = out[traj_column].map(lambda x: old_to_new[int(x)])
    return out, old_to_new, True


def assert_dense_check_ins(data: pd.DataFrame) -> None:
    assert data.check_ins_id.unique().shape[0] == data.check_ins_id.max() + 1, (
        "check_ins_id is not chronological order in raw data"
    )


def he_size_list_from_groupby(data: pd.DataFrame, traj_column: str) -> list:
    """Same shape as generate_traj2traj_data's traj_size_adjust (.tolist() after groupby)."""
    return data.groupby(traj_column)["PoiId"].apply(lambda s: s.nunique()).tolist()


def main() -> int:
    # --- check_ins_id holes ---
    df = pd.DataFrame(
        {"check_ins_id": [0, 1, 5], "UTCTimeOffsetEpoch": [100, 110, 150]}
    )
    assert df.check_ins_id.nunique() != df.check_ins_id.max() + 1
    try:
        assert_dense_check_ins(df)
        raise AssertionError("expected assert to fail on holes")
    except AssertionError as exc:
        if "chronological" not in str(exc):
            raise
    out = ensure_dense_check_ins_id(df)
    assert_dense_check_ins(out)
    assert list(out.check_ins_id) == [0, 1, 2]

    dense = pd.DataFrame({"check_ins_id": [0, 1, 2], "UTCTimeOffsetEpoch": [1, 2, 3]})
    assert list(ensure_dense_check_ins_id(dense).check_ins_id) == [0, 1, 2]

    # --- traj id holes: he_size list must not IndexError ---
    traj_df = pd.DataFrame(
        {
            "pseudo_session_trajectory_id": [0, 0, 1, 1, 5, 5],
            "PoiId": [10, 11, 20, 21, 30, 31],
            "UserId": [1, 1, 2, 2, 3, 3],
        }
    )
    # Before densify: groupby.tolist() has len=3 but raw id 5 indexes off the end.
    he_before = he_size_list_from_groupby(traj_df, "pseudo_session_trajectory_id")
    assert len(he_before) == 3
    try:
        _ = he_before[5]
        raise AssertionError("expected IndexError for traj id 5 on short he_size")
    except IndexError:
        pass

    traj_out, traj_map, changed = ensure_dense_traj_id(traj_df)
    assert changed
    assert traj_map == {0: 0, 1: 1, 5: 2}
    assert list(sorted(traj_out["pseudo_session_trajectory_id"].unique())) == [0, 1, 2]
    he_after = he_size_list_from_groupby(traj_out, "pseudo_session_trajectory_id")
    assert len(he_after) == 3
    # All edge endpoints are valid list indices.
    for tid in traj_out["pseudo_session_trajectory_id"].unique():
        _ = he_after[int(tid)]

    # Already dense traj ids → no-op
    dense_traj = pd.DataFrame(
        {"pseudo_session_trajectory_id": [0, 1, 2], "PoiId": [1, 2, 3], "UserId": [1, 2, 3]}
    )
    _, _, changed2 = ensure_dense_traj_id(dense_traj)
    assert changed2 is False

    print("SCOPE_TRAIN_REINDEX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
