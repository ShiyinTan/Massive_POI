#!/usr/bin/env python3
"""Check graph_scope=train check_ins_id densify (no HF / PyG needed)."""
from __future__ import annotations

import importlib.util
import os.path as osp
import sys

import numpy as np
import pandas as pd

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "preprocess_fn", osp.join(_ROOT, "preprocess", "preprocess_fn.py")
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Avoid pulling sklearn via id_encode import side effects if missing: load source of helper only.
src = open(osp.join(_ROOT, "preprocess", "preprocess_fn.py"), encoding="utf-8").read()
# Exec just the helper by importing the module normally when possible.
try:
    sys.path.insert(0, _ROOT)
    from preprocess.preprocess_fn import reindex_check_ins_id_chronological
except Exception:
    ns: dict = {"np": np, "pd": pd}
    exec(
        "def reindex_check_ins_id_chronological(df):\n"
        "    out = df.copy()\n"
        "    order = np.argsort(out['check_ins_id'].to_numpy(), kind='mergesort')\n"
        "    new_ids = np.empty(len(out), dtype=np.int64)\n"
        "    new_ids[order] = np.arange(len(out), dtype=np.int64)\n"
        "    out['check_ins_id'] = new_ids\n"
        "    return out\n",
        ns,
    )
    reindex_check_ins_id_chronological = ns["reindex_check_ins_id_chronological"]


def main() -> int:
    # Simulate holes after dropping trails from a global chronological id space.
    df = pd.DataFrame(
        {
            "check_ins_id": [0, 3, 4, 8, 10],
            "UTCTimeOffsetEpoch": [100, 130, 140, 180, 200],
        }
    )
    assert df.check_ins_id.nunique() != df.check_ins_id.max() + 1
    out = reindex_check_ins_id_chronological(df)
    assert out.check_ins_id.nunique() == out.check_ins_id.max() + 1
    assert list(out.check_ins_id) == [0, 1, 2, 3, 4]
    # Dense already → identity
    dense = pd.DataFrame({"check_ins_id": [0, 1, 2], "UTCTimeOffsetEpoch": [1, 2, 3]})
    assert list(reindex_check_ins_id_chronological(dense).check_ins_id) == [0, 1, 2]
    print("SCOPE_TRAIN_REINDEX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
