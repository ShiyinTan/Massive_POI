from __future__ import annotations

import pandas as pd

from pc_smir_sth_aligned.data.dataset import build_examples
from pc_smir_sth_aligned.data.vocab import fit_vocab


def _tiny_df():
    rows = []
    # one trail length 4 → last-only 1 example; expand → 3 examples
    for i, vid in enumerate([10, 20, 30, 40]):
        rows.append(
            {
                "trail_id": "t1",
                "user_id": 7,
                "venue_id": vid,
                "latitude": 39.9 + 0.01 * i,
                "longitude": 116.4,
                "venue_category": "Cafe",
                "venue_category_id_code": 1,
                "venue_city_latitude": 39.9,
                "venue_city_longitude": 116.4,
                "timestamp": pd.Timestamp(f"2013-01-01 10:0{i}:00"),
            }
        )
    return pd.DataFrame(rows)


def test_expand_prefixes_counts():
    df = _tiny_df()
    vocab = fit_vocab(df)
    last_only = build_examples(df, vocab, max_len=20, expand_prefixes=False)
    expanded = build_examples(df, vocab, max_len=20, expand_prefixes=True)
    assert len(last_only) == 1
    assert len(expanded) == 3
    assert last_only[0]["target_venue_raw"] == 40
    assert [e["target_venue_raw"] for e in expanded] == [20, 30, 40]
