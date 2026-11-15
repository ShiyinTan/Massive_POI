"""Eval/val/test history stays inside the split trail prefix (no dump leak)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from surd_guided_poi.data.preprocessing import build_samples_from_split, build_vocab_from_frames


def _frame(rows):
    return pd.DataFrame(rows)


def test_history_is_same_trail_prefix_only_within_split():
    """Test CSV sample must not pull train check-ins into sequence history."""
    base = datetime(2013, 1, 1, 12, 0, 0)
    train = _frame(
        [
            {
                "trail_id": "tr_train",
                "user_id": 1,
                "venue_id": 100,
                "timestamp": pd.Timestamp(base),
                "latitude": 39.0,
                "longitude": 116.0,
                "venue_category": "A",
            },
            {
                "trail_id": "tr_train",
                "user_id": 1,
                "venue_id": 101,
                "timestamp": pd.Timestamp(base + timedelta(hours=1)),
                "latitude": 39.1,
                "longitude": 116.1,
                "venue_category": "B",
            },
        ]
    )
    # Same user, different trail in test; later times
    test = _frame(
        [
            {
                "trail_id": "tr_test",
                "user_id": 1,
                "venue_id": 200,
                "timestamp": pd.Timestamp(base + timedelta(days=10)),
                "latitude": 40.0,
                "longitude": 117.0,
                "venue_category": "C",
            },
            {
                "trail_id": "tr_test",
                "user_id": 1,
                "venue_id": 201,
                "timestamp": pd.Timestamp(base + timedelta(days=10, hours=1)),
                "latitude": 40.1,
                "longitude": 117.1,
                "venue_category": "D",
            },
            {
                "trail_id": "tr_test",
                "user_id": 1,
                "venue_id": 202,
                "timestamp": pd.Timestamp(base + timedelta(days=10, hours=2)),
                "latitude": 40.2,
                "longitude": 117.2,
                "venue_category": "E",
            },
        ]
    )
    vocab = build_vocab_from_frames({"train": train, "val": train, "test": test})
    # Even if someone wrongly passes train+test as user_history_df, sequence history
    # still comes only from the test trail prefixes.
    samples = build_samples_from_split(
        test,
        vocab,
        "test",
        use_all_prefixes=True,
        user_history_df=pd.concat([train, test], ignore_index=True),
    )
    assert len(samples) == 2  # predict 2nd and 3rd check-ins
    train_poi_idxs = {vocab.poi2idx[100], vocab.poi2idx[101]}
    for s in samples:
        assert s.split == "test"
        assert s.trail_id == "tr_test"
        assert all(ts < s.target_timestamp for ts in s.history_timestamps)
        # No train venues in sequence history
        assert set(s.history_poi_ids).isdisjoint(train_poi_idxs)
        assert vocab.idx2poi[s.target_poi_id] in {201, 202}


def test_history_never_includes_target_or_future_in_trail():
    base = datetime(2013, 2, 1, 8, 0, 0)
    df = _frame(
        [
            {
                "trail_id": "t1",
                "user_id": 7,
                "venue_id": 10 + i,
                "timestamp": pd.Timestamp(base + timedelta(hours=i)),
                "latitude": 39.0 + 0.01 * i,
                "longitude": 116.0,
                "venue_category": "X",
            }
            for i in range(4)
        ]
    )
    vocab = build_vocab_from_frames({"train": df, "val": df, "test": df})
    samples = build_samples_from_split(df, vocab, "val", use_all_prefixes=True)
    # targets at indices 1,2,3 → histories of length 1,2,3
    assert [len(s.history_poi_ids) for s in samples] == [1, 2, 3]
    for s in samples:
        assert s.target_poi_id not in s.history_poi_ids
        assert all(ts < s.target_timestamp for ts in s.history_timestamps)
        assert s.history_poi_ids == [
            vocab.poi2idx[10 + j] for j in range(len(s.history_poi_ids))
        ]
