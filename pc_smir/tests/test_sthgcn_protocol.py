"""STHGCN-aligned eval protocol (eval.sthgcn_protocol) unit tests."""

from __future__ import annotations

import pandas as pd

from pc_smir.data.dataset import build_examples, train_seen_user_poi_sets
from pc_smir.data.vocab import OOV_ID, fit_vocab


def _row(user, trail, venue, cat, ts, lat=39.9, lon=116.4):
    return {
        "trail_id": trail,
        "user_id": user,
        "venue_id": venue,
        "latitude": lat,
        "longitude": lon,
        "venue_category": cat,
        "venue_category_id_code": 0,
        "venue_city_latitude": lat,
        "venue_city_longitude": lon,
        "timestamp": pd.Timestamp(ts),
    }


def _tiny_frames():
    # Train: user 1 visits venues A,B; user 2 visits A.
    train = pd.DataFrame(
        [
            _row(1, "t1", 10, "Cafe", "2020-01-01 10:00:00"),
            _row(1, "t1", 20, "Park", "2020-01-01 11:00:00"),
            _row(1, "t1", 10, "Cafe", "2020-01-01 12:00:00"),
            _row(2, "t2", 10, "Cafe", "2020-01-02 10:00:00"),
            _row(2, "t2", 20, "Park", "2020-01-02 11:00:00"),
        ]
    )
    # Test trail for seen user+POI, with an earlier train history for user 1.
    # Also includes a history-only OOV venue 99 that must map to UNK, not drop.
    test = pd.DataFrame(
        [
            _row(1, "t3", 99, "Museum", "2020-01-05 09:00:00"),  # history OOV
            _row(1, "t3", 20, "Park", "2020-01-05 10:00:00"),  # target seen in train
            # Unseen user 9 → should be dropped under protocol.
            _row(9, "t4", 10, "Cafe", "2020-01-06 10:00:00"),
            _row(9, "t4", 20, "Park", "2020-01-06 11:00:00"),
            # Seen user, unseen target POI 77 → dropped under protocol.
            _row(2, "t5", 10, "Cafe", "2020-01-07 10:00:00"),
            _row(2, "t5", 77, "Zoo", "2020-01-07 11:00:00"),
        ]
    )
    return train, test


def test_remove_unseen_user_poi_and_history_unk():
    train, test = _tiny_frames()
    vocab = fit_vocab(train)
    users, pois = train_seen_user_poi_sets(train)
    all_df = pd.concat([train, test], ignore_index=True)

    legacy = build_examples(test, vocab, max_len=8)
    # Legacy keeps all 3 trails (OOV target still encoded; Dataset may drop later).
    assert len(legacy) == 3

    proto = build_examples(
        test,
        vocab,
        max_len=8,
        sthgcn_protocol=True,
        all_df=all_df,
        train_user_ids=users,
        train_venue_ids=pois,
    )
    # Only trail t3 survives (user 1 + target 20 both in train).
    assert len(proto) == 1
    ex = proto[0]
    assert ex["user_id"] == 1
    assert ex["target_venue_raw"] == 20
    assert ex["target_poi_id"] != OOV_ID
    # History includes OOV venue 99 → UNK id, sample kept.
    assert OOV_ID in ex["history_poi_ids"]


def test_temporal_visibility_pulls_earlier_user_checkins():
    train, test = _tiny_frames()
    vocab = fit_vocab(train)
    users, pois = train_seen_user_poi_sets(train)
    all_df = pd.concat([train, test], ignore_index=True)

    # Same-trail-only history for t3 would be just venue 99.
    trail_only = build_examples(test, vocab, max_len=20)
    t3 = next(e for e in trail_only if e["trail_id"] == "t3")
    assert len(t3["history_poi_ids"]) == 1

    proto = build_examples(
        test,
        vocab,
        max_len=20,
        sthgcn_protocol=True,
        all_df=all_df,
        train_user_ids=users,
        train_venue_ids=pois,
    )
    ex = proto[0]
    # User 1's earlier train check-ins (t1) are visible before the t3 target.
    assert len(ex["history_poi_ids"]) > 1
    # Encoded train venue 10 should appear in the cross-trail history.
    id_10 = vocab.encode_poi(10)
    assert id_10 in ex["history_poi_ids"]
