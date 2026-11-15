"""Eval Acc@1 protocol: last POI per trail only."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from surd_guided_poi.data.preprocessing import build_samples_from_split, build_vocab_from_frames
from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking


def test_last_only_one_sample_per_trail():
    base = datetime(2013, 3, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "trail_id": "t1",
                "user_id": 1,
                "venue_id": 10 + i,
                "timestamp": pd.Timestamp(base + timedelta(hours=i)),
                "latitude": 39.0,
                "longitude": 116.0,
                "venue_category": "A",
            }
            for i in range(5)
        ]
    )
    vocab = build_vocab_from_frames({"train": df, "val": df, "test": df})
    all_pref = build_samples_from_split(df, vocab, "test", use_all_prefixes=True)
    last_only = build_samples_from_split(df, vocab, "test", use_all_prefixes=False)
    assert len(all_pref) == 4
    assert len(last_only) == 1
    assert last_only[0].target_poi_id == vocab.poi2idx[14]
    assert last_only[0].history_poi_ids == [vocab.poi2idx[10 + i] for i in range(4)]


def test_acc_at_1_alias_equals_recall_at_1():
    import torch
    from surd_guided_poi.data.dataset import NextPOIBatch
    from surd_guided_poi.models.full_model import SURDGuidedPOIModel

    B, L, K = 4, 3, 8
    model = SURDGuidedPOIModel(
        num_pois=K,
        num_users=2,
        hidden_dim=8,
        poi_emb_dim=8,
        subset_hidden_dim=8,
        rnn_type="rnn",
        dropout=0.0,
    )

    class _Loader:
        def __iter__(self):
            yield NextPOIBatch(
                history_poi_ids=torch.randint(0, K, (B, L)),
                history_lengths=torch.full((B,), L, dtype=torch.long),
                history_mask=torch.ones(B, L, dtype=torch.bool),
                history_unix=torch.linspace(1e9, 1e9 + 3600 * L, L).unsqueeze(0).expand(B, L).float(),
                history_coords=torch.zeros(B, L, 2),
                history_time_features=torch.zeros(B, L, 10),
                history_spatial_features=torch.zeros(B, L, 6),
                target_time_features=torch.zeros(B, 8),
                spatial_features=torch.zeros(B, 12),
                user_features=torch.zeros(B, 6),
                target_poi_id=torch.randint(0, K, (B,)),
                user_id=torch.zeros(B, dtype=torch.long),
                sample_id=torch.arange(B),
            )

    metrics = evaluate_ranking(model, _Loader(), torch.device("cpu"), ks=(1, 5, 10))
    assert metrics["acc@1"] == metrics["recall@1"]
    assert metrics["acc@5"] == metrics["recall@5"]
