"""Smoke tests for model.backbone = flashback | graph_flashback | sthgcn*."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from surd_guided_poi.data.dataset import NextPOIBatch
from surd_guided_poi.data.preprocessing import NextPOISample
from surd_guided_poi.models.backbone_factory import SUPPORTED_BACKBONES, build_backbone
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.models.graph_builders import (
    build_interact_graph_from_samples,
    build_transition_graph_from_samples,
)


def _fake_batch(B=4, L=5, num_pois=10):
    return NextPOIBatch(
        history_poi_ids=torch.randint(0, num_pois, (B, L)),
        history_lengths=torch.full((B,), L, dtype=torch.long),
        history_mask=torch.ones(B, L, dtype=torch.bool),
        history_unix=torch.linspace(1e9, 1e9 + 3600 * L, L).unsqueeze(0).expand(B, L).float(),
        history_coords=torch.randn(B, L, 2),
        history_time_features=torch.randn(B, L, 10),
        history_spatial_features=torch.randn(B, L, 6),
        target_time_features=torch.randn(B, 8),
        spatial_features=torch.randn(B, 12),
        user_features=torch.randn(B, 6),
        target_poi_id=torch.randint(0, num_pois, (B,)),
        user_id=torch.randint(0, 3, (B,)),
        sample_id=torch.arange(B),
    )


def _toy_samples(num_pois=10, num_users=3, n=20):
    samples = []
    for i in range(n):
        hist = [i % num_pois, (i + 1) % num_pois]
        ts0 = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i)
        samples.append(
            NextPOISample(
                sample_id=i,
                user_id=i % num_users,
                trail_id=f"t{i}",
                history_poi_ids=hist,
                history_timestamps=[ts0, ts0 + pd.Timedelta(hours=1)],
                history_lats=[39.9, 39.91],
                history_lons=[116.4, 116.41],
                history_unix=[1e9, 1e9 + 3600],
                history_time_features=np.zeros((2, 10), dtype=np.float32),
                history_spatial_features=np.zeros((2, 6), dtype=np.float32),
                target_poi_id=(i + 2) % num_pois,
                target_timestamp=ts0 + pd.Timedelta(hours=2),
                target_time_features=np.zeros(8, dtype=np.float32),
                spatial_features=np.zeros(12, dtype=np.float32),
                user_features=np.zeros(6, dtype=np.float32),
                split="train",
            )
        )
    return samples


def test_supported_backbone_names():
    assert "flashback" in SUPPORTED_BACKBONES
    assert "graph_flashback" in SUPPORTED_BACKBONES
    assert "sthgcn" in SUPPORTED_BACKBONES
    assert "sthgcn_seq" in SUPPORTED_BACKBONES


def test_graph_builders_shapes():
    samples = _toy_samples()
    trans = build_transition_graph_from_samples(samples, 10)
    interact = build_interact_graph_from_samples(samples, 3, 10)
    assert trans.shape == (10, 10)
    assert interact.shape == (3, 10)


def test_graph_flashback_forward_and_backward():
    samples = _toy_samples()
    model = SURDGuidedPOIModel(
        num_pois=10,
        num_users=3,
        hidden_dim=16,
        poi_emb_dim=16,
        subset_hidden_dim=16,
        backbone="graph_flashback",
        train_samples=samples,
        share_poi_embedding=True,
    )
    batch = _fake_batch()
    out = model(batch, compute_subsets=False)
    assert out["rec_logits"].shape == (4, 10)
    loss = torch.nn.functional.cross_entropy(out["rec_logits"], batch.target_poi_id)
    loss.backward()
    assert model.backbone.flashback.encoder.weight.grad is not None


def test_sthgcn_full_rejects_nextpoi_factory_without_upstream():
    with pytest.raises((ValueError, ImportError)):
        build_backbone("sthgcn", num_pois=10, num_users=3, hidden_dim=16)


def test_deprecated_sthgcn_seq_still_runs():
    model = SURDGuidedPOIModel(
        num_pois=10,
        num_users=3,
        hidden_dim=16,
        poi_emb_dim=16,
        subset_hidden_dim=16,
        backbone="sthgcn_seq",
        sthgcn_num_layers=1,
        sthgcn_num_heads=4,
        share_poi_embedding=True,
    )
    batch = _fake_batch()
    logits = model.recommend(batch)
    assert logits.shape == (4, 10)


def test_build_backbone_rejects_unknown():
    with pytest.raises(ValueError):
        build_backbone("getnext", num_pois=5, num_users=2, hidden_dim=8)


def test_require_sthgcn_deps_message():
    from surd_guided_poi.data import sthgcn_hypergraph as sh

    try:
        sh.require_sthgcn_deps()
    except ImportError as e:
        assert "requirements-sthgcn" in str(e) or "torch_geometric" in str(e) or "PyG" in str(e)
    else:
        # PyG installed in this env — OK
        pass
