"""Smoke tests: Flashback backbone, subset isolation, detached weights, inference."""

from __future__ import annotations

import numpy as np
import torch

from surd_guided_poi.data.dataset import NextPOIBatch
from surd_guided_poi.information.weights import build_poi_specific_weights, weights_to_tensors
from surd_guided_poi.losses.surd_auxiliary_loss import unique_auxiliary_loss
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.models.subset_estimators import SOURCE_NAMES


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


def _model(num_pois=10, num_users=3, hidden=16):
    return SURDGuidedPOIModel(
        num_pois=num_pois,
        num_users=num_users,
        hidden_dim=hidden,
        poi_emb_dim=hidden,
        subset_hidden_dim=hidden,
        share_poi_embedding=True,
    )


def test_subset_heads_only_use_authorized_sources():
    model = _model()
    batch = _fake_batch()
    out = model(batch, compute_subsets=True)
    assert set(out["single_source_logits"]) == set(SOURCE_NAMES)
    assert "Q1_Q2" in out["multi_source_logits"]
    assert out["rec_logits"].shape == (4, 10)
    for k, logits in out["subset_logits"].items():
        assert logits.shape == (4, 10)


def test_flashback_recommendation_backward_ok():
    model = _model()
    batch = _fake_batch()
    out = model(batch, compute_subsets=False)
    loss = torch.nn.functional.cross_entropy(out["rec_logits"], batch.target_poi_id)
    loss.backward()
    assert model.backbone.flashback.encoder.weight.grad is not None


def test_surd_weights_are_detached():
    poi_surd = {
        y: {
            "unique": {"Q1": 1.0, "Q2": 0.0, "Q3": 0.0},
            "redundant": {},
            "synergistic": {"Q1_Q2": 0.0, "Q1_Q3": 0.0, "Q2_Q3": 0.0, "Q1_Q2_Q3": 0.0},
        }
        for y in range(10)
    }
    poi_surd[1]["unique"] = {"Q1": 0.0, "Q2": 1.0, "Q3": 0.0}
    weights = build_poi_specific_weights(
        poi_surd,
        poi_sample_counts=np.ones(10, dtype=int) * 50,
        min_poi_samples=3,
        shrink_kappa=0.0,
    )
    tables = weights_to_tensors(weights, num_pois=10, device=torch.device("cpu"))
    assert not tables["unique"].requires_grad
    model = _model()
    batch = _fake_batch()
    batch.target_poi_id = torch.tensor([0, 1, 0, 1])
    out = model(batch, compute_subsets=True)
    loss = unique_auxiliary_loss(out["single_source_logits"], batch.target_poi_id, tables["unique"])
    loss.backward()
    assert tables["unique"][0, 0] > tables["unique"][0, 1]
    assert tables["unique"][1, 1] > tables["unique"][1, 0]


def test_inference_does_not_need_labels():
    model = _model()
    batch = _fake_batch()
    batch.target_poi_id = None  # type: ignore
    logits = model.recommend(batch)
    assert logits.shape == (4, 10)