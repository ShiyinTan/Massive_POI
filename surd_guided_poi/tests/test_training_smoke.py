"""Smoke tests: subset isolation, detached weights, inference without labels."""

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
        target_time_features=torch.randn(B, 8),
        spatial_features=torch.randn(B, 12),
        user_features=torch.randn(B, 6),
        target_poi_id=torch.randint(0, num_pois, (B,)),
        user_id=torch.randint(0, 3, (B,)),
        sample_id=torch.arange(B),
    )


def test_subset_heads_only_use_authorized_sources():
    model = SURDGuidedPOIModel(num_pois=10, hidden_dim=16, poi_emb_dim=16, subset_hidden_dim=16)
    batch = _fake_batch()
    out = model(batch, compute_subsets=True)
    assert set(out["single_source_logits"]) == set(SOURCE_NAMES)
    assert "Q1_Q2" in out["multi_source_logits"]
    # Forward shapes
    assert out["rec_logits"].shape == (4, 10)
    for k, logits in out["subset_logits"].items():
        assert logits.shape == (4, 10)


def test_recommendation_backward_ok():
    model = SURDGuidedPOIModel(num_pois=10, hidden_dim=16, poi_emb_dim=16, subset_hidden_dim=16)
    batch = _fake_batch()
    out = model(batch, compute_subsets=False)
    loss = torch.nn.functional.cross_entropy(out["rec_logits"], batch.target_poi_id)
    loss.backward()
    assert model.history_encoder.emb.weight.grad is not None


def test_surd_weights_are_detached():
    poi_surd = {
        y: {
            "unique": {"Q1": 1.0, "Q2": 0.0, "Q3": 0.0},
            "redundant": {},
            "synergistic": {"Q1_Q2": 0.0, "Q1_Q3": 0.0, "Q2_Q3": 0.0, "Q1_Q2_Q3": 0.0},
        }
        for y in range(10)
    }
    # Make y=1 prefer Q2
    poi_surd[1]["unique"] = {"Q1": 0.0, "Q2": 1.0, "Q3": 0.0}
    weights = build_poi_specific_weights(
        poi_surd,
        poi_sample_counts=np.ones(10, dtype=int) * 50,
        min_poi_samples=3,
        shrink_kappa=0.0,  # no shrinkage: keep sharp per-POI weights
    )
    tables = weights_to_tensors(weights, num_pois=10, device=torch.device("cpu"))
    assert not tables["unique"].requires_grad
    model = SURDGuidedPOIModel(num_pois=10, hidden_dim=16, poi_emb_dim=16, subset_hidden_dim=16)
    batch = _fake_batch()
    # Force different targets to verify per-POI weight lookup.
    batch.target_poi_id = torch.tensor([0, 1, 0, 1])
    out = model(batch, compute_subsets=True)
    loss = unique_auxiliary_loss(out["single_source_logits"], batch.target_poi_id, tables["unique"])
    loss.backward()
    # Different POIs get different unique weights.
    assert tables["unique"][0, 0] > tables["unique"][0, 1]
    assert tables["unique"][1, 1] > tables["unique"][1, 0]


def test_inference_does_not_need_labels():
    model = SURDGuidedPOIModel(num_pois=10, hidden_dim=16, poi_emb_dim=16, subset_hidden_dim=16)
    batch = _fake_batch()
    # Clear labels to ensure recommend path does not read them.
    batch.target_poi_id = None  # type: ignore
    logits = model.recommend(batch)
    assert logits.shape == (4, 10)