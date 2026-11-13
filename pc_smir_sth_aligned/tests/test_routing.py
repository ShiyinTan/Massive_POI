from __future__ import annotations

import torch

from pc_smir_sth_aligned.models.pcsmir import PCSMIR


def _batch(bsz=2, L=5, C=8, n_pois=20, n_users=5, surd_dim=10):
    return {
        "history_poi_ids": torch.randint(2, n_pois, (bsz, L)),
        "history_categories": torch.randint(2, 6, (bsz, L)),
        "history_hours": torch.randint(0, 24, (bsz, L)),
        "history_dows": torch.randint(0, 7, (bsz, L)),
        "history_time_gaps": torch.rand(bsz, L),
        "history_cells": torch.randint(0, 100, (bsz, L)),
        "history_coordinates": torch.randn(bsz, L, 2),
        "history_mask": torch.ones(bsz, L),
        "query_time_features": torch.randn(bsz, 7),
        "stable_user_features": torch.randn(bsz, 8),
        "source_available": torch.ones(bsz, 4),
        "user_id": torch.randint(2, n_users, (bsz,)),
        "last_poi_id": torch.randint(2, n_pois, (bsz,)),
        "candidate_poi_ids": torch.randint(2, n_pois, (bsz, C)),
        "candidate_metadata": torch.randn(bsz, C, 5),
        "candidate_surd": torch.randn(bsz, C, surd_dim),
        "candidate_surd_valid": torch.ones(bsz, C),
        "candidate_mask": torch.ones(bsz, C),
        "target_index": torch.zeros(bsz, dtype=torch.long),
    }


def test_routes_sum_to_one():
    surd_dim = 10
    model = PCSMIR(n_pois=20, n_cats=6, n_users=5, d=16, surd_dim=surd_dim, dropout=0.0)
    batch = _batch(surd_dim=surd_dim)
    scores, routes = model.score(batch)
    sums = routes.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert scores.shape == batch["candidate_poi_ids"].shape


def test_unavailable_experts_zero_weight():
    surd_dim = 10
    model = PCSMIR(n_pois=20, n_cats=6, n_users=5, d=16, surd_dim=surd_dim, dropout=0.0)
    batch = _batch(surd_dim=surd_dim)
    batch["source_available"][:, 0] = 0  # disable Q1
    _, routes = model.score(batch)
    # Expert 0 is single-source U0; should be ~0
    assert torch.all(routes[:, :, 0] < 1e-5)


def test_personalization_only_ablation():
    surd_dim = 10
    model = PCSMIR(
        n_pois=20,
        n_cats=6,
        n_users=5,
        d=16,
        surd_dim=surd_dim,
        dropout=0.0,
        ablation="personalization_only",
    )
    batch = _batch(surd_dim=surd_dim)
    scores, routes = model.score(batch)
    assert scores.shape == batch["candidate_poi_ids"].shape
    assert torch.all(routes == 0)