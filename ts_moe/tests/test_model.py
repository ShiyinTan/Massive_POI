"""Forward shapes, per-candidate priors, and the sampled objective."""

from __future__ import annotations

import torch

from ts_moe.dataset.build import frames_to_examples, make_collate
from ts_moe.dataset.synthetic import make_synthetic_frames
from ts_moe.model.ts_moe import TSMoE
from ts_moe.training.config import load_config
from ts_moe.training.losses import total_loss
from ts_moe.training.loop import sample_candidates


def _tiny():
    frames = make_synthetic_frames(n_users=4, n_venues=5, trails_per_split=(3, 1, 1), length=4, seed=1)
    vocab, splits = frames_to_examples(frames, max_history=4, min_history=1, n_folds=2, seed=1)
    config = load_config(None)
    config["model"].update(
        {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.0,
            "max_len": 4,
            "expert_variant": 21,
            "top_k": 4,
            "routing_mode": "surd",
            "guidance_scale": 1.0,
        }
    )
    model = TSMoE(vocab, config).eval()
    batch = make_collate(vocab, 4)(splits["train"][:2])
    return model, batch, vocab, config


def test_forward_shapes_and_topk_gate_is_a_distribution():
    model, batch, _, _ = _tiny()
    candidates = sample_candidates(batch["target"], model.n_pois, n_neg=3)
    prior = torch.full((candidates.size(0), candidates.size(1), model.n_experts), 1.0 / model.n_experts)
    out = model(batch, candidates, prior)
    assert out["scores"].shape == candidates.shape
    assert out["pi"].shape == (candidates.size(0), candidates.size(1), 21)
    assert torch.allclose(out["pi"].sum(-1), torch.ones(candidates.shape), atol=1e-5)
    assert torch.isfinite(out["scores"]).all()


def test_scores_do_not_depend_on_the_label_tensor():
    model, batch, vocab, _ = _tiny()
    candidates = sample_candidates(batch["target"], model.n_pois, n_neg=3)
    prior = torch.rand(candidates.size(0), candidates.size(1), model.n_experts)
    prior = prior / prior.sum(-1, keepdim=True)
    first = model(batch, candidates, prior)["scores"]
    leaked = dict(batch)
    leaked["target"] = torch.randint(1, vocab.n_pois, batch["target"].shape)
    second = model(leaked, candidates, prior)["scores"]
    assert torch.allclose(first, second)


def test_each_candidate_prior_changes_its_own_route():
    model, batch, _, _ = _tiny()
    candidates = torch.stack(
        [torch.arange(1, model.n_pois) for _ in range(batch["user"].size(0))],
        dim=0,
    )
    eye = torch.eye(model.n_experts)
    with torch.no_grad():
        model.router.prior_map.weight.copy_(eye)
    prior_a = torch.zeros(candidates.size(0), candidates.size(1), model.n_experts)
    prior_b = torch.zeros_like(prior_a)
    prior_a[..., 0] = 1
    prior_b[..., 1] = 1
    pi_a = model(batch, candidates, prior_a)["pi"]
    pi_b = model(batch, candidates, prior_b)["pi"]
    assert not torch.allclose(pi_a, pi_b)
    # The two candidates in one row can carry different priors.
    mixed = prior_a.clone()
    mixed[:, 0, :] = prior_b[:, 0, :]
    pi_mixed = model(batch, candidates, mixed)["pi"]
    assert not torch.allclose(pi_mixed[:, 0], pi_mixed[:, 1])


def test_expert_variants_match_the_spec_counts():
    frames = make_synthetic_frames(n_users=3, n_venues=4, trails_per_split=(2, 1, 1), length=3, seed=0)
    vocab, _ = frames_to_examples(frames, max_history=3, min_history=1, n_folds=2, seed=0)
    config = load_config(None)
    config["model"].update({"d_model": 32, "n_heads": 4, "n_layers": 1, "dropout": 0.0, "max_len": 3})
    for variant, expected, redundancy, synergy in (
        (4, 4, 0, 0),
        (10, 10, 6, 0),
        (21, 21, 6, 11),
    ):
        config["model"]["expert_variant"] = variant
        config["model"]["expert_groups"] = None
        model = TSMoE(vocab, config)
        assert model.n_experts == expected
        assert len(model.redundancy) == redundancy
        assert len(model.synergy) == synergy
    config["model"]["expert_groups"] = ["synergy"]
    synergy_only = TSMoE(vocab, config)
    assert synergy_only.n_experts == 11


def test_objective_backward_reaches_experts_not_the_prior():
    model, batch, _, config = _tiny()
    model.train()
    candidates = sample_candidates(batch["target"], model.n_pois, n_neg=3)
    prior = torch.full((candidates.size(0), candidates.size(1), model.n_experts), 1.0 / model.n_experts)
    prior.requires_grad_(True)
    out = model(batch, candidates, prior)
    losses = total_loss(
        out["scores"],
        out["pi"],
        prior,
        out["shared_loss"],
        mode="surd",
        lambda_route=config["loss"]["lambda_route"],
        lambda_balance=config["loss"]["lambda_balance"],
        lambda_shared=1.0,
    )
    losses["loss"].backward()
    assert prior.grad is None
    assert model.router.gate.weight.grad is not None
    assert model.redundancy["UP"].shared[0].weight.grad is not None
    assert torch.isfinite(losses["loss"]).all()
