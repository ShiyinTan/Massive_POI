"""SURD atom assignment, expert-mass mapping, and cross-fit priors."""

from __future__ import annotations

import numpy as np

from ts_moe.dataset.build import Example
from ts_moe.surd.atoms import ALL_SUBSETS, EXPERTS, PAIRS, SOURCES, atoms_to_expert_mass, subset_key
from ts_moe.surd.decomposition import assign_surd_from_specific_mi
from ts_moe.surd.prior import estimate_cross_fit_priors, estimate_raw_expert_mass
from ts_moe.surd.specific_mi import enforce_subset_monotonicity, routing_distribution


def _blank(k: int = 1) -> dict[str, np.ndarray]:
    return {subset_key(subset): np.zeros(k) for subset in ALL_SUBSETS}


def _atom_total(atoms: dict) -> float:
    return (
        sum(atoms["unique"].values())
        + sum(atoms["redundant"].values())
        + sum(atoms["synergistic"].values())
    )


def test_unique_information_is_booked_to_user():
    specific = _blank()
    specific["U"][:] = 1.0
    projected = enforce_subset_monotonicity(specific)
    atoms = assign_surd_from_specific_mi(projected)[0]
    assert atoms["unique"]["U"] > 0.9
    assert atoms["unique"]["P"] < 1e-8
    assert abs(_atom_total(atoms) - float(projected["U_P_T_L"][0])) < 1e-8


def test_pairwise_redundancy_when_two_sources_match():
    specific = _blank()
    specific["U"][:] = 1.0
    specific["P"][:] = 1.0
    projected = enforce_subset_monotonicity(specific)
    atoms = assign_surd_from_specific_mi(projected)[0]
    assert sum(atoms["unique"].values()) < 1e-8
    assert abs(sum(atoms["redundant"].values()) - 1.0) < 1e-8
    mass = atoms_to_expert_mass(atoms)
    # Pool left after the zero-information sources is the pair UP.
    assert abs(mass[4 + PAIRS.index(("U", "P"))] - 1.0) < 1e-8


def test_synergy_only_when_the_joint_carries_information():
    specific = _blank()
    specific["U_P_T_L"][:] = 1.0
    projected = enforce_subset_monotonicity(specific)
    atoms = assign_surd_from_specific_mi(projected)[0]
    assert abs(atoms["synergistic"].get("U_P_T_L", 0.0) - 1.0) < 1e-8
    assert sum(atoms["unique"].values()) < 1e-8
    assert sum(atoms["redundant"].values()) < 1e-8


def test_higher_order_redundancy_is_split_across_pairs():
    atoms = {
        "unique": {name: 0.0 for name in SOURCES},
        "redundant": {"U_P_T": 3.0},
        "synergistic": {},
    }
    mass = atoms_to_expert_mass(atoms)
    assert abs(mass[4 + PAIRS.index(("U", "P"))] - 1.0) < 1e-8
    assert abs(mass[4 + PAIRS.index(("U", "T"))] - 1.0) < 1e-8
    assert abs(mass[4 + PAIRS.index(("P", "T"))] - 1.0) < 1e-8
    assert abs(mass[4:10].sum() - 3.0) < 1e-8
    assert abs(mass[10:].sum()) < 1e-8


def test_routing_distribution_clips_negatives_and_shrinks_unseen():
    raw = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, -5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    counts = np.array([0.0, 100.0, 0.0])
    prior = routing_distribution(raw, counts, kappa=10.0)
    assert np.allclose(prior.sum(-1), 1.0)
    assert (prior >= 0).all()
    # Negative component is dropped, so POI 1 prefers expert 0.
    assert prior[1, 0] > prior[1, 1]
    # Unseen POI 2 copies the global profile, which is dominated by POI 1.
    assert np.allclose(prior[2], prior[1])
    assert len(EXPERTS) == 21


def _examples(fold: int, target: int, n: int = 6) -> list[Example]:
    rows = []
    for i in range(n):
        rows.append(
            Example(
                user=fold,
                poi=[1],
                hour=[8],
                dow=[1],
                lat=[fold * 0.1],
                lon=[0.2],
                target=target,
                fold=fold,
                trail_id=f"f{fold}-{i}",
            )
        )
    return rows


def test_cross_fit_prior_ignores_the_held_out_fold():
    cfg = {"estimator": "discrete", "n_folds": 2, "alpha": 0.5, "kappa": 1.0, "n_bins": 4, "eps": 1e-8, "seed": 0}
    original = _examples(0, target=1) + _examples(1, target=2)
    changed = _examples(0, target=3) + _examples(1, target=2)
    first = estimate_cross_fit_priors(original, n_pois=4, n_users=2, cfg=cfg)
    second = estimate_cross_fit_priors(changed, n_pois=4, n_users=2, cfg=cfg)
    # Fold 0's table is estimated only on fold 1, whose labels did not change.
    assert np.allclose(first.fold_priors[0], second.fold_priors[0])
    raw_first, _ = estimate_raw_expert_mass(original, 4, 2, cfg)
    raw_second, _ = estimate_raw_expert_mass(changed, 4, 2, cfg)
    # Moving fold-0 labels from POI 1 to POI 3 moves raw atom mass. The shrunk
    # routing table can stay numerically equal when every observed POI has the
    # same shape, so the leakage check is the held-out table above.
    assert raw_first[1].sum() > raw_second[1].sum()
    assert raw_second[3].sum() > raw_first[3].sum()
    assert np.allclose(first.fold_priors.sum(-1), 1.0)


def test_neural_estimator_returns_a_distribution():
    cfg = {
        "estimator": "neural",
        "n_folds": 2,
        "alpha": 1.0,
        "kappa": 1.0,
        "eps": 1e-8,
        "seed": 0,
        "eval_fraction": 0.25,
        "max_len": 4,
        "device": "cpu",
        "neural": {"d_model": 8, "epochs": 1, "batch_size": 8, "lr": 1e-3, "neg_samples": 2, "full_softmax_limit": 32},
    }
    examples = _examples(0, target=1, n=8) + _examples(1, target=2, n=8)
    bundle = estimate_cross_fit_priors(examples, n_pois=4, n_users=2, cfg=cfg)
    assert bundle.full_prior.shape == (4, 21)
    assert np.isfinite(bundle.full_prior).all()
    assert np.allclose(bundle.fold_priors.sum(-1), 1.0)
