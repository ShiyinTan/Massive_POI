"""Cross-fit POI→expert SURD prior.

For fold ``k``, specific MI and the SURD atoms are computed only on trails
whose fold is not ``k`` (spec §9). Validation and test use one prior fit on
all training trails. The result is cached as ``R_SURD[y] ∈ R^{21}`` and is
not differentiated through the recommender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset.build import Example
from .atoms import atoms_to_expert_mass, expert_names, select_experts
from .decomposition import assign_surd_from_specific_mi
from .estimators import estimate_specific_mi_discrete, estimate_specific_mi_neural
from .specific_mi import enforce_subset_monotonicity, routing_distribution


@dataclass
class PriorBundle:
    """``fold_priors[k]`` is the table training fold ``k`` is allowed to read."""

    fold_priors: np.ndarray  # [F, K, 21]
    full_prior: np.ndarray  # [K, 21]
    expert_names: list[str]

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            fold_priors=self.fold_priors,
            full_prior=self.full_prior,
            expert_names=np.array(self.expert_names),
        )

    @classmethod
    def load(cls, path) -> "PriorBundle":
        with np.load(path, allow_pickle=False) as payload:
            names = [str(x) for x in payload["expert_names"].tolist()]
            return cls(
                fold_priors=payload["fold_priors"],
                full_prior=payload["full_prior"],
                expert_names=names,
            )


def _raw_mass_from_specific(specific: dict[str, np.ndarray], n_pois: int) -> np.ndarray:
    projected = enforce_subset_monotonicity(specific)
    atoms = assign_surd_from_specific_mi(projected)
    mass = np.zeros((n_pois, 21), dtype=np.float64)
    if len(atoms) != n_pois:
        raise RuntimeError(f"SURD returned {len(atoms)} states, expected {n_pois}")
    for y, poi_atoms in enumerate(atoms):
        mass[y] = atoms_to_expert_mass(poi_atoms)
    mass[0] = 0.0
    return mass


def estimate_raw_expert_mass(
    examples: list[Example],
    n_pois: int,
    n_users: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[n_pois, 21]`` raw (pre-shrink) expert mass and MI counts."""
    kind = str(cfg.get("estimator", "discrete"))
    if kind == "discrete":
        specific, counts = estimate_specific_mi_discrete(
            examples,
            n_pois,
            alpha=float(cfg.get("alpha", 1.0)),
            n_bins=int(cfg.get("n_bins", 8)),
        )
    elif kind == "neural":
        neural_cfg = cfg.get("neural", {})
        device_name = cfg.get("device", "cpu")
        import torch

        device = torch.device(device_name)
        specific, counts = estimate_specific_mi_neural(
            examples,
            n_pois,
            n_users,
            d_model=int(neural_cfg.get("d_model", 32)),
            epochs=int(neural_cfg.get("epochs", 5)),
            batch_size=int(neural_cfg.get("batch_size", 256)),
            lr=float(neural_cfg.get("lr", 1e-3)),
            neg_samples=int(neural_cfg.get("neg_samples", 64)),
            full_softmax_limit=int(neural_cfg.get("full_softmax_limit", 4096)),
            eval_fraction=float(cfg.get("eval_fraction", 0.2)),
            max_len=int(cfg.get("max_len", 50)),
            seed=int(cfg.get("seed", 0)),
            device=device,
        )
    else:
        raise ValueError(f"unknown SURD estimator {kind!r} (expected discrete or neural)")
    return _raw_mass_from_specific(specific, n_pois), counts


def _finalize(raw: np.ndarray, counts: np.ndarray, cfg: dict) -> np.ndarray:
    return routing_distribution(
        raw,
        counts,
        kappa=float(cfg.get("kappa", 10.0)),
        eps=float(cfg.get("eps", 1e-8)),
    )


def estimate_cross_fit_priors(
    examples: list[Example],
    n_pois: int,
    n_users: int,
    cfg: dict,
) -> PriorBundle:
    """Build the per-fold tables and the full-train table used at val/test."""
    n_folds = int(cfg.get("n_folds", 5))
    present = {ex.fold for ex in examples}
    if any(fold < 0 or fold >= n_folds for fold in present):
        raise ValueError(f"example folds must lie in [0, {n_folds}), got {sorted(present)}")

    tables = []
    for fold in range(n_folds):
        complement = [ex for ex in examples if ex.fold != fold]
        fold_cfg = dict(cfg)
        fold_cfg["seed"] = int(cfg.get("seed", 0)) + fold + 1
        if not complement:
            tables.append(np.full((n_pois, 21), 1.0 / 21))
            continue
        raw, counts = estimate_raw_expert_mass(complement, n_pois, n_users, fold_cfg)
        tables.append(_finalize(raw, counts, fold_cfg))

    full_cfg = dict(cfg)
    raw_full, counts_full = estimate_raw_expert_mass(examples, n_pois, n_users, full_cfg)
    full = _finalize(raw_full, counts_full, full_cfg)
    return PriorBundle(
        fold_priors=np.stack(tables, axis=0),
        full_prior=full,
        expert_names=expert_names(),
    )


def slice_prior(table: np.ndarray, indices: list[int], eps: float = 1e-8) -> np.ndarray:
    """Restrict a 21-d routing prior to the active experts and renormalize.

    Rows whose active mass is 0 become uniform on the active set.
    """
    part = np.asarray(table, dtype=np.float64)[..., indices]
    total = part.sum(axis=-1, keepdims=True)
    n_exp = len(indices)
    uniform = np.full(part.shape[:-1] + (n_exp,), 1.0 / n_exp)
    safe = np.divide(part, np.maximum(total, eps))
    return np.where(total > 0, safe, uniform)


def active_expert_index(variant: int = 21, groups=None) -> tuple[list[int], list]:
    return select_experts(variant=variant, groups=groups)
