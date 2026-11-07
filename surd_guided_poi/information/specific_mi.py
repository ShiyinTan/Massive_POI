"""POI-specific specific mutual information from subset probability estimates.

\tilde{i}(y; Q_A) = E_{q_A | Y=y}[ log P(Y=y | Q_A) / P(Y=y) ]

Estimated on an independent estimation/calibration split (not the same
samples used to fit the subset heads in the current Phase A round).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.subset_estimators import SUBSET_NAMES, subset_key


def _log_base(units: str):
    if units == "nats":
        return np.log, torch.log
    if units == "bits":
        return np.log2, lambda x: torch.log2(x)
    raise ValueError(f"Unknown units: {units}")


def estimate_specific_mi(
    subset_probs: Mapping[str, np.ndarray],
    labels: np.ndarray,
    poi_prior: np.ndarray,
    eps: float = 1e-12,
    units: str = "nats",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Parameters
    ----------
    subset_probs:
        Mapping subset_key -> array [N, K] of P_A(Y | Q_A) on estimation samples.
    labels:
        Target POI indices, shape [N].
    poi_prior:
        Training prior P(Y=y) = N_y / N, shape [K] (not 1/K).
    """
    np_log, _ = _log_base(units)
    labels = np.asarray(labels, dtype=np.int64)
    prior = np.clip(np.asarray(poi_prior, dtype=np.float64), eps, None)
    K = prior.shape[0]
    N = labels.shape[0]
    counts = np.bincount(labels, minlength=K).astype(np.int64)

    specific: Dict[str, np.ndarray] = {}
    for key, probs in subset_probs.items():
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (N, K):
            raise ValueError(f"{key}: expected probs shape {(N, K)}, got {p.shape}")
        p = np.clip(p, eps, 1.0)
        # Per-sample log-ratio for the *true* class only.
        p_true = p[np.arange(N), labels]
        log_ratio = np_log(p_true) - np_log(prior[labels])
        mi = np.zeros(K, dtype=np.float64)
        for y in np.unique(labels):
            mask = labels == y
            mi[y] = float(log_ratio[mask].mean())
        # POIs never observed on estimation split keep 0 (marked unreliable via counts).
        specific[key] = mi
    return specific, counts


def enforce_subset_monotonicity(
    specific_mi: Mapping[str, np.ndarray],
    subsets: Sequence[Tuple[str, ...]] = SUBSET_NAMES,
) -> Dict[str, np.ndarray]:
    """
    SURD requires i(y; Q_A) not to decrease when A grows.

    Follows the official ``surd.neural`` monotone projection: replace each
    value by the max over subsets of A (empty subset treated as 0).
    """
    out: Dict[str, np.ndarray] = {}
    for subset in sorted(subsets, key=len):
        key = subset_key(subset)
        if key not in specific_mi:
            continue
        v = np.maximum(np.asarray(specific_mi[key], dtype=np.float64), 0.0)
        for smaller, smaller_v in out.items():
            smaller_set = set(smaller.split("_"))
            if smaller_set < set(subset):
                v = np.maximum(v, smaller_v)
        out[key] = v
    # Preserve original key order for callers.
    return {subset_key(s): out[subset_key(s)] for s in subsets if subset_key(s) in out}


@torch.no_grad()
def collect_subset_probs(
    model,
    loader,
    device: torch.device,
    temperature: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run subset heads over a loader; return probs per subset and labels."""
    model.eval()
    all_logits: Dict[str, list] = {}
    labels: list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, compute_subsets=True)
        subset_logits: Dict[str, torch.Tensor] = out["subset_logits"]  # type: ignore
        for k, logits in subset_logits.items():
            all_logits.setdefault(k, []).append(logits.detach().cpu())
        labels.append(batch.target_poi_id.detach().cpu())
    label_arr = torch.cat(labels, dim=0).numpy()
    probs: Dict[str, np.ndarray] = {}
    for k, chunks in all_logits.items():
        logits = torch.cat(chunks, dim=0)
        if temperature and k in temperature:
            logits = logits / float(temperature[k])
        probs[k] = F.softmax(logits, dim=-1).numpy().astype(np.float64)
    return probs, label_arr