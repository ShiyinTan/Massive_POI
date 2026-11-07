"""Probability calibration helpers for subset estimators (NLL / Brier / ECE)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def nll_and_brier(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape
    p_true = probs[np.arange(n), labels]
    nll = float(-np.log(p_true).mean())
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"nll": nll, "brier": brier}


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def temperature_scale(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """Fit a single temperature on a held-out set (not the test set)."""
    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


def summarize_calibration(
    probs_by_subset: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, p in probs_by_subset.items():
        stats = nll_and_brier(p, labels)
        stats["ece"] = expected_calibration_error(p, labels)
        out[k] = stats
    return out