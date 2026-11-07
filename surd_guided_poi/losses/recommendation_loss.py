"""Standard next-POI cross-entropy (no class balancing in v1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recommendation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)