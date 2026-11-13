from __future__ import annotations

import torch
import torch.nn.functional as F


def full_softmax_ce(scores: torch.Tensor, target_index: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Cross-entropy over candidate axis."""
    logits = scores / max(temperature, 1e-6)
    return F.cross_entropy(logits, target_index)


def bpr_loss(scores: torch.Tensor, target_index: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    bsz, n_cand = scores.shape
    row = torch.arange(bsz, device=scores.device)
    pos = scores[row, target_index].unsqueeze(1)
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask[row, target_index] = False
    if candidate_mask is not None:
        neg_mask = neg_mask & (candidate_mask > 0)
    # Mean softplus over negatives.
    loss = F.softplus(-(pos - scores))
    loss = loss.masked_fill(~neg_mask, 0.0)
    denom = neg_mask.float().sum(dim=1).clamp_min(1.0)
    return (loss.sum(dim=1) / denom).mean()


def routing_entropy(routes: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    ent = -(routes.clamp_min(1e-8) * routes.clamp_min(1e-8).log()).sum(dim=-1)
    if candidate_mask is not None:
        ent = ent * candidate_mask
        return ent.sum() / candidate_mask.sum().clamp_min(1.0)
    return ent.mean()