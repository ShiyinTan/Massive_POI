"""Next-POI ranking metrics: Recall@K, NDCG@K, MRR."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def _topk_hits(logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int]):
    max_k = max(ks)
    topk = torch.topk(logits, k=min(max_k, logits.size(-1)), dim=-1).indices
    return topk


def recall_at_k(topk: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    hits = (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).float()
    return float(hits.mean().item())


def mrr_at_k(topk: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    eq = topk[:, :k] == targets.unsqueeze(1)
    # rank = first hit index + 1
    has = eq.any(dim=1)
    ranks = eq.float().argmax(dim=1) + 1
    rr = torch.zeros_like(ranks, dtype=torch.float)
    rr[has] = 1.0 / ranks[has].float()
    return float(rr.mean().item())


def ndcg_at_k(topk: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    eq = topk[:, :k] == targets.unsqueeze(1)
    has = eq.any(dim=1)
    ranks = eq.float().argmax(dim=1) + 1
    dcg = torch.zeros(targets.size(0), dtype=torch.float, device=targets.device)
    dcg[has] = 1.0 / torch.log2(ranks[has].float() + 1.0)
    return float(dcg.mean().item())


@torch.no_grad()
def evaluate_ranking(
    model,
    loader,
    device: torch.device,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        logits = model.recommend(batch)
        all_logits.append(logits.cpu())
        all_targets.append(batch.target_poi_id.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    topk = _topk_hits(logits, targets, ks)
    metrics = {}
    for k in ks:
        kk = min(k, topk.size(1))
        metrics[f"recall@{k}"] = recall_at_k(topk, targets, kk)
        metrics[f"ndcg@{k}"] = ndcg_at_k(topk, targets, kk)
        metrics[f"mrr@{k}"] = mrr_at_k(topk, targets, kk)
    return metrics