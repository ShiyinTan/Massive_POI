"""Long-tail POI evaluation by training-frequency buckets."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .ranking_metrics import mrr_at_k, ndcg_at_k, recall_at_k


def frequency_buckets(
    train_counts: np.ndarray,
    cutoffs: Sequence[float] = (0.2, 0.5),
) -> Dict[str, np.ndarray]:
    """
    Assign each POI to head / mid / tail by cumulative check-in mass.

    POIs sorted by descending frequency; head = top mass until cutoffs[0],
    mid until cutoffs[1], rest = tail.
    """
    counts = np.asarray(train_counts, dtype=np.float64)
    order = np.argsort(-counts)
    total = counts.sum()
    cum = np.cumsum(counts[order]) / max(total, 1.0)
    bucket = np.full(len(counts), "tail", dtype=object)
    c0, c1 = cutoffs[0], cutoffs[1]
    for rank, poi in enumerate(order):
        if counts[poi] <= 0:
            bucket[poi] = "unseen_in_train"
            continue
        if cum[rank] <= c0:
            bucket[poi] = "head"
        elif cum[rank] <= c1:
            bucket[poi] = "mid"
        else:
            bucket[poi] = "tail"
    return {"poi_bucket": bucket, "order": order, "cum_mass": cum}


@torch.no_grad()
def evaluate_by_frequency(
    logits: torch.Tensor,
    targets: torch.Tensor,
    poi_bucket: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Dict[str, float]]:
    topk = torch.topk(logits, k=min(max(ks), logits.size(-1)), dim=-1).indices
    out: Dict[str, Dict[str, float]] = {}
    for name in ("head", "mid", "tail"):
        mask = torch.tensor(
            [poi_bucket[int(t)] == name for t in targets.tolist()],
            dtype=torch.bool,
        )
        if mask.sum() == 0:
            continue
        sub_top = topk[mask]
        sub_tgt = targets[mask]
        metrics = {"n": int(mask.sum().item())}
        for k in ks:
            kk = min(k, sub_top.size(1))
            metrics[f"recall@{k}"] = recall_at_k(sub_top, sub_tgt, kk)
            metrics[f"ndcg@{k}"] = ndcg_at_k(sub_top, sub_tgt, kk)
            metrics[f"mrr@{k}"] = mrr_at_k(sub_top, sub_tgt, kk)
        out[name] = metrics
    return out