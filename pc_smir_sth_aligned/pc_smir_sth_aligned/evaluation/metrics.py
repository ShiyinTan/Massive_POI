from __future__ import annotations

import numpy as np
import torch


def rank_metrics_from_scores(
    scores: torch.Tensor,
    target_index: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Event-averaged Recall@K / Acc@K / NDCG@K / MRR.

    Acc@K is an alias of Recall@K (single-label next-POI ranking), matching
    Massive-STEPS / STHGCN reporting.
    """
    if candidate_mask is not None:
        scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min / 4)
    # ranks: 0 = best
    # Handle ties by argsort stable descending
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    # position of target
    matches = order == target_index.unsqueeze(1)
    ranks = matches.float().argmax(dim=1)  # first True
    # If somehow missing, rank = n_cand
    present = matches.any(dim=1)
    ranks = torch.where(present, ranks, torch.full_like(ranks, scores.size(1)))

    ranks_np = ranks.detach().cpu().numpy()
    out = {}
    for k in ks:
        hits = (ranks_np < k).astype(np.float64)
        recall_k = float(hits.mean())
        out[f"recall@{k}"] = recall_k
        out[f"acc@{k}"] = recall_k
        dcg = np.where(ranks_np < k, 1.0 / np.log2(ranks_np + 2), 0.0)
        out[f"ndcg@{k}"] = float(dcg.mean())
    out["mrr"] = float((1.0 / (ranks_np + 1)).mean())
    out["n"] = int(len(ranks_np))
    return out


def merge_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    total_n = sum(d.get("n", 0) for d in dicts)
    keys = [k for k in dicts[0] if k != "n"]
    out = {"n": float(total_n)}
    for k in keys:
        num = sum(d[k] * d.get("n", 0) for d in dicts)
        out[k] = float(num / max(total_n, 1))
    return out
