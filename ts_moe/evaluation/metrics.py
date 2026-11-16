"""Recall@K, NDCG@K, MRR@K, head/mid/tail splits, and routing histograms.

A target missing from the candidate list (two-stage recall miss) counts as
a non-hit. The target is not injected back into the list.

Head / mid / tail are rank terciles of training-target frequency: the least
frequent third of observed POIs is tail. MRR@K is the reciprocal rank when
the rank is at most K, else 0.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model.ts_moe import TSMoE


def popularity_buckets(train_targets: np.ndarray, n_pois: int) -> dict[int, str]:
    counts = np.bincount(np.asarray(train_targets, dtype=np.int64), minlength=n_pois)
    observed = np.arange(1, n_pois)[counts[1:] > 0]
    if len(observed) == 0:
        return {}
    order = observed[np.argsort(counts[observed], kind="mergesort")]
    buckets: dict[int, str] = {}
    n = len(order)
    for rank, poi in enumerate(order):
        if rank < n / 3.0:
            buckets[int(poi)] = "tail"
        elif rank < 2.0 * n / 3.0:
            buckets[int(poi)] = "mid"
        else:
            buckets[int(poi)] = "head"
    return buckets


def _hour_period(hour: int) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _rank_of(candidates: np.ndarray, scores: np.ndarray, target: int) -> int | None:
    order = np.argsort(-scores, kind="mergesort")
    hits = np.where(candidates[order] == target)[0]
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def _accumulate(ranks: list[int | None], ks: list[int]) -> dict:
    n = max(len(ranks), 1)
    recall, ndcg, mrr = {}, {}, {}
    for k in ks:
        hits = 0
        nd = 0.0
        rr = 0.0
        for rank in ranks:
            if rank is not None and rank <= k:
                hits += 1
                nd += 1.0 / np.log2(rank + 1)
                rr += 1.0 / rank
        key = str(k)
        recall[key] = hits / n
        ndcg[key] = nd / n
        mrr[key] = rr / n
    return {"recall": recall, "ndcg": ndcg, "mrr": mrr, "count": len(ranks)}


def _mean_pi(rows: list[np.ndarray], n_experts: int) -> list[float]:
    if not rows:
        return [0.0] * n_experts
    return np.stack(rows, axis=0).mean(axis=0).tolist()


@torch.no_grad()
def evaluate_loader(
    model: TSMoE,
    loader,
    prior_table: torch.Tensor,
    train_targets: np.ndarray,
    config: dict,
) -> dict:
    model.eval()
    eval_cfg = config["eval"]
    ks = [int(k) for k in eval_cfg["ks"]]
    buckets = popularity_buckets(train_targets, model.n_pois)
    grouped: dict[str, list[int | None]] = {"all": [], "head": [], "mid": [], "tail": []}
    route_true: dict[str, list[np.ndarray]] = {name: [] for name in ("head", "mid", "tail")}
    route_pred: dict[str, list[np.ndarray]] = {name: [] for name in ("head", "mid", "tail")}
    route_hour: dict[str, list[np.ndarray]] = {
        name: [] for name in ("night", "morning", "afternoon", "evening")
    }
    expert_names = []
    from ..surd.atoms import expert_names as names_of

    # Names follow the active expert order stored on the model.
    full_names = names_of()
    expert_names = [full_names[i] for i in model.expert_indices]

    for batch in loader:
        batch = {k: v.to(prior_table.device) for k, v in batch.items()}
        out = model.rank_batch(
            batch,
            prior_table,
            strategy=str(eval_cfg["candidate_strategy"]),
            recall_m=int(eval_cfg["recall_m"]),
            chunk_size=int(eval_cfg["chunk_size"]),
        )
        candidates = out["candidates"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        pi = out["pi"].detach().cpu().numpy()
        targets = batch["target"].detach().cpu().numpy()
        hours = batch["hour"].detach().cpu().numpy()
        for i, target in enumerate(targets.tolist()):
            rank = _rank_of(candidates[i], scores[i], int(target))
            grouped["all"].append(rank)
            bucket = buckets.get(int(target))
            if bucket is not None:
                grouped[bucket].append(rank)
            top = int(np.argmax(scores[i]))
            if rank is not None:
                pos = int(np.where(candidates[i] == target)[0][0])
                if bucket is not None:
                    route_true[bucket].append(pi[i, pos])
                route_hour[_hour_period(int(hours[i, -1]))].append(pi[i, pos])
            pred_bucket = buckets.get(int(candidates[i, top]))
            if pred_bucket is not None:
                route_pred[pred_bucket].append(pi[i, top])

    summary = _accumulate(grouped["all"], ks)
    by_bucket = {name: _accumulate(grouped[name], ks) for name in ("head", "mid", "tail")}
    n_experts = model.n_experts
    return {
        "recall": summary["recall"],
        "ndcg": summary["ndcg"],
        "mrr": summary["mrr"],
        "count": summary["count"],
        "by_popularity": by_bucket,
        "routing_true_target": {name: _mean_pi(rows, n_experts) for name, rows in route_true.items()},
        "routing_top1": {name: _mean_pi(rows, n_experts) for name, rows in route_pred.items()},
        "routing_by_hour": {name: _mean_pi(rows, n_experts) for name, rows in route_hour.items()},
        "expert_names": expert_names,
        "tail_recall": by_bucket["tail"]["recall"],
        "long_tail_ndcg": by_bucket["tail"]["ndcg"],
    }
