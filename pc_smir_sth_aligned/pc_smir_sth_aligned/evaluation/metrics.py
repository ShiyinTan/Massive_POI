from __future__ import annotations

from typing import Any, Mapping

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


def _infer_ks(metrics: Mapping[str, Any]) -> list[int]:
    ks: set[int] = set()
    for key in metrics:
        if not isinstance(key, str):
            continue
        low = key.lower()
        for prefix in ("acc@", "recall@", "ndcg@"):
            if low.startswith(prefix):
                suffix = low[len(prefix) :]
                if suffix.isdigit():
                    ks.add(int(suffix))
    return sorted(ks) if ks else [1, 5, 10, 20]


def _lookup(metrics: Mapping[str, Any], *names: str) -> float | None:
    lower = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        if name.lower() in lower:
            return float(lower[name.lower()])
    return None


def format_metrics_table(
    splits: Mapping[str, Mapping[str, Any]],
    *,
    epoch: int | None = None,
    train_loss: float | None = None,
    title: str | None = None,
) -> str:
    """Pretty-print ranking metrics for one or more splits (aligned columns).

    Acc@K is the primary hit-rate column (= Recall@K for single-label next-POI).
    Prefer this over raw JSON dumps in training logs.
    """
    if not splits:
        return ""

    ks: list[int] = []
    seen: set[int] = set()
    for m in splits.values():
        for k in _infer_ks(m):
            if k not in seen:
                seen.add(k)
                ks.append(k)
    ks = sorted(ks)

    headers = ["split", "n", "loss"]
    headers.extend(f"Acc@{k}" for k in ks)
    headers.extend(f"NDCG@{k}" for k in ks)
    headers.extend(["MRR", "Ent"])

    raw_rows: list[list[str]] = []
    for split_name, metrics in splits.items():
        n_v = _lookup(metrics, "n")
        loss_v = _lookup(metrics, "loss")
        mrr_v = _lookup(metrics, "mrr")
        ent_v = _lookup(metrics, "route_entropy", "entropy", "ent")
        cells = [
            str(split_name),
            "-" if n_v is None else str(int(n_v)),
            "-" if loss_v is None else f"{loss_v:.4f}",
        ]
        for k in ks:
            v = _lookup(metrics, f"acc@{k}", f"recall@{k}")
            cells.append("-" if v is None else f"{v:.4f}")
        for k in ks:
            v = _lookup(metrics, f"ndcg@{k}")
            cells.append("-" if v is None else f"{v:.4f}")
        cells.append("-" if mrr_v is None else f"{mrr_v:.4f}")
        cells.append("-" if ent_v is None else f"{ent_v:.4f}")
        raw_rows.append(cells)

    widths = [len(h) for h in headers]
    for row in raw_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: list[str], *, left_first: bool) -> str:
        parts: list[str] = []
        for i, cell in enumerate(cells):
            if i == 0 and left_first:
                parts.append(f"{cell:<{widths[i]}}")
            else:
                parts.append(f"{cell:>{widths[i]}}")
        return "  ".join(parts)

    lines: list[str] = []
    if title is not None:
        lines.append(title)
    elif epoch is not None:
        loss_bit = f"  train_loss={train_loss:.4f}" if train_loss is not None else ""
        lines.append(f"Epoch {epoch}{loss_bit}  (Acc@K = Recall@K)")
    else:
        lines.append("Metrics  (Acc@K = Recall@K)")

    lines.append(_line(headers, left_first=True))
    lines.append("  ".join("-" * w for w in widths))
    for row in raw_rows:
        lines.append(_line(row, left_first=True))
    return "\n".join(lines)
