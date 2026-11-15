"""Pretty formatting helpers for STHGCN training / eval logs.

Acc@K ≡ Recall@K for single-label next-POI ranking (Massive-STEPS / paper naming).
Display only — does not change metric computation.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_KS = (1, 5, 10, 20)


def metrics_dict(
    recall_res: Mapping[int, float],
    ndcg_res: Mapping[int, float],
    map_res: Mapping[int, float],
    mrr_res: float,
    *,
    loss: Optional[float] = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Flat metric dict with Acc@K aliases for Recall@K."""
    out: Dict[str, float] = {}
    for k in ks:
        if k in recall_res:
            out[f"Acc@{k}"] = float(recall_res[k])
            out[f"Recall@{k}"] = float(recall_res[k])
        if k in ndcg_res:
            out[f"NDCG@{k}"] = float(ndcg_res[k])
        if k in map_res:
            out[f"MAP@{k}"] = float(map_res[k])
    out["MRR"] = float(mrr_res)
    if loss is not None:
        out["loss"] = float(loss)
    return out


def format_metrics_line(
    recall_res: Mapping[int, float],
    mrr_res: float,
    *,
    ndcg_res: Optional[Mapping[int, float]] = None,
    loss: Optional[float] = None,
    ks: Sequence[int] = DEFAULT_KS,
    precision: int = 4,
) -> str:
    """One-line summary: ``Acc@1=0.1234  Acc@5=0.3456  ...  MRR=0.2100``."""
    parts = []
    for k in ks:
        if k in recall_res:
            parts.append(f"Acc@{k}={float(recall_res[k]):.{precision}f}")
    if ndcg_res is not None:
        for k in ks:
            if k in ndcg_res:
                parts.append(f"NDCG@{k}={float(ndcg_res[k]):.{precision}f}")
    parts.append(f"MRR={float(mrr_res):.{precision}f}")
    if loss is not None:
        parts.append(f"loss={float(loss):.{precision}f}")
    return "  ".join(parts)


def format_metrics_block(
    split: str,
    recall_res: Mapping[int, float],
    ndcg_res: Mapping[int, float],
    map_res: Mapping[int, float],
    mrr_res: float,
    *,
    loss: Optional[float] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    ks: Sequence[int] = DEFAULT_KS,
    precision: int = 4,
    width: int = 72,
) -> str:
    """Aligned multi-line block for console / log files."""
    title_bits = [split]
    if epoch is not None:
        title_bits.append(f"epoch={epoch}")
    if step is not None:
        title_bits.append(f"step={step}")
    title = "  ".join(title_bits)

    headers = [f"Acc@{k}" for k in ks] + [f"NDCG@{k}" for k in ks] + ["MRR"]
    values = []
    for k in ks:
        values.append(f"{float(recall_res[k]):.{precision}f}" if k in recall_res else "-")
    for k in ks:
        values.append(f"{float(ndcg_res[k]):.{precision}f}" if k in ndcg_res else "-")
    values.append(f"{float(mrr_res):.{precision}f}")

    col_w = [max(len(h), len(v)) for h, v in zip(headers, values)]
    header_line = "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_w))
    value_line = "  ".join(f"{v:>{w}}" for v, w in zip(values, col_w))
    rule = "─" * max(width, len(header_line))

    lines = [
        rule,
        f" {title}  (Acc@K ≡ Recall@K)",
    ]
    if loss is not None:
        lines.append(f" loss={float(loss):.{precision}f}")
    # Compact MAP line (secondary)
    map_bits = "  ".join(
        f"MAP@{k}={float(map_res[k]):.{precision}f}" for k in ks if k in map_res
    )
    lines.extend([header_line, value_line])
    if map_bits:
        lines.append(f" {map_bits}")
    lines.append(rule)
    return "\n".join(lines)


def format_epoch_train(epoch: int, total_epochs: int, step: int, loss: float, *, precision: int = 4) -> str:
    width = max(2, len(str(total_epochs)))
    return (
        f"[Training] Epoch {epoch + 1:0{width}d}/{total_epochs}  "
        f"step={step}  avg_loss={float(loss):.{precision}f}"
    )


def metrics_to_serializable(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-friendly copy (floats only)."""
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            out[key] = value
    return out
