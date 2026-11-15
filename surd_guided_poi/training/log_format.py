"""Readable one-line / block formatters for training logs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


def _metric_key_order(keys: Sequence[str]) -> list[str]:
    """Prefer recall@k then ndcg@k in ascending k, then leftover keys."""
    recall, ndcg, other = [], [], []
    for k in keys:
        if k.startswith("recall@"):
            recall.append(k)
        elif k.startswith("ndcg@"):
            ndcg.append(k)
        else:
            other.append(k)

    def _k_num(name: str) -> int:
        try:
            return int(name.split("@", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(recall, key=_k_num) + sorted(ndcg, key=_k_num) + sorted(other)


def short_metric_name(name: str) -> str:
    return name.replace("recall@", "R@").replace("ndcg@", "N@")


def format_metrics(metrics: Mapping[str, Any], *, precision: int = 4) -> str:
    """``R@1=0.0123  R@5=0.0456  N@5=0.0312``."""
    parts = []
    for key in _metric_key_order(list(metrics.keys())):
        val = metrics[key]
        if isinstance(val, (int, float)):
            parts.append(f"{short_metric_name(key)}={float(val):.{precision}f}")
        else:
            parts.append(f"{short_metric_name(key)}={val}")
    return "  ".join(parts)


def format_train_losses(stats: Mapping[str, float], *, precision: int = 4) -> str:
    """``loss=0.8912  rec=0.7011  U=0.1023  S=0.0878``."""
    order = ("loss", "rec", "u", "s")
    labels = {"loss": "loss", "rec": "rec", "u": "U", "s": "S"}
    parts = []
    for key in order:
        if key in stats:
            parts.append(f"{labels[key]}={float(stats[key]):.{precision}f}")
    for key, val in stats.items():
        if key not in order:
            parts.append(f"{key}={float(val):.{precision}f}")
    return "  ".join(parts)


def format_global_u(unique: Mapping[str, float], *, precision: int = 3) -> str:
    """``Q1=0.410  Q2=0.330  Q3=0.260``."""
    return "  ".join(f"{k}={float(v):.{precision}f}" for k, v in unique.items())


def log_block(
    title: str,
    lines: Sequence[str],
    *,
    width: int = 64,
) -> str:
    """Multi-line block with a light rule."""
    bar = "─" * width
    body = "\n".join(f"  {line}" for line in lines if line)
    return f"{bar}\n {title}\n{body}\n{bar}"


def epoch_title(phase: str, epoch: int, total: Optional[int] = None) -> str:
    if total is None:
        return f"{phase}  {epoch}"
    width = max(2, len(str(total)))
    return f"{phase}  {epoch:0{width}d}/{total}"
