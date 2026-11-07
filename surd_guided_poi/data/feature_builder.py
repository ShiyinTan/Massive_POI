"""Shared helpers for tensorizing sample fields."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from .preprocessing import NextPOISample


TIME_FEATURE_DIM = 8
SPATIAL_FEATURE_DIM = 12
USER_FEATURE_DIM = 6


def pad_history(
    samples: Sequence[NextPOISample],
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return history_poi_ids [B,L], lengths [B], mask [B,L]."""
    lengths = [len(s.history_poi_ids) for s in samples]
    L = max(lengths) if max_len is None else max_len
    B = len(samples)
    ids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, s in enumerate(samples):
        h = s.history_poi_ids[-L:]
        ids[i, : len(h)] = torch.tensor(h, dtype=torch.long)
        mask[i, : len(h)] = True
    length_t = torch.tensor([min(l, L) for l in lengths], dtype=torch.long)
    return ids, length_t, mask