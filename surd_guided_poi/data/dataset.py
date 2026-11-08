"""PyTorch Dataset / collate for next-POI samples (Flashback-compatible fields)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_builder import HISTORY_TIME_FEATURE_DIM, pad_history
from .preprocessing import NextPOISample


@dataclass
class NextPOIBatch:
    """Padded training-stage batch / 训练阶段一个 batch（张量）.

    Shapes (B=batch, L=max history in batch)::

        history_poi_ids         [B, L]
        history_lengths         [B]
        history_mask            [B, L]
        history_unix            [B, L]       # Flashback
        history_coords          [B, L, 2]
        history_time_features   [B, L, 10]   # Q2：每步日历+Δt
        target_time_features    [B, 8]       # Q2：目标时刻
        spatial_features        [B, 12]
        user_features           [B, ...]
        target_poi_id           [B]
        user_id                 [B]
        sample_id               [B]
    """

    history_poi_ids: torch.Tensor  # [B, L]
    history_lengths: torch.Tensor  # [B]
    history_mask: torch.Tensor  # [B, L]
    history_unix: torch.Tensor  # [B, L] Flashback temporal kernel
    history_coords: torch.Tensor  # [B, L, 2]
    history_time_features: torch.Tensor  # [B, L, 10]
    target_time_features: torch.Tensor  # [B, 8]
    spatial_features: torch.Tensor
    user_features: torch.Tensor
    target_poi_id: torch.Tensor
    user_id: torch.Tensor
    sample_id: torch.Tensor

    def to(self, device: torch.device) -> "NextPOIBatch":
        return NextPOIBatch(
            **{k: v.to(device) if torch.is_tensor(v) else v for k, v in self.__dict__.items()}
        )


class NextPOIDataset(Dataset):
    """Thin wrapper: index → ``NextPOISample`` (collate builds the batch)."""

    def __init__(self, samples: Sequence[NextPOISample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NextPOISample:
        return self.samples[idx]


def collate_next_poi(batch: List[NextPOISample]) -> NextPOIBatch:
    """List[NextPOISample] → padded ``NextPOIBatch`` for the train loop.

    Example / 示意 B=2::

        samples[0].history_poi_ids = [0, 3]
        samples[1].history_poi_ids = [5]
        → history_poi_ids = tensor([[0, 3], [5, PAD]]), lengths = [2, 1]
    """
    hist, lengths, mask = pad_history(batch)
    B, L = hist.shape
    unix = torch.zeros(B, L, dtype=torch.float32)
    coords = torch.zeros(B, L, 2, dtype=torch.float32)
    hist_time = torch.zeros(B, L, HISTORY_TIME_FEATURE_DIM, dtype=torch.float32)
    for i, s in enumerate(batch):
        h = s.history_unix[-L:]
        unix[i, : len(h)] = torch.tensor(h, dtype=torch.float32)
        ht = np.asarray(s.history_time_features, dtype=np.float32)
        if ht.ndim == 1:
            ht = ht.reshape(1, -1)
        ht = ht[-L:]
        hist_time[i, : ht.shape[0]] = torch.from_numpy(ht)
        la = s.history_lats[-L:]
        lo = s.history_lons[-L:]
        for j, (a, b) in enumerate(zip(la, lo)):
            coords[i, j, 0] = 0.0 if (isinstance(a, float) and np.isnan(a)) else float(a)
            coords[i, j, 1] = 0.0 if (isinstance(b, float) and np.isnan(b)) else float(b)
    return NextPOIBatch(
        history_poi_ids=hist,
        history_lengths=lengths,
        history_mask=mask,
        history_unix=unix,
        history_coords=coords,
        history_time_features=hist_time,
        target_time_features=torch.tensor(
            np.stack([s.target_time_features for s in batch], axis=0), dtype=torch.float32
        ),
        spatial_features=torch.tensor(
            np.stack([s.spatial_features for s in batch], axis=0), dtype=torch.float32
        ),
        user_features=torch.tensor(
            np.stack([s.user_features for s in batch], axis=0), dtype=torch.float32
        ),
        target_poi_id=torch.tensor([s.target_poi_id for s in batch], dtype=torch.long),
        user_id=torch.tensor([s.user_id for s in batch], dtype=torch.long),
        sample_id=torch.tensor([s.sample_id for s in batch], dtype=torch.long),
    )


def batch_to_dict(batch: NextPOIBatch) -> Dict[str, torch.Tensor]:
    return dict(batch.__dict__)