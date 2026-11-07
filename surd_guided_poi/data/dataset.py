"""PyTorch Dataset / collate for next-POI samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_builder import SPATIAL_FEATURE_DIM, TIME_FEATURE_DIM, USER_FEATURE_DIM, pad_history
from .preprocessing import NextPOISample


@dataclass
class NextPOIBatch:
    history_poi_ids: torch.Tensor
    history_lengths: torch.Tensor
    history_mask: torch.Tensor
    target_time_features: torch.Tensor
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
    def __init__(self, samples: Sequence[NextPOISample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NextPOISample:
        return self.samples[idx]


def collate_next_poi(batch: List[NextPOISample]) -> NextPOIBatch:
    hist, lengths, mask = pad_history(batch)
    return NextPOIBatch(
        history_poi_ids=hist,
        history_lengths=lengths,
        history_mask=mask,
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