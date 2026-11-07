"""Probability heads for every non-empty subset of the 3 information sources."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

SOURCE_NAMES = ("Q1", "Q2", "Q3")


def all_subsets(names: Sequence[str] = SOURCE_NAMES) -> List[Tuple[str, ...]]:
    """Non-empty subsets ordered by size then lexicographically (SURD key order)."""
    out: List[Tuple[str, ...]] = []
    n = len(names)
    for k in range(1, n + 1):
        out.extend(combinations(names, k))
    return out


SUBSET_NAMES: List[Tuple[str, ...]] = all_subsets()


def subset_key(subset: Iterable[str]) -> str:
    return "_".join(subset)


class SubsetHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_pois: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubsetEstimators(nn.Module):
    """
    Seven subset probability heads for m=3.

    Each head only receives the concatenations of its authorized source
    representations — no bypass to other sources.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_pois: int,
        subset_hidden_dim: int = 64,
        dropout: float = 0.1,
        source_names: Sequence[str] = SOURCE_NAMES,
    ):
        super().__init__()
        self.source_names = tuple(source_names)
        self.subsets = all_subsets(self.source_names)
        self.heads = nn.ModuleDict()
        name_to_idx = {n: i for i, n in enumerate(self.source_names)}
        self._subset_indices: Dict[str, List[int]] = {}
        for subset in self.subsets:
            key = subset_key(subset)
            in_dim = hidden_dim * len(subset)
            self.heads[key] = SubsetHead(in_dim, subset_hidden_dim, num_pois, dropout)
            self._subset_indices[key] = [name_to_idx[s] for s in subset]

    def forward(self, source_hs: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return logits dict keyed by 'Q1', 'Q1_Q2', ..."""
        if len(source_hs) != len(self.source_names):
            raise ValueError("source_hs length must match number of sources")
        out: Dict[str, torch.Tensor] = {}
        for key, idxs in self._subset_indices.items():
            parts = [source_hs[i] for i in idxs]
            x = torch.cat(parts, dim=-1)
            out[key] = self.heads[key](x)
        return out

    def single_source_logits(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: logits[k] for k in self.source_names}

    def multi_source_logits(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in logits.items() if "_" in k}