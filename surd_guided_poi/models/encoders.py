"""SURD information-source encoders (Q1/Q2/Q3) on Massive-STEPS fields.

These are *side* encoders used by subset probability heads for Specific MI /
SURD. The main recommendation logits come from the Flashback backbone.
Q1 can optionally share Flashback's location embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Q1: personal preference / history from the POI sequence."""

    def __init__(
        self,
        num_pois: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int | None = None,
        shared_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        if shared_embedding is not None:
            self.emb = shared_embedding
            self._owns_emb = False
        else:
            self.emb = nn.Embedding(num_pois, emb_dim, padding_idx=padding_idx)
            self._owns_emb = True
        self.gru = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, poi_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.emb(poi_ids))
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]
        empty = lengths == 0
        if empty.any():
            h = h.clone()
            h[empty] = 0
        return self.out(h)


class TemporalEncoder(nn.Module):
    """Q2: temporal context from target-time features."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        return self.net(time_features)


class SpatialEncoder(nn.Module):
    """Q3: spatial context from history location statistics."""

    def __init__(self, in_dim: int = 12, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.net(spatial_features)