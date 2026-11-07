"""Three information-source encoders (Q1 history, Q2 temporal, Q3 spatial)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Q1: POI embedding + GRU over the user history sequence."""

    def __init__(
        self,
        num_pois: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_pois, emb_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, poi_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # poi_ids: [B, L], lengths: [B], mask: [B, L]
        x = self.dropout(self.emb(poi_ids))
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]
        # Zero-out empty histories (should be rare given min_history_len).
        empty = lengths == 0
        if empty.any():
            h = h.clone()
            h[empty] = 0
        return self.out(h)


class TemporalEncoder(nn.Module):
    """Q2: hour / weekday / periodic time features -> MLP."""

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
    """Q3: history location statistics / distances -> MLP."""

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