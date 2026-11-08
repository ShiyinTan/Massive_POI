"""SURD information-source encoders (Q1/Q2/Q3) on Massive-STEPS fields.

These are *side* encoders used by subset probability heads for Specific MI /
SURD. The main recommendation logits come from the Flashback backbone.
Q1 can optionally share Flashback's location embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Q1 历史偏好：POI 序列 → 固定向量。

    输入::

        poi_ids  [B, L]   # 历史 POI id（可含 padding）
        lengths  [B]
        mask     [B, L]

    输出::

        h1  [B, hidden_dim]   # 例 hidden_dim=64 → [B, 64]
    """

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
        x = self.dropout(self.emb(poi_ids))  # [B, L, emb]
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]  # [B, hidden]
        empty = lengths == 0
        if empty.any():
            h = h.clone()
            h[empty] = 0
        return self.out(h)  # [B, hidden]


class TemporalEncoder(nn.Module):
    """Q2 时间：目标时刻特征 → 向量。

    输入 ``time_features`` [B, 8]；输出 ``h2`` [B, hidden_dim]。
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        return self.net(time_features)  # [B, 8] → [B, hidden]


class SpatialEncoder(nn.Module):
    """Q3 空间：历史位置统计特征 → 向量。

    输入 ``spatial_features`` [B, 12]；输出 ``h3`` [B, hidden_dim]。
    """

    def __init__(self, in_dim: int = 12, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        return self.net(spatial_features)  # [B, 12] → [B, hidden]