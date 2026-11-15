"""DEPRECATED: local NextPOIBatch Transformer — NOT Massive-STEPS STHGCN.

Kept only as ``model.backbone: sthgcn_seq`` for ablations. Production baseline
``backbone: sthgcn`` must use the full hypergraph path in ``backbone_sthgcn.py``
(PyG NeighborSampler + ``Massive-STEPS/.../model/sthgcn.py``).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch

warnings.warn(
    "backbone_sthgcn_seq / backbone=sthgcn_seq is deprecated and is NOT the "
    "Massive-STEPS hypergraph STHGCN. Use model.backbone=sthgcn instead.",
    DeprecationWarning,
    stacklevel=2,
)


class STHGCNSeqBackbone(nn.Module):
    """Deprecated local sequence Transformer (misleading former ``sthgcn``)."""

    def __init__(
        self,
        num_pois: int,
        num_users: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        use_user_embedding: bool = True,
        use_time_embedding: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = max(1, min(num_heads, hidden_dim))
            while hidden_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        self.num_pois = num_pois
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.padding_idx = num_pois
        self.use_user_embedding = use_user_embedding
        self.use_time_embedding = use_time_embedding

        self.poi_embedding = nn.Embedding(num_pois + 1, hidden_dim, padding_idx=self.padding_idx)
        self.user_embedding = nn.Embedding(num_users, hidden_dim) if use_user_embedding else None
        self.hour_embedding = nn.Embedding(24, hidden_dim) if use_time_embedding else None
        self.weekday_embedding = nn.Embedding(7, hidden_dim) if use_time_embedding else None
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_pois)

    def _time_ids(self, unix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sec = unix.long()
        days = torch.div(sec, 86400, rounding_mode="floor")
        weekday = (days + 3) % 7
        hour = torch.div(sec % 86400, 3600, rounding_mode="floor").clamp(0, 23)
        return hour, weekday

    def forward(
        self,
        batch: NextPOIBatch,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        del h
        x = batch.history_poi_ids
        mask = batch.history_mask
        emb = self.poi_embedding(x)

        if self.use_time_embedding and self.hour_embedding is not None:
            hour, weekday = self._time_ids(batch.history_unix)
            emb = emb + self.hour_embedding(hour) + self.weekday_embedding(weekday)

        if self.use_user_embedding and self.user_embedding is not None:
            u = self.user_embedding(batch.user_id).unsqueeze(1)
            emb = emb + u

        emb = self.pos_dropout(emb)
        key_padding = ~mask
        encoded = self.transformer(emb, src_key_padding_mask=key_padding)

        lengths = batch.history_lengths.clamp(min=1).float().unsqueeze(-1)
        masked = encoded * mask.unsqueeze(-1).float()
        pooled = masked.sum(dim=1) / lengths
        last_logits = self.fc(pooled)
        seq_logits = self.fc(encoded).transpose(0, 1).contiguous()
        return last_logits, seq_logits, None

    @property
    def location_embedding(self) -> nn.Embedding:
        return self.poi_embedding


# Back-compat alias
STHGCNBackbone = STHGCNSeqBackbone
