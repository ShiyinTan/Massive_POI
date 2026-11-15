"""STHGCN sequence baseline adapted to ``NextPOIBatch``.

Full hypergraph STHGCN (``Massive-STEPS/Spatio-Temporal-Hypergraph-Model``)
needs ``torch_geometric`` + NeighborSampler hypergraph batches — not drop-in
compatible with our official-CSV ``NextPOIBatch``. This module ports the
**SequentialTransformer** idea from that package (POI / time embeddings +
``nn.TransformerEncoder`` + linear POI head) so ``model.backbone: sthgcn``
trains and evaluates inside ``surd_guided_poi`` without PyG.

For the full hypergraph model, run the upstream entrypoint::

    Massive-STEPS/Spatio-Temporal-Hypergraph-Model/run.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch


class STHGCNBackbone(nn.Module):
    """Sequence Transformer next-POI backbone (STHGCN sequential baseline).

    I/O matches Flashback adapters: ``forward(batch) → (last_logits [B,K], …)``.
    """

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
            # Keep heads valid without surprising callers; bump heads down.
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
        weekday = (days + 3) % 7  # pandas dayofweek (Mon=0)
        hour = torch.div(sec % 86400, 3600, rounding_mode="floor").clamp(0, 23)
        return hour, weekday

    def forward(
        self,
        batch: NextPOIBatch,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        del h  # no RNN state
        x = batch.history_poi_ids  # [B, L]
        mask = batch.history_mask  # True = valid
        emb = self.poi_embedding(x)

        if self.use_time_embedding and self.hour_embedding is not None:
            hour, weekday = self._time_ids(batch.history_unix)
            emb = emb + self.hour_embedding(hour) + self.weekday_embedding(weekday)

        if self.use_user_embedding and self.user_embedding is not None:
            u = self.user_embedding(batch.user_id).unsqueeze(1)  # [B,1,d]
            emb = emb + u

        emb = self.pos_dropout(emb)
        # Transformer key padding: True = ignore
        key_padding = ~mask
        encoded = self.transformer(emb, src_key_padding_mask=key_padding)  # [B,L,d]

        # Masked mean over valid steps (SequentialTransformer-style pool).
        lengths = batch.history_lengths.clamp(min=1).float().unsqueeze(-1)  # [B,1]
        masked = encoded * mask.unsqueeze(-1).float()
        pooled = masked.sum(dim=1) / lengths  # [B,d]
        last_logits = self.fc(pooled)  # [B,K]

        # Seq logits: project every step (optional; last-step used elsewhere).
        seq_logits = self.fc(encoded).transpose(0, 1).contiguous()  # [L,B,K]
        return last_logits, seq_logits, None

    @property
    def location_embedding(self) -> nn.Embedding:
        return self.poi_embedding
