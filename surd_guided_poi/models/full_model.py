"""SURD-guided next-POI model: Flashback backbone + SURD subset heads."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch
from ..data.feature_builder import (
    HISTORY_TIME_FEATURE_DIM,
    SPATIAL_FEATURE_DIM,
    TIME_FEATURE_DIM,
)
from .backbone_flashback import FlashbackBackbone
from .encoders import HistoryEncoder, SpatialEncoder, TemporalEncoder
from .subset_estimators import SOURCE_NAMES, SubsetEstimators


class SURDGuidedPOIModel(nn.Module):
    """Flashback 推荐 + SURD 侧路（Q1/Q2/Q3 + 7 子集头）。

    记号：B=batch，L=历史长，K=num_pois，d=hidden_dim。

    ``forward(batch)`` 主要输出::

        rec_logits              [B, K]     # Flashback 下一站分数
        h1,h2,h3 / source_hs    各 [B, d]  # 三个信息源向量
        subset_logits["Q1"] …   各 [B, K]  # 7 个头
        single_source_logits    Q1/Q2/Q3 → [B, K]
        multi_source_logits     Q1_Q2…   → [B, K]

    ``recommend(batch)`` → 仅 ``[B, K]``（推理只用 Flashback，不查 SURD 权重）。
    """

    def __init__(
        self,
        num_pois: int,
        num_users: int,
        hidden_dim: int = 64,
        poi_emb_dim: int = 64,
        num_layers_history: int = 1,
        subset_hidden_dim: int = 64,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        lambda_t: float = 0.1,
        lambda_s: float = 100.0,
        share_poi_embedding: bool = True,
    ):
        super().__init__()
        self.num_pois = num_pois
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.source_names = SOURCE_NAMES

        self.backbone = FlashbackBackbone(
            num_pois=num_pois,
            num_users=num_users,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
        )

        # Flashback embedding is (num_pois + 1) with padding_idx = num_pois.
        shared_emb = self.backbone.location_embedding if share_poi_embedding else None
        self.history_encoder = HistoryEncoder(
            num_pois=num_pois + 1,
            emb_dim=poi_emb_dim if shared_emb is None else shared_emb.embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_history,
            dropout=dropout,
            padding_idx=num_pois,
            shared_embedding=shared_emb,
        )
        self.temporal_encoder = TemporalEncoder(
            hist_in_dim=HISTORY_TIME_FEATURE_DIM,
            tgt_in_dim=TIME_FEATURE_DIM,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.spatial_encoder = SpatialEncoder(
            in_dim=SPATIAL_FEATURE_DIM, hidden_dim=hidden_dim, dropout=dropout
        )
        self.subset_estimators = SubsetEstimators(
            hidden_dim=hidden_dim,
            num_pois=num_pois,
            subset_hidden_dim=subset_hidden_dim,
            dropout=dropout,
        )

    def encode_sources(self, batch: NextPOIBatch) -> list[torch.Tensor]:
        """侧路三源：各返回 [B, d]，只给 SURD 子集头用。"""
        h1 = self.history_encoder(batch.history_poi_ids, batch.history_lengths, batch.history_mask)
        # Q2：历史时间序列 [B,L,10] + 目标时刻 [B,8] → h2 [B,d]
        h2 = self.temporal_encoder(
            batch.history_time_features,
            batch.history_lengths,
            batch.history_mask,
            batch.target_time_features,
        )
        h3 = self.spatial_encoder(batch.spatial_features)
        return [h1, h2, h3]  # 三个 [B, d]

    def encode(self, batch: NextPOIBatch) -> list[torch.Tensor]:
        """Alias for Phase-A trainer code."""
        return self.encode_sources(batch)

    def forward(self, batch: NextPOIBatch, compute_subsets: bool = True) -> Dict[str, object]:
        rec_logits, _, _ = self.backbone(batch)  # [B, K]
        hs = self.encode_sources(batch)  # [h1,h2,h3] 各 [B, d]
        out: Dict[str, object] = {
            "rec_logits": rec_logits,
            "source_hs": hs,
            "h1": hs[0],
            "h2": hs[1],
            "h3": hs[2],
        }
        if compute_subsets:
            subset_logits = self.subset_estimators(hs)  # 7 × [B, K]
            out["subset_logits"] = subset_logits
            out["single_source_logits"] = self.subset_estimators.single_source_logits(subset_logits)
            out["multi_source_logits"] = self.subset_estimators.multi_source_logits(subset_logits)
        return out

    def recommend(self, batch: NextPOIBatch) -> torch.Tensor:
        """推理：只要 Flashback logits [B, K]。"""
        logits, _, _ = self.backbone(batch)
        return logits