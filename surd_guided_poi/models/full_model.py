"""SURD-guided next-POI model: Flashback backbone + SURD subset heads."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch
from ..data.feature_builder import SPATIAL_FEATURE_DIM, TIME_FEATURE_DIM
from .backbone_flashback import FlashbackBackbone
from .encoders import HistoryEncoder, SpatialEncoder, TemporalEncoder
from .subset_estimators import SOURCE_NAMES, SubsetEstimators


class SURDGuidedPOIModel(nn.Module):
    """Flashback recommendation + side encoders / subset heads for SURD.

    * ``recommend`` / ``rec_logits`` — Flashback only (inference path).
    * Q1/Q2/Q3 encoders + 7 subset heads — Specific MI / U/S auxiliary training.
    * Q1 may share Flashback's POI embedding when ``share_poi_embedding=True``.
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
            in_dim=TIME_FEATURE_DIM, hidden_dim=hidden_dim, dropout=dropout
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
        """Q1 history, Q2 time, Q3 space — used only for SURD subset heads."""
        h1 = self.history_encoder(batch.history_poi_ids, batch.history_lengths, batch.history_mask)
        h2 = self.temporal_encoder(batch.target_time_features)
        h3 = self.spatial_encoder(batch.spatial_features)
        return [h1, h2, h3]

    def encode(self, batch: NextPOIBatch) -> list[torch.Tensor]:
        """Alias for Phase-A trainer code."""
        return self.encode_sources(batch)

    def forward(self, batch: NextPOIBatch, compute_subsets: bool = True) -> Dict[str, object]:
        rec_logits, _, _ = self.backbone(batch)
        hs = self.encode_sources(batch)
        out: Dict[str, object] = {
            "rec_logits": rec_logits,
            "source_hs": hs,
            "h1": hs[0],
            "h2": hs[1],
            "h3": hs[2],
        }
        if compute_subsets:
            subset_logits = self.subset_estimators(hs)
            out["subset_logits"] = subset_logits
            out["single_source_logits"] = self.subset_estimators.single_source_logits(subset_logits)
            out["multi_source_logits"] = self.subset_estimators.multi_source_logits(subset_logits)
        return out

    def recommend(self, batch: NextPOIBatch) -> torch.Tensor:
        """Inference: Flashback only (no label-dependent SURD weights)."""
        logits, _, _ = self.backbone(batch)
        return logits
