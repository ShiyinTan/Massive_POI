"""Full SURD-guided next-POI model: shared encoders + rec head + subset heads."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.dataset import NextPOIBatch
from ..data.feature_builder import SPATIAL_FEATURE_DIM, TIME_FEATURE_DIM
from .encoders import HistoryEncoder, SpatialEncoder, TemporalEncoder
from .recommendation import RecommendationHead
from .subset_estimators import SOURCE_NAMES, SubsetEstimators


class SURDGuidedPOIModel(nn.Module):
    def __init__(
        self,
        num_pois: int,
        hidden_dim: int = 64,
        poi_emb_dim: int = 64,
        num_layers_history: int = 1,
        subset_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_pois = num_pois
        self.hidden_dim = hidden_dim
        self.source_names = SOURCE_NAMES

        self.history_encoder = HistoryEncoder(
            num_pois=num_pois,
            emb_dim=poi_emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_history,
            dropout=dropout,
        )
        self.temporal_encoder = TemporalEncoder(
            in_dim=TIME_FEATURE_DIM, hidden_dim=hidden_dim, dropout=dropout
        )
        self.spatial_encoder = SpatialEncoder(
            in_dim=SPATIAL_FEATURE_DIM, hidden_dim=hidden_dim, dropout=dropout
        )
        self.recommendation = RecommendationHead(
            hidden_dim=hidden_dim, num_pois=num_pois, num_sources=3, dropout=dropout
        )
        self.subset_estimators = SubsetEstimators(
            hidden_dim=hidden_dim,
            num_pois=num_pois,
            subset_hidden_dim=subset_hidden_dim,
            dropout=dropout,
        )

    def encode(self, batch: NextPOIBatch) -> list[torch.Tensor]:
        h1 = self.history_encoder(batch.history_poi_ids, batch.history_lengths, batch.history_mask)
        h2 = self.temporal_encoder(batch.target_time_features)
        h3 = self.spatial_encoder(batch.spatial_features)
        return [h1, h2, h3]

    def forward(self, batch: NextPOIBatch, compute_subsets: bool = True) -> Dict[str, object]:
        hs = self.encode(batch)
        rec_logits, h_rec = self.recommendation(hs)
        out: Dict[str, object] = {
            "rec_logits": rec_logits,
            "h_rec": h_rec,
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
        """Inference path: unified recommendation head only (no label-dependent weights)."""
        hs = self.encode(batch)
        logits, _ = self.recommendation(hs)
        return logits