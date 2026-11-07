"""Legacy concat+MLP recommendation head (kept for ablations / tests only).

Production recommendation uses ``FlashbackBackbone`` in ``full_model.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RecommendationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_pois: int, num_sources: int = 3, dropout: float = 0.1):
        super().__init__()
        in_dim = hidden_dim * num_sources
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_pois)

    def forward(self, source_hs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat(source_hs, dim=-1)
        h_rec = self.fusion(h)
        logits = self.classifier(h_rec)
        return logits, h_rec