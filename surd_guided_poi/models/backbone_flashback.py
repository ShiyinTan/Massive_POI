"""Flashback next-POI backbone — thin adapter over Massive-STEPS/Flashback.

Source of truth: ``Massive-STEPS/Flashback/network.py`` (fixed repo path).
This module only converts our batch-first ``NextPOIBatch`` to Flashback's
sequence-first tensors and returns the last valid timestep's logits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Registers Massive-STEPS/Flashback on sys.path (fixed relative layout).
from ..data.massive_steps import FLASHBACK_DIR  # noqa: F401
from ..data.dataset import NextPOIBatch

# Bare imports — same style as Massive-STEPS/Flashback/train.py
from network import Flashback, RnnFactory, create_h0_strategy  # type: ignore


class FlashbackBackbone(nn.Module):
    """Recommendation backbone: Flashback RNN → last-step POI logits."""

    def __init__(
        self,
        num_pois: int,
        num_users: int,
        hidden_dim: int = 64,
        rnn_type: str = "gru",
        lambda_t: float = 0.1,
        lambda_s: float = 100.0,
    ):
        super().__init__()
        self.num_pois = num_pois
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.padding_idx = num_pois
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        # Spatiotemporal decay kernels (Flashback defaults).
        def f_t(delta_t, user_len):
            day = delta_t / 86400
            return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
                -(day * self.lambda_t)
            )

        def f_s(delta_s, user_len):
            return torch.exp(-(delta_s * self.lambda_s))

        rnn_factory = RnnFactory(rnn_type)
        self.is_lstm = rnn_factory.is_lstm()
        self.flashback = Flashback(
            num_pois,
            num_users,
            hidden_dim,
            f_t,
            f_s,
            rnn_factory,
            padding_idx=self.padding_idx,
        )
        self._h0_strategy = create_h0_strategy(hidden_dim, self.is_lstm)

    def initial_hidden(self, batch_size: int, device: torch.device):
        return self._h0_strategy.on_init(batch_size, device)

    def forward(
        self,
        batch: NextPOIBatch,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        last_logits : [B, num_pois]
        seq_logits : [L, B, num_pois]
        h_next : hidden state
        """
        # Flashback expects [seq, batch]; our batch stores [batch, seq].
        x = batch.history_poi_ids.transpose(0, 1).contiguous()
        t = batch.history_unix.transpose(0, 1).contiguous()
        s = batch.history_coords.transpose(0, 1).contiguous()
        lengths = batch.history_lengths
        users = batch.user_id

        if h is None:
            h = self.initial_hidden(x.size(1), x.device)

        seq_logits, h_next = self.flashback(x, t, s, lengths, h, users)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(1), device=x.device)
        last_logits = seq_logits[last_idx, batch_idx, :]
        return last_logits, seq_logits, h_next

    @property
    def location_embedding(self) -> nn.Embedding:
        """Flashback POI embedding (shared with SURD Q1 when configured)."""
        return self.flashback.encoder
