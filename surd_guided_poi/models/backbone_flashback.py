"""Flashback next-POI backbone (Massive-STEPS/Flashback), batch-adapted.

Imports the official Flashback network shipped under
``Massive-STEPS/Flashback/network.py`` and adapts our ``NextPOIBatch``
(batch-first histories) to Flashback's sequence-first tensors.

Why Flashback:
* Already adapted to Massive-STEPS official train/val/test CSVs in this repo
* Well-known sequential next-POI baseline (RNN + spatiotemporal decay)
* Clean surface for wrapping SURD guidance on top
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.massive_steps import ensure_flashback_on_path
from ..data.dataset import NextPOIBatch


def _import_flashback():
    ensure_flashback_on_path()
    from network import Flashback, RnnFactory, create_h0_strategy  # type: ignore

    return Flashback, RnnFactory, create_h0_strategy


class FlashbackBackbone(nn.Module):
    """Recommendation backbone: Massive-STEPS Flashback → last-step POI logits."""

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
        Flashback, RnnFactory, create_h0_strategy = _import_flashback()
        self.num_pois = num_pois
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.padding_idx = num_pois
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        f_t = lambda delta_t, user_len: ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t)
        )
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s * self.lambda_s))

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
        # Flashback expects [seq, batch]
        x = batch.history_poi_ids.transpose(0, 1).contiguous()
        t = batch.history_unix.transpose(0, 1).contiguous()
        s = batch.history_coords.transpose(0, 1).contiguous()
        lengths = batch.history_lengths
        users = batch.user_id
        if h is None:
            h = self.initial_hidden(x.size(1), x.device)
        seq_logits, h_next = self.flashback(x, t, s, lengths, h, users)
        # Gather last valid timestep per sequence.
        last_idx = (lengths - 1).clamp(min=0)  # [B]
        b_idx = torch.arange(x.size(1), device=x.device)
        last_logits = seq_logits[last_idx, b_idx, :]
        return last_logits, seq_logits, h_next

    @property
    def location_embedding(self) -> nn.Embedding:
        return self.flashback.encoder