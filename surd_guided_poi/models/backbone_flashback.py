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

from ..data.dataset import NextPOIBatch
from ..data.massive_steps import add_flashback_dir_to_sys_path

# 显式：先把 Massive-STEPS/Flashback 放进 sys.path，再按官方方式裸导入
add_flashback_dir_to_sys_path()
from network import Flashback, RnnFactory, create_h0_strategy  # type: ignore


class FlashbackBackbone(nn.Module):
    """推荐骨干：官方 Flashback → 每条序列最后一步的 POI logits。

    记号：B=batch，L=历史长度，K=num_pois（POI 个数）。

    输入 ``NextPOIBatch``（我们这边是 batch-first）::

        history_poi_ids  [B, L]
        history_unix     [B, L]
        history_coords   [B, L, 2]
        history_lengths  [B]
        user_id          [B]

    输出::

        last_logits  [B, K]      # 下一站分数（训练/推理主要用这个）
        seq_logits   [L, B, K]   # Flashback 逐步输出（序列维在前）
        h_next       隐状态
    """

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
        # Flashback 要 [L, B]；我们 batch 里是 [B, L]
        x = batch.history_poi_ids.transpose(0, 1).contiguous()  # [L, B]
        t = batch.history_unix.transpose(0, 1).contiguous()  # [L, B]
        s = batch.history_coords.transpose(0, 1).contiguous()  # [L, B, 2]
        lengths = batch.history_lengths  # [B]
        users = batch.user_id  # [B]

        if h is None:
            h = self.initial_hidden(x.size(1), x.device)

        seq_logits, h_next = self.flashback(x, t, s, lengths, h, users)  # [L,B,K]
        last_idx = (lengths - 1).clamp(min=0)  # [B]
        batch_idx = torch.arange(x.size(1), device=x.device)
        last_logits = seq_logits[last_idx, batch_idx, :]  # [B, K]
        return last_logits, seq_logits, h_next

    @property
    def location_embedding(self) -> nn.Embedding:
        """Flashback POI embedding (shared with SURD Q1 when configured)."""
        return self.flashback.encoder
