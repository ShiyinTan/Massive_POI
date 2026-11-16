"""Graph-Flashback backbone — adapter over Massive-STEPS/Graph-Flashback.

Source of truth: ``Massive-STEPS/Graph-Flashback/network.py`` (Flashback + GCN
on POI transition / user–POI interact graphs). Same ``NextPOIBatch`` I/O as
``FlashbackBackbone``; builds hour-of-week ``t_slot`` from unix timestamps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix

from ..data.dataset import NextPOIBatch
from ..data.massive_steps import import_graph_flashback_network

_gf = import_graph_flashback_network()
Flashback = _gf.Flashback
RnnFactory = _gf.RnnFactory
create_h0_strategy = _gf.create_h0_strategy


def unix_to_timeslot(unix: torch.Tensor) -> torch.Tensor:
    """Massive-STEPS Graph-Flashback timeslot: ``hour + weekday * 24`` ∈ [0, 167].

    Matches ``pd.to_datetime(..., unit='s').dt.hour + dt.weekday * 24`` (Mon=0).
    """
    sec = unix.long()
    days = torch.div(sec, 86400, rounding_mode="floor")
    weekday = (days + 3) % 7  # Unix epoch day 0 = Thursday → pandas weekday 3
    hour = torch.div(sec % 86400, 3600, rounding_mode="floor")
    return (hour + weekday * 24).float()


class GraphFlashbackBackbone(nn.Module):
    """推荐骨干：官方 Graph-Flashback → 每条序列最后一步的 POI logits。

    输入 ``NextPOIBatch``（batch-first）→ 输出 ``last_logits [B,K]``。
    """

    def __init__(
        self,
        num_pois: int,
        num_users: int,
        transition_graph: coo_matrix,
        interact_graph: csr_matrix,
        hidden_dim: int = 64,
        rnn_type: str = "gru",
        lambda_t: float = 0.1,
        lambda_s: float = 100.0,
        lambda_loc: float = 1.0,
        lambda_user: float = 1.0,
        use_weight: bool = False,
        use_graph_user: bool = False,
        use_spatial_graph: bool = False,
        spatial_graph: Optional[coo_matrix] = None,
        friend_graph: Optional[coo_matrix] = None,
    ):
        super().__init__()
        self.num_pois = num_pois
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.padding_idx = num_pois
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

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
            lambda_loc,
            lambda_user,
            use_weight,
            transition_graph,
            spatial_graph,
            friend_graph,
            use_graph_user,
            use_spatial_graph,
            interact_graph,
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
        x = batch.history_poi_ids.transpose(0, 1).contiguous()  # [L, B]
        t = batch.history_unix.transpose(0, 1).contiguous()  # [L, B]
        t_slot = unix_to_timeslot(batch.history_unix).transpose(0, 1).contiguous()
        s = batch.history_coords.transpose(0, 1).contiguous()  # [L, B, 2]
        lengths = batch.history_lengths
        users = batch.user_id

        if h is None:
            h = self.initial_hidden(x.size(1), x.device)

        seq_logits, h_next = self.flashback(x, t, t_slot, s, lengths, h, users)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(1), device=x.device)
        last_logits = seq_logits[last_idx, batch_idx, :]
        return last_logits, seq_logits, h_next

    @property
    def location_embedding(self) -> nn.Embedding:
        return self.flashback.encoder
