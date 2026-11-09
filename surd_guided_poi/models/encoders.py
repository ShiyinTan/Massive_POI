"""SURD information-source encoders (Q1/Q2/Q3) on Massive-STEPS fields.

These are *side* encoders used by subset probability heads for Specific MI /
SURD. The main recommendation logits come from the Flashback backbone.
Q1 can optionally share Flashback's location embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Q1 历史偏好：POI 序列 → 固定向量。

    输入::

        poi_ids  [B, L]   # 历史 POI id（可含 padding）
        lengths  [B]
        mask     [B, L]

    输出::

        h1  [B, hidden_dim]   # 例 hidden_dim=64 → [B, 64]
    """

    def __init__(
        self,
        num_pois: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int | None = None,
        shared_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        if shared_embedding is not None:
            self.emb = shared_embedding
            self._owns_emb = False
        else:
            self.emb = nn.Embedding(num_pois, emb_dim, padding_idx=padding_idx)
            self._owns_emb = True
        self.gru = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, poi_ids: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.emb(poi_ids))  # [B, L, emb]
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]  # [B, hidden]
        empty = lengths == 0
        if empty.any():
            h = h.clone()
            h[empty] = 0
        return self.out(h)  # [B, hidden]


class TemporalEncoder(nn.Module):
    """Q2 时间：历史时间序列 + 目标时刻 → 融合向量。

    输入张量::

        history_time_features  [B, L, 10]
        lengths / mask         [B] / [B, L]
        target_time_features   [B, 8]

    目标 8 维（``_time_features``）::

        0–1 小时 sin/cos；2–3 星期 sin/cos；4 周末；5–6 月份 sin/cos；7 hour/24

    历史每步 10 维（``_history_time_features``）::

        0–7 同上（该历史步日历）；8 log1p(Δ小时相对目标)；9 Δ小时/24

    流程：每步 Linear→GRU → h_hist；目标 MLP → h_tgt；concat 融合 → h2 [B, d]。
    """

    def __init__(
        self,
        hist_in_dim: int = 10,
        tgt_in_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hist_in = nn.Linear(hist_in_dim, hidden_dim)
        self.hist_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.tgt_net = nn.Sequential(
            nn.Linear(tgt_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        history_time_features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        target_time_features: torch.Tensor,
    ) -> torch.Tensor:
        # [B, L, D_t] → [B, L, d]
        x = self.dropout(self.hist_in(history_time_features))
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.hist_gru(packed)
        h_hist = h_n[-1]  # [B, d]
        empty = lengths == 0
        if empty.any():
            h_hist = h_hist.clone()
            h_hist[empty] = 0

        h_tgt = self.tgt_net(target_time_features)  # [B, d]
        return self.fuse(torch.cat([h_hist, h_tgt], dim=-1))  # [B, d]


class SpatialEncoder(nn.Module):
    """Q3 空间：历史位置序列 + 12 维汇总 → 融合向量（不含目标点）。

    输入张量::

        history_spatial_features  [B, L, 6]
        lengths / mask            [B] / [B, L]
        spatial_features          [B, 12]

    历史每步 6 维（``_history_spatial_features``）::

        0–1 lat/lon 归一化；2 缺坐标；3 log1p(距上一步 km)
        4 log1p(距上一步小时)；5 窗口内相对位置

    汇总 12 维（``_spatial_features``）::

        0–1 末点 lat/lon；2–3 均值；4–5 标准差
        6–7 log1p(平均/最大步距)；8 log1p(位移)；9 log1p(跨度小时)
        10 步数；11 末点缺坐标

    流程：每步 Linear→GRU → h_seq；汇总 MLP → h_sum；concat 融合 → h3 [B, d]。
    """

    def __init__(
        self,
        hist_in_dim: int = 6,
        summary_in_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hist_in = nn.Linear(hist_in_dim, hidden_dim)
        self.hist_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.summary_net = nn.Sequential(
            nn.Linear(summary_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        history_spatial_features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        # [B, L, D_s] → [B, L, d]
        x = self.dropout(self.hist_in(history_spatial_features))
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.hist_gru(packed)
        h_seq = h_n[-1]  # [B, d]
        empty = lengths == 0
        if empty.any():
            h_seq = h_seq.clone()
            h_seq[empty] = 0

        h_sum = self.summary_net(spatial_features)  # [B, d]
        return self.fuse(torch.cat([h_seq, h_sum], dim=-1))  # [B, d]
