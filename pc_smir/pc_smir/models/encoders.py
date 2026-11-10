"""Leakage-safe source encoders E1–E4 and candidate encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class PersonalEncoder(nn.Module):
    """E1: GRU over POI/category history + attention pooling + gate fusion."""

    def __init__(self, n_pois: int, n_cats: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.poi_emb = nn.Embedding(n_pois, d, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, d, padding_idx=0)
        self.gap_proj = nn.Linear(1, d)
        self.in_proj = nn.Linear(3 * d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.attn = nn.Linear(d, 1)
        self.gate = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout))
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, poi_ids, cat_ids, gaps, mask):
        # poi_ids: [B,L]
        x = torch.cat(
            [self.poi_emb(poi_ids), self.cat_emb(cat_ids), self.gap_proj(gaps.unsqueeze(-1))],
            dim=-1,
        )
        x = torch.tanh(self.in_proj(x)) * mask.unsqueeze(-1)
        packed_out, h_n = self.gru(x)
        # Attention pooling over non-pad.
        scores = self.attn(packed_out).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = (attn.unsqueeze(-1) * packed_out).sum(dim=1)
        short = h_n[-1]
        g = torch.sigmoid(self.gate(torch.cat([short, pooled], dim=-1)))
        h = g * short + (1 - g) * pooled
        return self.out(h)


class TemporalEncoder(nn.Module):
    """E2: query-time context + historical hour/dow GRU."""

    def __init__(self, d: int, query_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        self.hour_emb = nn.Embedding(24, d // 2)
        self.dow_emb = nn.Embedding(7, d // 2)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.query_mlp = nn.Sequential(nn.Linear(query_dim, d), nn.GELU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, hours, dows, mask, query_feats):
        x = torch.cat([self.hour_emb(hours.clamp(0, 23)), self.dow_emb(dows.clamp(0, 6))], dim=-1)
        x = x * mask.unsqueeze(-1)
        _, h_n = self.gru(x)
        hist = h_n[-1]
        q = self.query_mlp(query_feats)
        return self.fuse(torch.cat([hist, q], dim=-1))


class SpatialEncoder(nn.Module):
    """E3: cell embeddings + coordinate GRU + displacement features."""

    def __init__(self, n_cells: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.cell_emb = nn.Embedding(n_cells, d, padding_idx=0)
        self.coord_proj = nn.Linear(2, d)
        self.in_proj = nn.Linear(2 * d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, cells, coords, mask):
        # coords already normalized; displacements from consecutive observed steps.
        cell_e = self.cell_emb(cells.clamp(min=0, max=self.cell_emb.num_embeddings - 1))
        coord_e = self.coord_proj(coords)
        x = torch.tanh(self.in_proj(torch.cat([cell_e, coord_e], dim=-1))) * mask.unsqueeze(-1)
        _, h_n = self.gru(x)
        # last observed displacement magnitude proxy
        delta = coords[:, 1:] - coords[:, :-1]
        mag = delta.norm(dim=-1)
        mag = mag * mask[:, 1:]
        last_mag = mag.sum(dim=1, keepdim=True) / mask[:, 1:].sum(dim=1, keepdim=True).clamp_min(1.0)
        # mean coord of observed history
        mean_c = (coords * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        # use last coord x/y summary via mean + last mag
        extra = torch.cat([last_mag, mean_c[:, :1]], dim=-1)
        return self.out(torch.cat([h_n[-1], extra], dim=-1))


class StableUserEncoder(nn.Module):
    """E4: MLP over prefix-only stable statistics."""

    def __init__(self, in_dim: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, feats, available):
        h = self.mlp(feats)
        return available.unsqueeze(-1) * h + (1 - available.unsqueeze(-1)) * self.null


class CandidateEncoder(nn.Module):
    def __init__(self, n_pois: int, meta_dim: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.poi_emb = nn.Embedding(n_pois, d, padding_idx=0)
        self.meta_mlp = nn.Sequential(nn.Linear(meta_dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, candidate_ids, candidate_meta):
        # [B,C], [B,C,F]
        e = self.poi_emb(candidate_ids)
        m = self.meta_mlp(candidate_meta)
        return self.out(torch.cat([e, m], dim=-1)), e, m


class SourceEncoders(nn.Module):
    def __init__(self, n_pois: int, n_cats: int, d: int, stable_dim: int = 8, n_cells: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.e1 = PersonalEncoder(n_pois, n_cats, d, dropout)
        self.e2 = TemporalEncoder(d, dropout=dropout)
        self.e3 = SpatialEncoder(n_cells, d, dropout)
        self.e4 = StableUserEncoder(stable_dim, d, dropout)
        self.nulls = nn.Parameter(torch.zeros(4, d))

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.e1(
            batch["history_poi_ids"],
            batch["history_categories"],
            batch["history_time_gaps"],
            batch["history_mask"],
        )
        h2 = self.e2(
            batch["history_hours"],
            batch["history_dows"],
            batch["history_mask"],
            batch["query_time_features"],
        )
        h3 = self.e3(batch["history_cells"], batch["history_coordinates"], batch["history_mask"])
        avail = batch["source_available"]  # [B,4]
        h4 = self.e4(batch["stable_user_features"], avail[:, 3])
        sources = torch.stack([h1, h2, h3, h4], dim=1)  # [B,4,d]
        sources = avail.unsqueeze(-1) * sources + (1 - avail.unsqueeze(-1)) * self.nulls.unsqueeze(0)
        return sources, avail