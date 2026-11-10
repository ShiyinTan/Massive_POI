"""POI-conditioned SURD-guided router."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProfileEncoder(nn.Module):
    def __init__(self, in_dim: int, d_s: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d_s), nn.GELU(), nn.LayerNorm(d_s))
        self.unknown = nn.Parameter(torch.zeros(d_s))

    def forward(self, profile, valid):
        # profile [B,C,F], valid [B,C]
        h = self.net(profile)
        unk = self.unknown.view(1, 1, -1).expand_as(h)
        return valid.unsqueeze(-1) * h + (1 - valid.unsqueeze(-1)) * unk


class POIConditionedRouter(nn.Module):
    """Shared MLP router with expert-ID embeddings (spec §6.1)."""

    def __init__(
        self,
        d: int,
        n_experts: int,
        surd_in_dim: int,
        meta_dim: int,
        d_r: int | None = None,
        d_expert_id: int = 16,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        d_r = d_r or d
        self.n_experts = n_experts
        self.temperature = temperature
        self.ctx = nn.Sequential(nn.Linear(d * 4 + 4, d_r), nn.GELU(), nn.LayerNorm(d_r))
        self.profile_enc = ProfileEncoder(surd_in_dim, d_r)
        self.cand_proj = nn.Linear(d, d_r)
        self.meta_proj = nn.Linear(meta_dim, d_r)
        self.expert_id = nn.Embedding(n_experts, d_expert_id)
        self.route = nn.Sequential(
            nn.Linear(4 * d_r + d_expert_id + 1, d_r),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_r, 1),
        )

    def forward(
        self,
        sources,
        cand_repr,
        cand_meta,
        surd_profile,
        surd_valid,
        source_mask,
        expert_mask,
        candidate_mask=None,
    ):
        """
        Returns route weights [B,C,E] that sum to 1 over experts (masked).
        """
        bsz, n_cand, _ = cand_repr.shape
        # Context q from sources.
        flat = torch.cat([sources.reshape(bsz, -1), source_mask], dim=-1)
        q = self.ctx(flat)  # [B,d_r]
        q = q.unsqueeze(1).expand(-1, n_cand, -1)
        e = self.cand_proj(cand_repr)
        c = self.meta_proj(cand_meta)
        x = self.profile_enc(surd_profile, surd_valid)
        # reliability flag retained explicitly
        rel = surd_valid.unsqueeze(-1)

        # Per-expert logits
        ids = torch.arange(self.n_experts, device=cand_repr.device)
        v = self.expert_id(ids).view(1, 1, self.n_experts, -1).expand(bsz, n_cand, -1, -1)
        base = torch.cat([q, e, c, x], dim=-1).unsqueeze(2).expand(-1, -1, self.n_experts, -1)
        rel_b = rel.unsqueeze(2).expand(-1, -1, self.n_experts, -1)
        inp = torch.cat([base, v, rel_b], dim=-1)
        r = self.route(inp).squeeze(-1)  # [B,C,E]

        # Mask unavailable experts with large negative logit; keep fallback available.
        neg = torch.finfo(r.dtype).min / 4
        r = r.masked_fill(expert_mask <= 0, neg)
        if candidate_mask is not None:
            r = r.masked_fill(candidate_mask.unsqueeze(-1) <= 0, neg)

        alpha = F.softmax(r / max(self.temperature, 1e-6), dim=-1)
        # Zero out masked experts after softmax and renorm for numerical safety.
        alpha = alpha * expert_mask
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if candidate_mask is not None:
            alpha = alpha * candidate_mask.unsqueeze(-1)
        return alpha