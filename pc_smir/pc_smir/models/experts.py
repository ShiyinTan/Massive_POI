"""Single-source, pairwise, higher-order, and fallback interaction experts."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, d),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d, d),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d, 1),
    )


class InteractionExperts(nn.Module):
    """Explicit experts for selected subsets + fallback candidate-context expert."""

    def __init__(
        self,
        d: int,
        meta_dim: int,
        pairs: list[tuple[int, int]] | None = None,
        triples: list[tuple[int, int, int]] | None = None,
        dropout: float = 0.1,
        use_bilinear: bool = True,
        include_fallback: bool = True,
    ):
        super().__init__()
        self.d = d
        self.pairs = pairs or [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
        self.triples = triples or [(0, 1, 2), (0, 2, 3)]
        self.use_bilinear = use_bilinear
        self.include_fallback = include_fallback

        self.expert_names: list[str] = [f"U{i}" for i in range(4)]
        self.expert_names += [f"S{i}{j}" for i, j in self.pairs]
        self.expert_names += [f"T{i}{j}{k}" for i, j, k in self.triples]
        if include_fallback:
            self.expert_names.append("FALLBACK")
        self.n_experts = len(self.expert_names)

        self.single_heads = nn.ModuleList([_mlp(2 * d + meta_dim, d, dropout) for _ in range(4)])
        self.pair_W = nn.ModuleList(
            [nn.Bilinear(d, d, 1, bias=False) if use_bilinear else nn.Identity() for _ in self.pairs]
        )
        pair_in = 3 * d + (1 if use_bilinear else 0) + d + meta_dim + 2
        self.pair_heads = nn.ModuleList([_mlp(pair_in, d, dropout) for _ in self.pairs])

        self.triple_P = nn.ModuleList(
            [nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)]) for _ in self.triples]
        )
        trip_in = 3 * d + 3 * d + d + d + meta_dim
        self.triple_heads = nn.ModuleList([_mlp(trip_in, d, dropout) for _ in self.triples])

        if include_fallback:
            self.fallback = _mlp(d + meta_dim, d, dropout)

    def _score_singles(self, src, cand_repr, cand_meta, sm):
        logits, masks = [], []
        for i, head in enumerate(self.single_heads):
            hi = src[:, :, i, :]
            logits.append(head(torch.cat([hi, cand_repr, cand_meta], dim=-1)).squeeze(-1))
            masks.append(sm[:, :, i])
        return logits, masks

    def _score_pairs(self, src, cand_repr, cand_meta, sm, bsz, n_cand, d):
        logits, masks = [], []
        for p_idx, (i, j) in enumerate(self.pairs):
            hi, hj = src[:, :, i, :], src[:, :, j, :]
            parts = [hi, hj, hi * hj]
            if self.use_bilinear:
                bil = self.pair_W[p_idx](hi.reshape(-1, d), hj.reshape(-1, d)).reshape(bsz, n_cand, 1)
                parts.append(bil)
            parts.extend([cand_repr, cand_meta, sm[:, :, i : i + 1], sm[:, :, j : j + 1]])
            logits.append(self.pair_heads[p_idx](torch.cat(parts, dim=-1)).squeeze(-1))
            masks.append(torch.minimum(sm[:, :, i], sm[:, :, j]))
        return logits, masks

    def _score_triples(self, src, cand_repr, cand_meta, sm):
        logits, masks = [], []
        for t_idx, (i, j, k) in enumerate(self.triples):
            hi, hj, hk = src[:, :, i, :], src[:, :, j, :], src[:, :, k, :]
            Pi, Pj, Pk = self.triple_P[t_idx]
            lowrank = Pi(hi) * Pj(hj) * Pk(hk)
            inp = torch.cat(
                [hi, hj, hk, hi * hj, hi * hk, hj * hk, lowrank, cand_repr, cand_meta],
                dim=-1,
            )
            logits.append(self.triple_heads[t_idx](inp).squeeze(-1))
            masks.append(torch.minimum(torch.minimum(sm[:, :, i], sm[:, :, j]), sm[:, :, k]))
        return logits, masks

    def forward(self, sources, cand_repr, cand_meta, source_mask):
        """
        sources: [B,M,d]
        cand_repr: [B,C,d]
        cand_meta: [B,C,F]
        source_mask: [B,M]
        returns logits [B,C,E], expert_mask [B,C,E]
        """
        bsz, n_cand, d = cand_repr.shape
        src = sources.unsqueeze(1).expand(-1, n_cand, -1, -1)
        sm = source_mask.unsqueeze(1).expand(-1, n_cand, -1)

        logits, masks = self._score_singles(src, cand_repr, cand_meta, sm)
        p_logits, p_masks = self._score_pairs(src, cand_repr, cand_meta, sm, bsz, n_cand, d)
        logits.extend(p_logits)
        masks.extend(p_masks)
        t_logits, t_masks = self._score_triples(src, cand_repr, cand_meta, sm)
        logits.extend(t_logits)
        masks.extend(t_masks)

        if self.include_fallback:
            logits.append(self.fallback(torch.cat([cand_repr, cand_meta], dim=-1)).squeeze(-1))
            masks.append(torch.ones(bsz, n_cand, device=cand_repr.device))

        return torch.stack(logits, dim=-1), torch.stack(masks, dim=-1)
