"""PC-SMIR full model."""

from __future__ import annotations

import torch
import torch.nn as nn

from pc_smir.models.encoders import CandidateEncoder, SourceEncoders
from pc_smir.models.experts import InteractionExperts
from pc_smir.models.router import POIConditionedRouter


class PCSMIR(nn.Module):
    def __init__(
        self,
        n_pois: int,
        n_cats: int,
        d: int = 64,
        meta_dim: int = 5,
        surd_dim: int = 1,
        stable_dim: int = 8,
        dropout: float = 0.1,
        pairs: list[tuple[int, int]] | None = None,
        triples: list[tuple[int, int, int]] | None = None,
        use_bilinear: bool = True,
        router_temperature: float = 1.0,
        ablation: str = "full",
    ):
        super().__init__()
        self.ablation = ablation
        self.sources = SourceEncoders(n_pois, n_cats, d, stable_dim=stable_dim, dropout=dropout)
        self.candidates = CandidateEncoder(n_pois, meta_dim, d, dropout=dropout)
        self.experts = InteractionExperts(
            d,
            meta_dim,
            pairs=pairs,
            triples=triples,
            dropout=dropout,
            use_bilinear=use_bilinear,
            include_fallback=True,
        )
        self.router = POIConditionedRouter(
            d=d,
            n_experts=self.experts.n_experts,
            surd_in_dim=max(surd_dim, 1),
            meta_dim=meta_dim,
            dropout=dropout,
            temperature=router_temperature,
        )
        # Shared all-source baseline scorer (optional additive; default disabled to avoid double-count).
        self.baseline = nn.Sequential(
            nn.Linear(d * 4 + d + meta_dim, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.use_additive_baseline = False

    def score(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        sources, source_mask = self.sources(batch)
        cand_repr, _, _ = self.candidates(batch["candidate_poi_ids"], batch["candidate_metadata"])
        expert_logits, expert_mask = self.experts(
            sources, cand_repr, batch["candidate_metadata"], source_mask
        )
        candidate_mask = batch.get("candidate_mask")
        surd = batch["candidate_surd"]
        surd_valid = batch["candidate_surd_valid"]

        if self.ablation == "uniform":
            routes = expert_mask / expert_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.ablation == "no_surd":
            surd = torch.zeros_like(surd)
            surd_valid = torch.zeros_like(surd_valid)
            routes = self.router(
                sources,
                cand_repr,
                batch["candidate_metadata"],
                surd,
                surd_valid,
                source_mask,
                expert_mask,
                candidate_mask,
            )
        elif self.ablation == "concat_mlp":
            # Ignore experts; use baseline only.
            bsz, n_cand, _ = cand_repr.shape
            flat = sources.reshape(bsz, -1).unsqueeze(1).expand(-1, n_cand, -1)
            scores = self.baseline(torch.cat([flat, cand_repr, batch["candidate_metadata"]], dim=-1)).squeeze(-1)
            routes = torch.zeros(bsz, n_cand, self.experts.n_experts, device=scores.device)
            return scores, routes
        else:
            routes = self.router(
                sources,
                cand_repr,
                batch["candidate_metadata"],
                surd,
                surd_valid,
                source_mask,
                expert_mask,
                candidate_mask,
            )

        scores = (routes * expert_logits).sum(dim=-1)
        if self.use_additive_baseline:
            bsz, n_cand, _ = cand_repr.shape
            flat = sources.reshape(bsz, -1).unsqueeze(1).expand(-1, n_cand, -1)
            scores = scores + self.baseline(torch.cat([flat, cand_repr, batch["candidate_metadata"]], dim=-1)).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min / 4)
        return scores, routes

    def forward(self, batch):
        return self.score(batch)