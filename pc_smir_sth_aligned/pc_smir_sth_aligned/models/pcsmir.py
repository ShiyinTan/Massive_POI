"""PC-SMIR (STH-aligned): SURD routing + strong user/transition personalization head."""

from __future__ import annotations

import torch
import torch.nn as nn

from pc_smir_sth_aligned.models.encoders import CandidateEncoder, SourceEncoders
from pc_smir_sth_aligned.models.experts import InteractionExperts
from pc_smir_sth_aligned.models.router import POIConditionedRouter


class PCSMIR(nn.Module):
    def __init__(
        self,
        n_pois: int,
        n_cats: int,
        n_users: int = 2,
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
        use_user_poi: bool = True,
        use_transition: bool = True,
        user_poi_scale: float = 1.0,
        transition_scale: float = 1.0,
        pop_bias_scale: float = 1.0,
    ):
        super().__init__()
        self.ablation = ablation
        self.use_user_poi = use_user_poi
        self.use_transition = use_transition
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

        # STHGCN/GETNext-style personalization: user × POI and last-POI → next.
        self.user_emb = nn.Embedding(max(n_users, 2), d, padding_idx=0)
        self.user_poi_emb = nn.Embedding(n_pois, d, padding_idx=0)
        self.trans_src_emb = nn.Embedding(n_pois, d, padding_idx=0)
        self.trans_dst_emb = nn.Embedding(n_pois, d, padding_idx=0)
        self.user_poi_logit_scale = nn.Parameter(torch.tensor(float(user_poi_scale)))
        self.transition_logit_scale = nn.Parameter(torch.tensor(float(transition_scale)))
        self.pop_bias_scale = nn.Parameter(torch.tensor(float(pop_bias_scale)))
        self.poi_bias = nn.Embedding(n_pois, 1, padding_idx=0)
        nn.init.zeros_(self.poi_bias.weight)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.user_poi_emb.weight, std=0.02)
        nn.init.normal_(self.trans_src_emb.weight, std=0.02)
        nn.init.normal_(self.trans_dst_emb.weight, std=0.02)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
            self.user_poi_emb.weight[0].zero_()
            self.trans_src_emb.weight[0].zero_()
            self.trans_dst_emb.weight[0].zero_()

    def _personalization_scores(self, batch, cand_ids: torch.Tensor) -> torch.Tensor:
        """User–POI + transition + popularity/item bias over candidates [B,C]."""
        scores = torch.zeros(cand_ids.shape, device=cand_ids.device, dtype=torch.float32)
        if self.use_user_poi and "user_id" in batch:
            u = self.user_emb(batch["user_id"].clamp(min=0))  # [B,d]
            p = self.user_poi_emb(cand_ids)  # [B,C,d]
            scores = scores + self.user_poi_logit_scale * (p * u.unsqueeze(1)).sum(dim=-1)
        if self.use_transition and "last_poi_id" in batch:
            src = self.trans_src_emb(batch["last_poi_id"].clamp(min=0))  # [B,d]
            dst = self.trans_dst_emb(cand_ids)
            scores = scores + self.transition_logit_scale * (dst * src.unsqueeze(1)).sum(dim=-1)
        # candidate_metadata[:, :, 1] = log1p(train_pop)
        if "candidate_metadata" in batch:
            scores = scores + self.pop_bias_scale * batch["candidate_metadata"][..., 1]
        scores = scores + self.poi_bias(cand_ids).squeeze(-1)
        return scores

    def score(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        sources, source_mask = self.sources(batch)
        cand_repr, _, _ = self.candidates(batch["candidate_poi_ids"], batch["candidate_metadata"])
        expert_logits, expert_mask = self.experts(
            sources, cand_repr, batch["candidate_metadata"], source_mask
        )
        candidate_mask = batch.get("candidate_mask")
        surd = batch["candidate_surd"]
        surd_valid = batch["candidate_surd_valid"]
        pers = self._personalization_scores(batch, batch["candidate_poi_ids"])

        if self.ablation == "personalization_only":
            bsz, n_cand = cand_repr.shape[:2]
            routes = torch.zeros(bsz, n_cand, self.experts.n_experts, device=cand_repr.device)
            scores = pers
            if candidate_mask is not None:
                scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min / 4)
            return scores, routes
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
            bsz, n_cand, _ = cand_repr.shape
            flat = sources.reshape(bsz, -1).unsqueeze(1).expand(-1, n_cand, -1)
            scores = self.baseline(torch.cat([flat, cand_repr, batch["candidate_metadata"]], dim=-1)).squeeze(-1)
            scores = scores + pers
            routes = torch.zeros(bsz, n_cand, self.experts.n_experts, device=scores.device)
            if candidate_mask is not None:
                scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min / 4)
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

        scores = (routes * expert_logits).sum(dim=-1) + pers
        if self.use_additive_baseline:
            bsz, n_cand, _ = cand_repr.shape
            flat = sources.reshape(bsz, -1).unsqueeze(1).expand(-1, n_cand, -1)
            scores = scores + self.baseline(torch.cat([flat, cand_repr, batch["candidate_metadata"]], dim=-1)).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min / 4)
        return scores, routes

    def forward(self, batch):
        return self.score(batch)
