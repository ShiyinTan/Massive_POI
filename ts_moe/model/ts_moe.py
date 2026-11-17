"""Target-conditioned SURD-guided Mixture-of-Experts Transformer.

Forward path (spec §14):

    four sequences
      → independent Transformer encoders
      → candidate-aware cross-attention
      → SURD prior + context router
      → unique / redundancy / synergy experts, top-k
      → base score + MoE score
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dataset.build import Vocab
from ..surd.atoms import SOURCE_INDEX, select_experts
from .experts import ExpertFFN, RedundancyExpert, SynergyExpert


class CandidateCrossAttention(nn.Module):
    """``z = softmax(q Wq (H Wk)^T / sqrt(d)) H Wv`` with one projection set."""

    def __init__(self, d_model: int):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** 0.5

    def forward(self, query: torch.Tensor, memory: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        attn = torch.matmul(self.wq(query), self.wk(memory).transpose(1, 2)) / self.scale
        attn = attn.masked_fill(pad[:, None, :], torch.finfo(attn.dtype).min)
        weight = torch.nan_to_num(torch.softmax(attn, dim=-1), nan=0.0)
        return torch.matmul(weight, self.wv(memory))


class ContextRouter(nn.Module):
    """``ℓ = g + λ_r W_r r`` then ``π = softmax(ℓ / τ)`` (spec §6)."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.candidate = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model, n_experts)
        self.prior_map = nn.Linear(n_experts, n_experts, bias=False)

    def forward(
        self,
        zs: list[torch.Tensor],
        query: torch.Tensor,
        prior: torch.Tensor,
        mode: str,
        temperature: float,
        guidance: float,
    ) -> torch.Tensor:
        context = self.context(torch.cat(zs, dim=-1))
        g = self.gate(torch.cat([context, self.candidate(query)], dim=-1))
        if mode == "surd":
            logits = g + float(guidance) * self.prior_map(prior)
        elif mode == "plain":
            logits = g
        elif mode == "random":
            logits = torch.randn_like(g)
        elif mode == "fixed":
            logits = torch.zeros_like(g)
        else:
            raise ValueError(f"unknown routing mode {mode!r}")
        pi = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        if mode in {"random", "fixed"}:
            pi = pi.detach()
        return pi


class TSMoE(nn.Module):
    def __init__(self, vocab: Vocab, config: dict):
        super().__init__()
        model_cfg = config["model"]
        d_model = int(model_cfg["d_model"])
        n_heads = int(model_cfg["n_heads"])
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_pois = vocab.n_pois
        self.top_k = int(model_cfg["top_k"])
        self.temperature = float(model_cfg["temperature"])
        self.guidance = float(model_cfg["guidance_scale"])
        self.routing_mode = str(model_cfg["routing_mode"])
        self.max_len = int(model_cfg["max_len"])
        dropout = float(model_cfg["dropout"])

        indices, specs = select_experts(
            variant=int(model_cfg["expert_variant"]),
            groups=model_cfg.get("expert_groups"),
        )
        self.expert_indices = indices
        self.expert_specs = specs
        self.n_experts = len(specs)

        self.user_emb = nn.Embedding(vocab.n_users, d_model)
        self.poi_emb = nn.Embedding(vocab.n_pois, d_model, padding_idx=0)
        self.hour_emb = nn.Embedding(24, d_model)
        self.dow_emb = nn.Embedding(7, d_model)
        self.cat_emb = nn.Embedding(vocab.n_cats, d_model)
        self.proj_u = nn.Linear(d_model, d_model)
        self.proj_p = nn.Linear(d_model, d_model)
        self.proj_t = nn.Linear(d_model, d_model)
        self.proj_l = nn.Linear(2, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)

        # Category and coordinates are added into q_y (spec §3.1). Zero-init
        # keeps the initial query equal to the POI embedding.
        self.cat_proj = nn.Linear(d_model, d_model)
        self.xy_proj = nn.Linear(2, d_model)
        nn.init.zeros_(self.cat_proj.weight)
        nn.init.zeros_(self.cat_proj.bias)
        nn.init.zeros_(self.xy_proj.weight)
        nn.init.zeros_(self.xy_proj.bias)

        self.encoders = nn.ModuleList(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=int(model_cfg["n_layers"]),
            )
            for _ in range(4)
        )
        self.cross = nn.ModuleList(CandidateCrossAttention(d_model) for _ in range(4))

        self.unique = nn.ModuleDict()
        self.redundancy = nn.ModuleDict()
        self.synergy = nn.ModuleDict()
        self.syn_proj = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        for kind, members in specs:
            key = "".join(members)
            if kind == "unique":
                self.unique[key] = ExpertFFN(d_model)
            elif kind == "redundancy":
                self.redundancy[key] = RedundancyExpert(d_model)
            else:
                member_idx = tuple(SOURCE_INDEX[name] for name in members)
                self.synergy[key] = SynergyExpert(d_model, member_idx)

        self.router = ContextRouter(d_model, self.n_experts)
        self.base = nn.Sequential(
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.w_base = nn.Linear(d_model, 1, bias=False)
        self.w_moe = nn.Linear(d_model, 1, bias=False)

        self.register_buffer("poi_lat", torch.tensor(vocab.poi_lat, dtype=torch.float32))
        self.register_buffer("poi_lon", torch.tensor(vocab.poi_lon, dtype=torch.float32))
        self.register_buffer("poi_cat", torch.tensor(vocab.poi_cat, dtype=torch.long))

    def encode(self, batch: dict[str, torch.Tensor]) -> dict:
        length = batch["poi"].size(1)
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")
        user = self.user_emb(batch["user"]).unsqueeze(1).expand(-1, length, -1)
        poi = self.poi_emb(batch["poi"])
        time = self.hour_emb(batch["hour"]) + self.dow_emb(batch["dow"])
        valid_len = (~batch["pad"]).sum(dim=1)
        index = torch.arange(length, device=batch["poi"].device).unsqueeze(0)
        pos_ids = (index - (length - valid_len).unsqueeze(1)).clamp(min=0)
        pos = self.pos_emb(pos_ids)
        projected = [
            self.proj_u(user),
            self.proj_p(poi),
            self.proj_t(time),
            self.proj_l(batch["loc"]),
        ]
        hidden = []
        for encoder, features in zip(self.encoders, projected):
            h0 = (features + pos).masked_fill(batch["pad"].unsqueeze(-1), 0.0)
            hidden.append(encoder(h0, src_key_padding_mask=batch["pad"]))
        return {"H": hidden, "pad": batch["pad"]}

    def candidate_query(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        xy = torch.stack([self.poi_lat[candidate_ids], self.poi_lon[candidate_ids]], dim=-1)
        return (
            self.poi_emb(candidate_ids)
            + self.cat_proj(self.cat_emb(self.poi_cat[candidate_ids]))
            + self.xy_proj(xy)
        )

    def _expert_outputs(self, zs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [layer(zs[i]) for i, layer in enumerate(self.syn_proj)]
        outputs = []
        recon = []
        for kind, members in self.expert_specs:
            key = "".join(members)
            if kind == "unique":
                outputs.append(self.unique[key](zs[SOURCE_INDEX[members[0]]]))
            elif kind == "redundancy":
                i, j = SOURCE_INDEX[members[0]], SOURCE_INDEX[members[1]]
                output, rec = self.redundancy[key](zs[i], zs[j])
                outputs.append(output)
                recon.append(rec)
            else:
                outputs.append(self.synergy[key](zs, projected))
        stacked = torch.stack(outputs, dim=2)
        if recon:
            shared = torch.stack(recon).mean()
        else:
            shared = stacked.new_zeros(())
        return stacked, shared

    def _mix(self, outputs: torch.Tensor, pi: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "fixed":
            return (outputs * pi.unsqueeze(-1)).sum(dim=2)
        k = min(self.top_k, outputs.size(2))
        value, index = torch.topk(pi, k=k, dim=-1)
        value = value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gather_index = index.unsqueeze(-1).expand(-1, -1, -1, outputs.size(-1))
        selected = torch.gather(outputs, 2, gather_index)
        return (selected * value.unsqueeze(-1)).sum(dim=2)

    def score_encoded(
        self,
        encoded: dict,
        candidate_ids: torch.Tensor,
        prior: torch.Tensor,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score each candidate with its own SURD prior (spec §9).

        ``prior`` is ``[B, M, E]`` already restricted to the active experts.
        The target label is not an input.
        """
        mode = self.routing_mode if mode is None else mode
        prior = prior.detach()
        query = self.candidate_query(candidate_ids)
        zs = [layer(query, memory, encoded["pad"]) for layer, memory in zip(self.cross, encoded["H"])]
        outputs, shared = self._expert_outputs(zs)
        pi = self.router(zs, query, prior, mode, self.temperature, self.guidance)
        mixed = self._mix(outputs, pi, mode)
        base = self.base(torch.cat(zs + [query], dim=-1))
        s_base = self.w_base(base).squeeze(-1)
        s_moe = self.w_moe(mixed).squeeze(-1)
        return {
            "scores": s_base + s_moe,
            "s_base": s_base,
            "s_moe": s_moe,
            "pi": pi,
            "shared_loss": shared,
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        candidate_ids: torch.Tensor,
        prior: torch.Tensor,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.score_encoded(self.encode(batch), candidate_ids, prior, mode=mode)

    @torch.no_grad()
    def recall_candidates(self, encoded: dict, m: int) -> torch.Tensor:
        """Lightweight §10 recall: pooled POI-encoder state dotted with POI embeddings.

        This is only a candidate generator. Scores on the recalled set come
        from the full base + MoE head.
        """
        memory = encoded["H"][1]
        valid = (~encoded["pad"]).unsqueeze(-1).to(memory.dtype)
        pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        scores = pooled @ self.poi_emb.weight.T
        scores = scores.clone()
        scores[:, 0] = torch.finfo(scores.dtype).min
        k = min(int(m), scores.size(1) - 1)
        return torch.topk(scores, k=k, dim=-1).indices

    def rank_batch(
        self,
        batch: dict[str, torch.Tensor],
        prior_table: torch.Tensor,
        *,
        strategy: str,
        recall_m: int,
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        """Score a candidate set for every row.

        ``strategy="full"`` scores every real POI. ``strategy="two_stage"``
        recalls ``recall_m`` POIs and runs target-conditioned routing on them.
        """
        encoded = self.encode(batch)
        if strategy == "full":
            ids = torch.arange(1, self.n_pois, device=batch["poi"].device)
            candidates = ids.unsqueeze(0).expand(batch["poi"].size(0), -1)
        elif strategy == "two_stage":
            candidates = self.recall_candidates(encoded, recall_m)
        else:
            raise ValueError(f"unknown candidate strategy {strategy!r}")

        score_parts = []
        pi_parts = []
        for start in range(0, candidates.size(1), chunk_size):
            part = candidates[:, start : start + chunk_size]
            out = self.score_encoded(encoded, part, prior_table[part])
            score_parts.append(out["scores"])
            pi_parts.append(out["pi"])
        return {
            "scores": torch.cat(score_parts, dim=1),
            "pi": torch.cat(pi_parts, dim=1),
            "candidates": candidates,
        }
