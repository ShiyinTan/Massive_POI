#!/usr/bin/env python3
"""Shape check for early vs late modality fusion (no HF data / no PyG).

Early (paper): CheckinEmbedding concat → mean-pool → Linear → [B, num_poi].
Late: P/C/T/L each kept at embed_size through a stand-in encoder, then
concat(+user) → MLP → [B, num_poi]. Stand-in encoder is mean-pool (same as
hypergraph_edges=none); real late path uses HypergraphTransformer stacks.
"""
from __future__ import annotations

import importlib.util as ilu
import os.path as osp
import sys

import torch
from torch import nn

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
_LAYER = osp.join(_ROOT, "layer", "embedding_layer.py")
_CLI = osp.join(_ROOT, "utils", "ablation_cli.py")


def _load(name: str, path: str):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_dataset_args():
    class NS:
        num_user = 8
        num_poi = 20
        num_category = 6
        padding_user_id = 0
        padding_poi_id = 0
        padding_poi_category = 0
        padding_weekday_id = 0
        padding_hour_id = 0

    return NS()


def early_logits(checkin_emb: nn.Module, x_ci: torch.Tensor, batch_size: int, num_poi: int):
    fused = checkin_emb(x_ci)  # [N_ci, 5D]
    assert fused.size(-1) == 5 * checkin_emb.embed_size
    pooled = fused.mean(dim=0, keepdim=True).expand(batch_size, -1)
    scorer = nn.Linear(pooled.size(-1), num_poi)
    return scorer(pooled)


def late_logits(checkin_emb: nn.Module, x_ci: torch.Tensor, batch_size: int, num_poi: int):
    mods = checkin_emb.forward_modalities(x_ci)
    d = checkin_emb.embed_size
    loc_proj = nn.Linear(2, d)
    loc = loc_proj(mods["L_raw"])

    def encode(feat: torch.Tensor) -> torch.Tensor:
        # Stand-in for one HypergraphTransformer stack (edges=none mean-pool).
        return feat.mean(dim=0, keepdim=True).expand(batch_size, -1)

    p, c, t, l = encode(mods["P"]), encode(mods["C"]), encode(mods["T"]), encode(loc)
    user_q = encode(mods["user"])
    for tns in (p, c, t, l, user_q):
        assert tns.shape == (batch_size, d), tns.shape
    fused = torch.cat([p, c, t, l, user_q], dim=-1)
    assert fused.size(-1) == 5 * d
    mlp = nn.Sequential(nn.Linear(5 * d, 5 * d), nn.ReLU(), nn.Linear(5 * d, num_poi))
    return mlp(fused)


def main() -> int:
    torch.manual_seed(0)
    emb_mod = _load("embedding_layer_standalone", _LAYER)
    cli = _load("ablation_cli_standalone", _CLI)

    # run_name token
    class NS:
        pass

    early_ns = NS()
    early_ns.use_user_embed = "on"
    early_ns.hypergraph_edges = "both"
    early_ns.graph_scope = "full"
    early_ns.unseen_prefix_poi = "unk"
    early_ns.modality_fusion = "early"
    assert cli.ablation_run_name(early_ns) == "paper"

    late_ns = NS()
    late_ns.use_user_embed = "on"
    late_ns.hypergraph_edges = "both"
    late_ns.graph_scope = "full"
    late_ns.unseen_prefix_poi = "unk"
    late_ns.modality_fusion = "late"
    assert cli.ablation_run_name(late_ns) == "fusion_late"

    ds = _tiny_dataset_args()
    embed_size = 4
    num_poi = ds.num_poi
    n_ci = 12
    batch_size = 3
    # columns: user, poi, cat, ?, lon, lat, weekday, hour
    x_ci = torch.zeros(n_ci, 8)
    x_ci[:, 0] = torch.randint(1, ds.num_user + 1, (n_ci,))
    x_ci[:, 1] = torch.randint(1, ds.num_poi + 1, (n_ci,))
    x_ci[:, 2] = torch.randint(1, ds.num_category + 1, (n_ci,))
    x_ci[:, 4] = torch.randn(n_ci)
    x_ci[:, 5] = torch.randn(n_ci)
    x_ci[:, 6] = torch.randint(1, 8, (n_ci,))
    x_ci[:, 7] = torch.randint(1, 25, (n_ci,))

    checkin_emb = emb_mod.CheckinEmbedding(
        embed_size=embed_size,
        fusion_type="concat",
        dataset_args=ds,
        use_user_embed=True,
    )

    early_out = early_logits(checkin_emb, x_ci, batch_size, num_poi)
    late_out = late_logits(checkin_emb, x_ci, batch_size, num_poi)
    assert early_out.shape == (batch_size, num_poi), early_out.shape
    assert late_out.shape == (batch_size, num_poi), late_out.shape
    assert early_out.shape == late_out.shape

    print(
        f"MODALITY_FUSION_SHAPE_OK: early={tuple(early_out.shape)} "
        f"late={tuple(late_out.shape)} (POI-class logits); "
        f"run_name late→fusion_late."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
