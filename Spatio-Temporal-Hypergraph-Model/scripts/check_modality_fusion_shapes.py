#!/usr/bin/env python3
"""Shape check for single-sequence late pull-out (no HF data / no PyG).

``late_sequence=none`` (paper): CheckinEmbedding concat → mean-pool → Linear →
``[B, num_poi]`` (no extra MLP).

``late_sequence`` in {poi, category, time, location}: remaining modalities stay
on the paper fused path (pulled P/C/T slots zeroed); the pulled sequence goes
through a stand-in encoder at ``embed_size``; concat(main, pulled) → MLP →
``[B, num_poi]``. Stand-in encoder is mean-pool (same as hypergraph_edges=none);
the real path uses HypergraphTransformer stacks.
"""
from __future__ import annotations

import importlib.util as ilu
import os.path as osp

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


def _ns(**kwargs):
    class NS:
        pass

    obj = NS()
    obj.use_user_embed = "on"
    obj.hypergraph_edges = "both"
    obj.graph_scope = "full"
    obj.unseen_prefix_poi = "unk"
    obj.late_sequence = "none"
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def paper_logits(checkin_emb: nn.Module, x_ci: torch.Tensor, batch_size: int, num_poi: int):
    fused = checkin_emb(x_ci)  # [N_ci, 5D]
    assert fused.size(-1) == 5 * checkin_emb.embed_size
    pooled = fused.mean(dim=0, keepdim=True).expand(batch_size, -1)
    scorer = nn.Linear(pooled.size(-1), num_poi)
    return scorer(pooled)


def late_one_logits(
    checkin_emb: nn.Module,
    x_ci: torch.Tensor,
    batch_size: int,
    num_poi: int,
    late_sequence: str,
):
    d = checkin_emb.embed_size
    exclude = late_sequence if late_sequence in ("poi", "category", "time") else None
    main = checkin_emb(x_ci, exclude=exclude)
    assert main.size(-1) == 5 * d
    main_q = main.mean(dim=0, keepdim=True).expand(batch_size, -1)

    pulled = checkin_emb.sequence_feature(x_ci, late_sequence)
    if late_sequence == "location":
        assert pulled.size(-1) == 2
        pulled = nn.Linear(2, d)(pulled)
    else:
        assert pulled.size(-1) == d
    late_q = pulled.mean(dim=0, keepdim=True).expand(batch_size, -1)
    fused = torch.cat([main_q, late_q], dim=-1)
    assert fused.size(-1) == 5 * d + d
    mlp = nn.Sequential(
        nn.Linear(5 * d + d, 5 * d + d),
        nn.ReLU(),
        nn.Linear(5 * d + d, num_poi),
    )
    return mlp(fused)


def main() -> int:
    torch.manual_seed(0)
    emb_mod = _load("embedding_layer_standalone", _LAYER)
    cli = _load("ablation_cli_standalone", _CLI)

    assert cli.ablation_run_name(_ns()) == "paper"
    for seq, token in (
        ("poi", "late_poi"),
        ("category", "late_category"),
        ("time", "late_time"),
        ("location", "late_location"),
    ):
        assert cli.ablation_run_name(_ns(late_sequence=seq)) == token, seq

    ds = _tiny_dataset_args()
    embed_size = 4
    num_poi = ds.num_poi
    n_ci = 12
    batch_size = 3
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

    paper_out = paper_logits(checkin_emb, x_ci, batch_size, num_poi)
    assert paper_out.shape == (batch_size, num_poi), paper_out.shape

    for seq in ("poi", "category", "time", "location"):
        late_out = late_one_logits(checkin_emb, x_ci, batch_size, num_poi, seq)
        assert late_out.shape == (batch_size, num_poi), (seq, late_out.shape)
        assert late_out.shape == paper_out.shape

    # Pulled P/C/T slots are zeroed on the fused path; location does not change concat.
    full = checkin_emb(x_ci)
    for seq, slice_ in (
        ("poi", slice(embed_size, 2 * embed_size)),
        ("category", slice(2 * embed_size, 3 * embed_size)),
        ("time", slice(3 * embed_size, 5 * embed_size)),
    ):
        excl = checkin_emb(x_ci, exclude=seq)
        assert torch.allclose(excl[..., slice_], torch.zeros_like(excl[..., slice_]))
        # Other slots match the paper concat.
        keep = [i for i in range(5 * embed_size) if i not in range(slice_.start, slice_.stop)]
        assert torch.allclose(excl[..., keep], full[..., keep])

    loc_excl = checkin_emb(x_ci, exclude="location")
    assert torch.allclose(loc_excl, full)

    print(
        f"LATE_SEQUENCE_SHAPE_OK: paper={tuple(paper_out.shape)}; "
        f"late_poi/category/time/location each={tuple(paper_out.shape)} "
        f"(POI-class logits); run_name tokens late_*."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
