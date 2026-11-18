"""Ablation CLI for Massive-STEPS STHGCN. Defaults reproduce the paper std_conf run."""
from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STHGCN (Massive-STEPS). Omitting ablation flags = paper defaults."
    )
    parser.add_argument("-f", "--yaml_file", help="The configuration file under conf/.", required=True)
    parser.add_argument(
        "--multi_run_mode",
        help="Run multiple experiments with the same config.",
        action="store_true",
    )
    parser.add_argument(
        "--use_user_embed",
        choices=["on", "off"],
        default="on",
        help="User embedding in CheckinEmbedding. off zeros the user lookup. Default: on.",
    )
    parser.add_argument(
        "--hypergraph_edges",
        choices=["both", "ci2traj", "traj2traj", "none"],
        default="both",
        help="Which hypergraph relations to message-pass. Default: both.",
    )
    parser.add_argument(
        "--graph_scope",
        choices=["full", "train"],
        default="full",
        help="full=sample.csv train∪val∪test (paper). train=only train trails as traj2traj neighbors. Default: full.",
    )
    parser.add_argument(
        "--unseen_prefix_poi",
        choices=["unk", "drop"],
        default="unk",
        help="unk=map train-unseen prefix POI to padding (paper). drop=drop eval sample if any prefix POI unseen. Default: unk.",
    )
    parser.add_argument(
        "--late_sequence",
        choices=["none", "poi", "category", "time", "location"],
        default="none",
        help="Pull out one sequence onto its own hypergraph stack; others stay on the "
        "paper fused path; fuse with concat+MLP just before softmax. "
        "none=paper path (no extra MLP). Default: none.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Override conf run_args.gpu (-1=CPU, >=0 CUDA index). Default: value from yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override conf run_args.seed. Default: value from yaml (or random with --multi_run_mode).",
    )
    return parser


def apply_ablation_args(cfg, args):
    """Wire CLI ablation flags into cfg so preprocess/model actually change behavior."""
    cfg.model_args.use_user_embed = args.use_user_embed
    cfg.model_args.hypergraph_edges = args.hypergraph_edges
    if args.hypergraph_edges in ("ci2traj", "none"):
        cfg.model_args.do_traj2traj = False
    cfg.dataset_args.graph_scope = args.graph_scope
    cfg.dataset_args.unseen_prefix_poi = args.unseen_prefix_poi
    cfg.model_args.late_sequence = args.late_sequence
    if args.gpu is not None:
        cfg.run_args.gpu = args.gpu
    if getattr(args, "seed", None) is not None:
        cfg.run_args.seed = int(args.seed)
    return cfg


def ablation_run_name(args) -> str:
    """Short run name for output dirs.

    All defaults → ``paper``. Otherwise only non-default tokens, joined by ``__``:
    ``no_user``, ``edges_ci2traj``, ``edges_traj2traj``, ``edges_none``,
    ``scope_train``, ``prefix_drop``, ``late_poi``, ``late_category``,
    ``late_time``, ``late_location``.
    """
    parts = []
    if getattr(args, "use_user_embed", "on") != "on":
        parts.append("no_user")
    edges = getattr(args, "hypergraph_edges", "both")
    if edges != "both":
        parts.append(f"edges_{edges}")
    if getattr(args, "graph_scope", "full") != "full":
        parts.append("scope_train")
    if getattr(args, "unseen_prefix_poi", "unk") != "unk":
        parts.append("prefix_drop")
    late = getattr(args, "late_sequence", "none")
    if late != "none":
        parts.append(f"late_{late}")
    return "paper" if not parts else "__".join(parts)


if __name__ == "__main__":
    build_arg_parser().parse_args()
