#!/usr/bin/env python3
"""Evaluate a checkpoint on Massive-STEPS official test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surd_guided_poi.data.cold_filter import (
    maybe_drop_unseen_user_poi,
    seen_user_poi_ids_from_train_frame,
)
from surd_guided_poi.data.dataset import NextPOIDataset, collate_next_poi
from surd_guided_poi.data.massive_steps import load_official_split_frames
from surd_guided_poi.data.preprocessing import (
    build_samples_from_split,
    build_vocab_from_frames,
    poi_sample_counts,
)
from surd_guided_poi.evaluation.long_tail_metrics import evaluate_by_frequency, frequency_buckets
from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking
from surd_guided_poi.models.full_model import build_surd_model
from surd_guided_poi.training.checkpoint import load_checkpoint, write_json
from surd_guided_poi.training.config import load_config


def drop_unseen_user_poi_enabled(cfg: dict) -> bool:
    eval_cfg = cfg.get("eval") or {}
    data_cfg = cfg.get("data") or {}
    if "drop_unseen_user_poi" in eval_cfg:
        return bool(eval_cfg["drop_unseen_user_poi"])
    if "drop_unseen_user_poi" in data_cfg:
        return bool(data_cfg["drop_unseen_user_poi"])
    return False


def _config_paths(args_config: list[str]) -> list[Path]:
    return [Path(c) if Path(c).is_file() else PKG_ROOT / c for c in args_config]


def _samples(frame, vocab, split: str, data_cfg: dict, eval_cfg: dict | None = None):
    # History = this split CSV only, same-trail prefixes before the target (no dump leak).
    # Default eval: last POI of each trail (Massive-STEPS Flashback Acc@1).
    eval_cfg = eval_cfg or {}
    prefixes = bool(eval_cfg.get("use_all_prefixes", data_cfg.get("use_all_prefixes", False)))
    return build_samples_from_split(
        frame,
        vocab,
        split,
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=prefixes,
        user_history_df=frame,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/dataset_beijing.yaml"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_json", default="outputs/eval_metrics.json")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. cuda / cuda:0 / cpu (default: config or auto)",
    )
    args = parser.parse_args()

    cfg = load_config(_config_paths(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.device:
        cfg["device"] = args.device

    # Same resolution as train.py: CLI > config > auto cuda/cpu
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg["data"]
    city = data_cfg["city"]
    cache_dir = PKG_ROOT / data_cfg.get("cache_dir", "outputs/data_cache") / city
    frames = load_official_split_frames(
        city,
        data_dir=data_cfg.get("data_dir"),
        cache_dir=cache_dir,
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
    )
    vocab = build_vocab_from_frames(frames)
    eval_cfg = cfg.get("eval", {})
    drop_cold = drop_unseen_user_poi_enabled(cfg)
    # Graphs for graph_flashback: rebuild from train with all prefixes (match train.py).
    train_samples_for_graphs = _samples(
        frames["train"], vocab, "train", data_cfg, {"use_all_prefixes": True}
    )
    test_samples = _samples(frames["test"], vocab, "test", data_cfg, eval_cfg)
    train_users, train_pois = seen_user_poi_ids_from_train_frame(frames["train"], vocab)
    test_samples = maybe_drop_unseen_user_poi(
        test_samples, train_user_ids=train_users, train_poi_ids=train_pois, enabled=drop_cold
    )
    # 长尾分桶要用整数 N_y，不能用归一化的 prior
    train_counts = poi_sample_counts(train_samples_for_graphs, vocab.num_pois)

    model = build_surd_model(
        vocab, cfg["model"], train_samples=train_samples_for_graphs
    )
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)

    test_loader = DataLoader(
        NextPOIDataset(test_samples),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_next_poi,
    )
    metrics = evaluate_ranking(model, test_loader, device, ks=cfg["eval"]["ks"])

    model.eval()
    logits_list, tgt_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits_list.append(model.recommend(batch).cpu())
            tgt_list.append(batch.target_poi_id.cpu())
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(tgt_list, dim=0)
    buckets = frequency_buckets(train_counts, cutoffs=cfg["eval"]["frequency_buckets"])
    long_tail = evaluate_by_frequency(logits, targets, buckets["poi_bucket"], ks=cfg["eval"]["ks"])

    result = {
        "ranking": metrics,
        "long_tail": long_tail,
        "split_protocol": "massive_steps_official",
        "drop_unseen_user_poi": drop_cold,
        "n_test": len(test_samples),
        "backbone": str(cfg.get("model", {}).get("backbone", "flashback")),
        "device": str(device),
    }
    out = Path(args.output_json)
    if not out.is_absolute():
        out = PKG_ROOT / out
    write_json(out, result)
    print(f"device={device}")
    print("ranking:", metrics)
    print("wrote", out)


if __name__ == "__main__":
    main()
