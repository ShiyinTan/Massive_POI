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

from surd_guided_poi.data.dataset import NextPOIDataset, collate_next_poi
from surd_guided_poi.data.massive_steps import load_official_split_frames
from surd_guided_poi.data.preprocessing import (
    build_samples_from_split,
    build_vocab_from_frames,
    poi_sample_counts,
)
from surd_guided_poi.evaluation.long_tail_metrics import evaluate_by_frequency, frequency_buckets
from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.training.checkpoint import load_checkpoint, write_json
from surd_guided_poi.training.config import load_config


def _config_paths(args_config: list[str]) -> list[Path]:
    return [Path(c) if Path(c).is_file() else PKG_ROOT / c for c in args_config]


def _samples(frame, vocab, split: str, data_cfg: dict):
    return build_samples_from_split(
        frame,
        vocab,
        split,
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
    )


def _build_model(vocab, model_cfg: dict) -> SURDGuidedPOIModel:
    hidden = int(model_cfg["hidden_dim"])
    return SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        num_users=vocab.num_users,
        hidden_dim=hidden,
        poi_emb_dim=int(model_cfg.get("poi_emb_dim", hidden)),
        subset_hidden_dim=int(model_cfg.get("subset_hidden_dim", hidden)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        rnn_type=str(model_cfg.get("rnn_type", "gru")),
        lambda_t=float(model_cfg.get("lambda_t", 0.1)),
        lambda_s=float(model_cfg.get("lambda_s", 100.0)),
        share_poi_embedding=bool(model_cfg.get("share_poi_embedding", True)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/dataset_beijing.yaml"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_json", default="outputs/eval_metrics.json")
    args = parser.parse_args()

    cfg = load_config(_config_paths(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_samples = _samples(frames["train"], vocab, "train", data_cfg)
    test_samples = _samples(frames["test"], vocab, "test", data_cfg)
    train_counts = poi_sample_counts(train_samples, vocab.num_pois)

    model = _build_model(vocab, cfg["model"])
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

    result = {"ranking": metrics, "long_tail": long_tail, "split_protocol": "massive_steps_official"}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = PKG_ROOT / out
    write_json(out, result)
    print(result)


if __name__ == "__main__":
    main()
