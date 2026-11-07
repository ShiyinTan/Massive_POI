#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the temporal test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surd_guided_poi.data.dataset import NextPOIDataset, collate_next_poi
from surd_guided_poi.data.preprocessing import (
    build_samples,
    build_vocab,
    load_checkins,
    poi_sample_counts,
)
from surd_guided_poi.data.splits import temporal_split_samples
from surd_guided_poi.evaluation.long_tail_metrics import evaluate_by_frequency, frequency_buckets
from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.training.checkpoint import load_checkpoint, write_json
from surd_guided_poi.training.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/dataset_beijing.yaml"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkins_csv", default=None)
    parser.add_argument("--output_json", default="outputs/eval_metrics.json")
    args = parser.parse_args()

    cfg_paths = [Path(c) if Path(c).is_file() else PKG_ROOT / c for c in args.config]
    cfg = load_config(cfg_paths)
    if args.checkins_csv:
        cfg["data"]["checkins_csv"] = args.checkins_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    df = load_checkins(
        checkins_csv=data_cfg.get("checkins_csv"),
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
        hf_filename=data_cfg.get("hf_filename"),
        cache_dir=str(PKG_ROOT / data_cfg.get("cache_dir", "outputs/data_cache")),
    )
    vocab = build_vocab(df)
    samples = build_samples(
        df,
        vocab,
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
    )
    splits = temporal_split_samples(
        samples,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=float(data_cfg["test_ratio"]),
    )
    train_counts = poi_sample_counts(splits["train"], vocab.num_pois)

    model = SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        poi_emb_dim=int(cfg["model"]["poi_emb_dim"]),
        num_layers_history=int(cfg["model"]["num_layers_history"]),
        subset_hidden_dim=int(cfg["model"]["subset_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)

    test_loader = DataLoader(
        NextPOIDataset(splits["test"]),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_next_poi,
    )
    metrics = evaluate_ranking(model, test_loader, device, ks=cfg["eval"]["ks"])

    # Long-tail breakdown
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

    result = {"ranking": metrics, "long_tail": long_tail}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = PKG_ROOT / out
    write_json(out, result)
    print(result)


if __name__ == "__main__":
    main()