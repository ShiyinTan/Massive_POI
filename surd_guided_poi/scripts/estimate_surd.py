#!/usr/bin/env python3
"""Fit subset heads from a checkpoint and dump Specific MI + SURD artifacts."""

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
    compute_poi_prior,
    load_checkins,
)
from surd_guided_poi.data.splits import carve_estimator_holdout, temporal_split_samples
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.training.alternating import AlternatingSURDTrainer
from surd_guided_poi.training.checkpoint import load_checkpoint
from surd_guided_poi.training.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/dataset_beijing.yaml"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/estimate_surd")
    parser.add_argument("--checkins_csv", default=None)
    args = parser.parse_args()

    cfg_paths = []
    for c in args.config:
        p = Path(c)
        if not p.is_file():
            p = PKG_ROOT / c
        cfg_paths.append(p)
    cfg = load_config(cfg_paths)
    if args.checkins_csv:
        cfg["data"]["checkins_csv"] = args.checkins_csv
    cfg["output_dir"] = args.output_dir

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
    model_train, estimation = carve_estimator_holdout(
        splits["train"], holdout_ratio=float(data_cfg["estimator_holdout_ratio"])
    )
    poi_prior = compute_poi_prior(splits["train"], vocab.num_pois)

    def loader(ss, shuffle=False):
        return DataLoader(
            NextPOIDataset(ss),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=shuffle,
            collate_fn=collate_next_poi,
        )

    model = SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        poi_emb_dim=int(cfg["model"]["poi_emb_dim"]),
        num_layers_history=int(cfg["model"]["num_layers_history"]),
        subset_hidden_dim=int(cfg["model"]["subset_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    load_checkpoint(args.checkpoint, model, map_location=device)

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = PKG_ROOT / out
    trainer = AlternatingSURDTrainer(
        model=model,
        train_loader=loader(model_train, True),
        val_loader=loader(splits["val"]),
        estimator_fit_loader=loader(model_train, True),
        estimation_loader=loader(estimation),
        poi_prior=poi_prior,
        config=cfg,
        device=device,
        output_dir=out,
    )
    trainer.refresh_surd(tag="manual")
    print("Wrote SURD artifacts under", out / "surd" / "manual")


if __name__ == "__main__":
    main()