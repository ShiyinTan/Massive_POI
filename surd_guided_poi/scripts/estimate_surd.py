#!/usr/bin/env python3
"""Fit subset heads from a checkpoint and dump Specific MI + SURD artifacts."""

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
    compute_poi_prior,
)
from surd_guided_poi.data.splits import carve_estimator_holdout
from surd_guided_poi.models.full_model import build_surd_model
from surd_guided_poi.training.alternating import AlternatingSURDTrainer
from surd_guided_poi.training.checkpoint import load_checkpoint
from surd_guided_poi.training.config import load_config


def _config_paths(args_config: list[str]) -> list[Path]:
    return [Path(c) if Path(c).is_file() else PKG_ROOT / c for c in args_config]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/dataset_beijing.yaml"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/estimate_surd")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(_config_paths(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    cfg["output_dir"] = args.output_dir

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
    train_samples = build_samples_from_split(
        frames["train"],
        vocab,
        "train",
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
    )
    # 与 train.py 相同：model_train 拟合子集头；estimation 只估 SURD
    model_train, estimation = carve_estimator_holdout(
        train_samples, holdout_ratio=float(data_cfg["estimator_holdout_ratio"])
    )
    # P(Y=y)=N_y/N，给 Specific MI / global SURD；不是原始计数
    poi_prior = compute_poi_prior(train_samples, vocab.num_pois)

    def make_loader(samples, shuffle=False):
        return DataLoader(
            NextPOIDataset(samples),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=shuffle,
            collate_fn=collate_next_poi,
        )

    model = build_surd_model(vocab, cfg["model"], train_samples=train_samples)
    load_checkpoint(args.checkpoint, model, map_location=device)

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = PKG_ROOT / out
    trainer = AlternatingSURDTrainer(
        model=model,
        train_loader=make_loader(model_train, shuffle=True),
        val_loader=make_loader(model_train),
        estimator_fit_loader=make_loader(model_train, shuffle=True),  # carve 的 model_train
        estimation_loader=make_loader(estimation),  # carve 的 holdout → 只估 SURD
        poi_prior=poi_prior,
        config=cfg,
        device=device,
        output_dir=out,
    )
    trainer.refresh_surd(tag="manual")
    print("Wrote SURD artifacts under", out / "surd" / "manual")


if __name__ == "__main__":
    main()
