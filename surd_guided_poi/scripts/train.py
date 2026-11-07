#!/usr/bin/env python3
"""Train SURD-guided next-POI on Massive-STEPS official splits + Flashback backbone."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow ``python scripts/train.py`` from anywhere: put repo root on path.
PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surd_guided_poi.data.dataset import NextPOIDataset, collate_next_poi
from surd_guided_poi.data.massive_steps import load_official_split_frames, resolve_official_splits
from surd_guided_poi.data.preprocessing import (
    build_samples_from_split,
    build_vocab_from_frames,
    compute_poi_prior,
    poi_sample_counts,
)
from surd_guided_poi.data.splits import assert_official_split_disjoint_trails, carve_estimator_holdout
from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.training.alternating import AlternatingSURDTrainer
from surd_guided_poi.training.checkpoint import load_checkpoint, write_json
from surd_guided_poi.training.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config_paths(config_args: list[str]) -> list[Path]:
    paths = []
    for c in config_args:
        p = Path(c)
        paths.append(p if p.is_file() else PKG_ROOT / c)
    return paths


def resolve_output_dir(raw: str | Path) -> Path:
    """Relative paths land under ``surd_guided_poi/`` unless they already start with that name."""
    output_dir = Path(raw)
    if output_dir.is_absolute():
        return output_dir
    if output_dir.parts and output_dir.parts[0] == PKG_ROOT.name:
        return REPO_ROOT / output_dir
    return PKG_ROOT / output_dir


def build_split_samples(frame, vocab, split_name: str, data_cfg: dict):
    return build_samples_from_split(
        frame,
        vocab,
        split_name,
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
        user_history_df=frame,
    )


def build_loaders(samples_dict, batch_size, num_workers):
    loaders = {}
    for name, samples in samples_dict.items():
        loaders[name] = DataLoader(
            NextPOIDataset(samples),
            batch_size=batch_size,
            shuffle=(name in {"train", "estimator_fit"}),
            num_workers=num_workers,
            collate_fn=collate_next_poi,
        )
    return loaders


def build_model(vocab, model_cfg: dict) -> SURDGuidedPOIModel:
    hidden = int(model_cfg["hidden_dim"])
    return SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        num_users=vocab.num_users,
        hidden_dim=hidden,
        poi_emb_dim=int(model_cfg.get("poi_emb_dim", hidden)),
        num_layers_history=int(model_cfg.get("num_layers_history", 1)),
        subset_hidden_dim=int(model_cfg.get("subset_hidden_dim", hidden)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        rnn_type=str(model_cfg.get("rnn_type", "gru")),
        lambda_t=float(model_cfg.get("lambda_t", 0.1)),
        lambda_s=float(model_cfg.get("lambda_s", 100.0)),
        share_poi_embedding=bool(model_cfg.get("share_poi_embedding", True)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/dataset_beijing.yaml", "configs/model_surd.yaml"],
    )
    parser.add_argument("--data_dir", default=None, help="Directory with official Massive-STEPS split CSVs")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_config_paths(args.config))
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = resolve_output_dir(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    data_cfg = cfg["data"]
    city = data_cfg["city"]
    cache_dir = PKG_ROOT / data_cfg.get("cache_dir", "outputs/data_cache") / city
    split_kwargs = dict(
        data_dir=data_cfg.get("data_dir"),
        cache_dir=cache_dir,
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
    )

    # Pipeline sketch / 数据流示意:
    #   CSV rows → frames["train"|"val"|"test"] (DataFrame, 1 row = 1 check-in)
    #            → NextPOISample list (history → target_poi_id)
    #            → DataLoader batches (padded tensors)
    # No default filter for users with total sequence length < 20.
    print("Loading Massive-STEPS official train/validation/test CSVs...")
    split_paths = resolve_official_splits(city, **split_kwargs)
    frames = load_official_split_frames(city, **split_kwargs)
    assert_official_split_disjoint_trails(frames)
    vocab = build_vocab_from_frames(frames)

    train_samples = build_split_samples(frames["train"], vocab, "train", data_cfg)
    val_samples = build_split_samples(frames["val"], vocab, "val", data_cfg)
    test_samples = build_split_samples(frames["test"], vocab, "test", data_cfg)

    model_train, estimation = carve_estimator_holdout(
        train_samples, holdout_ratio=float(data_cfg["estimator_holdout_ratio"])
    )
    # prior = N_y/N（概率，给 MI/SURD）；counts = N_y（整数，给日志/长尾）
    poi_prior = compute_poi_prior(train_samples, vocab.num_pois)
    train_counts = poi_sample_counts(train_samples, vocab.num_pois)
    write_json(
        output_dir / "data_summary.json",
        {
            "city": city,
            "split_protocol": "massive_steps_official_trail_stratified",
            "split_paths": {k: str(v) for k, v in split_paths.items()},
            "backbone": "Flashback (Massive-STEPS/Flashback)",
            "num_pois": vocab.num_pois,
            "num_users": vocab.num_users,
            "n_train": len(model_train),
            "n_estimation": len(estimation),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
            "setting": data_cfg.get("setting", "transductive"),
            "train_counts": train_counts.tolist(),  # 原始 N_y，不是 prior
        },
    )
    print(
        f"city={city} POIs={vocab.num_pois} users={vocab.num_users} "
        f"train={len(model_train)} est={len(estimation)} "
        f"val={len(val_samples)} test={len(test_samples)}"
    )
    print("split files:", {k: str(v) for k, v in split_paths.items()})

    loaders = build_loaders(
        {
            "train": model_train,
            "estimator_fit": model_train,
            "estimation": estimation,
            "val": val_samples,
            "test": test_samples,
        },
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model = build_model(vocab, cfg["model"])
    trainer = AlternatingSURDTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        estimator_fit_loader=loaders["estimator_fit"],
        estimation_loader=loaders["estimation"],
        poi_prior=poi_prior,
        config=cfg,
        device=device,
        output_dir=output_dir,
    )
    trainer.fit()

    best_path = output_dir / "best.pt"
    if best_path.is_file():
        load_checkpoint(best_path, model, map_location=device)
    test_metrics = evaluate_ranking(model, loaders["test"], device, ks=cfg["eval"]["ks"])
    write_json(output_dir / "test_metrics.json", test_metrics)
    print("TEST:", test_metrics)
    print("Done. Outputs in", output_dir)


if __name__ == "__main__":
    main()
