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
from surd_guided_poi.models.full_model import SURDGuidedPOIModel
from surd_guided_poi.training.alternating import AlternatingSURDTrainer
from surd_guided_poi.training.checkpoint import write_json
from surd_guided_poi.training.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(samples_dict, batch_size, num_workers):
    loaders = {}
    for name, samples in samples_dict.items():
        ds = NextPOIDataset(samples)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name in {"train", "estimator_fit"}),
            num_workers=num_workers,
            collate_fn=collate_next_poi,
        )
    return loaders


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

    cfg_paths = []
    for c in args.config:
        p = Path(c)
        if not p.is_file():
            p = PKG_ROOT / c
        cfg_paths.append(p)
    cfg = load_config(cfg_paths)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device

    set_seed(int(cfg.get("seed", 42)))
    device_str = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        if output_dir.parts and output_dir.parts[0] == PKG_ROOT.name:
            output_dir = REPO_ROOT / output_dir
        else:
            output_dir = PKG_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    data_cfg = cfg["data"]
    city = data_cfg["city"]
    cache_dir = PKG_ROOT / data_cfg.get("cache_dir", "outputs/data_cache") / city
    print("Loading Massive-STEPS official train/validation/test CSVs...")
    split_paths = resolve_official_splits(
        city,
        data_dir=data_cfg.get("data_dir"),
        cache_dir=cache_dir,
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
    )
    frames = load_official_split_frames(
        city,
        data_dir=data_cfg.get("data_dir"),
        cache_dir=cache_dir,
        hf_dataset_id=data_cfg.get("hf_dataset_id"),
    )
    assert_official_split_disjoint_trails(frames)
    vocab = build_vocab_from_frames(frames)

    # User features may use all check-ins of that user within the same split frame.
    train_samples = build_samples_from_split(
        frames["train"],
        vocab,
        "train",
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
        user_history_df=frames["train"],
    )
    val_samples = build_samples_from_split(
        frames["val"],
        vocab,
        "val",
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
        user_history_df=frames["val"],
    )
    test_samples = build_samples_from_split(
        frames["test"],
        vocab,
        "test",
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=bool(data_cfg["use_all_prefixes"]),
        user_history_df=frames["test"],
    )

    model_train, estimation = carve_estimator_holdout(
        train_samples, holdout_ratio=float(data_cfg["estimator_holdout_ratio"])
    )
    poi_prior = compute_poi_prior(train_samples, vocab.num_pois)
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
            "train_counts": poi_sample_counts(train_samples, vocab.num_pois).tolist(),
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

    model_cfg = cfg["model"]
    model = SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        num_users=vocab.num_users,
        hidden_dim=int(model_cfg["hidden_dim"]),
        poi_emb_dim=int(model_cfg.get("poi_emb_dim", model_cfg["hidden_dim"])),
        num_layers_history=int(model_cfg.get("num_layers_history", 1)),
        subset_hidden_dim=int(model_cfg.get("subset_hidden_dim", model_cfg["hidden_dim"])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        rnn_type=str(model_cfg.get("rnn_type", "gru")),
        lambda_t=float(model_cfg.get("lambda_t", 0.1)),
        lambda_s=float(model_cfg.get("lambda_s", 100.0)),
        share_poi_embedding=bool(model_cfg.get("share_poi_embedding", True)),
    )

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
        from surd_guided_poi.training.checkpoint import load_checkpoint

        load_checkpoint(best_path, model, map_location=device)
    from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking

    test_metrics = evaluate_ranking(model, loaders["test"], device, ks=cfg["eval"]["ks"])
    write_json(output_dir / "test_metrics.json", test_metrics)
    print("TEST:", test_metrics)
    print("Done. Outputs in", output_dir)


if __name__ == "__main__":
    main()