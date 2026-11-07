#!/usr/bin/env python3
"""Train SURD-guided next-POI recommendation (3 sources, alternating MI/SURD)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running as `python scripts/train.py` from surd_guided_poi/.
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
    poi_sample_counts,
)
from surd_guided_poi.data.splits import carve_estimator_holdout, temporal_split_samples
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
        help="YAML configs merged in order",
    )
    parser.add_argument("--checkins_csv", default=None, help="Override local check-ins CSV path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Resolve config paths relative to package root when needed.
    cfg_paths = []
    for c in args.config:
        p = Path(c)
        if not p.is_file():
            p = PKG_ROOT / c
        cfg_paths.append(p)
    cfg = load_config(cfg_paths)

    if args.checkins_csv:
        cfg["data"]["checkins_csv"] = args.checkins_csv
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.device:
        cfg["device"] = args.device

    set_seed(int(cfg.get("seed", 42)))
    device_str = cfg.get("device")
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        # Prefer package-local outputs/… paths; avoid double-prefixing repo-relative paths.
        if output_dir.parts and output_dir.parts[0] == PKG_ROOT.name:
            output_dir = REPO_ROOT / output_dir
        else:
            output_dir = PKG_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_resolved.json", cfg)

    data_cfg = cfg["data"]
    print("Loading check-ins...")
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
    # Prior from full train period (including estimation holdout) — still earlier than val/test.
    poi_prior = compute_poi_prior(splits["train"], vocab.num_pois)
    write_json(
        output_dir / "data_summary.json",
        {
            "num_pois": vocab.num_pois,
            "num_users": vocab.num_users,
            "n_samples": len(samples),
            "n_train": len(model_train),
            "n_estimation": len(estimation),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "setting": data_cfg.get("setting", "transductive"),
            "split": "temporal_by_target_timestamp",
            "train_counts": poi_sample_counts(splits["train"], vocab.num_pois).tolist(),
        },
    )
    print(
        f"POIs={vocab.num_pois} samples={len(samples)} "
        f"train={len(model_train)} est={len(estimation)} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )

    loaders = build_loaders(
        {
            "train": model_train,
            "estimator_fit": model_train,
            "estimation": estimation,
            "val": splits["val"],
            "test": splits["test"],
        },
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model = SURDGuidedPOIModel(
        num_pois=vocab.num_pois,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        poi_emb_dim=int(cfg["model"]["poi_emb_dim"]),
        num_layers_history=int(cfg["model"]["num_layers_history"]),
        subset_hidden_dim=int(cfg["model"]["subset_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
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
    result = trainer.fit()

    # Final test evaluation with best checkpoint if present.
    best_path = output_dir / "best.pt"
    if best_path.is_file():
        from surd_guided_poi.training.checkpoint import load_checkpoint

        load_checkpoint(best_path, model, map_location=device)
    from surd_guided_poi.evaluation.ranking_metrics import evaluate_ranking

    test_metrics = evaluate_ranking(model, loaders["test"], device, ks=cfg["eval"]["ks"])
    write_json(output_dir / "test_metrics.json", test_metrics)
    print("TEST:", test_metrics)
    print("Done. Outputs in", output_dir)
    return result


if __name__ == "__main__":
    main()