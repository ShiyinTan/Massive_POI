#!/usr/bin/env python3
"""Train SURD-guided next-POI on Massive-STEPS official splits.

Backbone selected by ``model.backbone``: flashback | graph_flashback | sthgcn.
"""

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

from surd_guided_poi.data.cold_filter import (
    maybe_drop_unseen_user_poi,
    seen_user_poi_ids_from_train_frame,
)
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
from surd_guided_poi.models.full_model import build_surd_model
from surd_guided_poi.training.alternating import AlternatingSURDTrainer
from surd_guided_poi.training.checkpoint import load_checkpoint, write_json
from surd_guided_poi.training.config import load_config
from surd_guided_poi.training.log_format import format_metrics, log_block


def drop_unseen_user_poi_enabled(cfg: dict) -> bool:
    """STHGCN-style cold filter; Flashback/Graph-Flashback leave this off."""
    eval_cfg = cfg.get("eval") or {}
    data_cfg = cfg.get("data") or {}
    if "drop_unseen_user_poi" in eval_cfg:
        return bool(eval_cfg["drop_unseen_user_poi"])
    if "drop_unseen_user_poi" in data_cfg:
        return bool(data_cfg["drop_unseen_user_poi"])
    return str(cfg.get("model", {}).get("backbone", "flashback")).lower() == "sthgcn"


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


def build_split_samples(
    frame,
    vocab,
    split_name: str,
    data_cfg: dict,
    *,
    use_all_prefixes: bool | None = None,
):
    # 每个 split 单独建样本：序列 history = 该 CSV 内同 trail 前缀（无跨 split dump）。
    # user_history_df=frame → 6 维 user_features 也只看本 split 且 timestamp < target。
    # val/test: prefer eval.use_all_prefixes=false (last POI only = Massive-STEPS Acc@1).
    prefixes = (
        bool(data_cfg["use_all_prefixes"]) if use_all_prefixes is None else bool(use_all_prefixes)
    )
    return build_samples_from_split(
        frame,
        vocab,
        split_name,
        max_history_len=int(data_cfg["max_history_len"]),
        min_history_len=int(data_cfg["min_history_len"]),
        use_all_prefixes=prefixes,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            "configs/base.yaml",
            "configs/dataset_beijing.yaml",
            "configs/model_flashback_aligned.yaml",
        ],
        help="Default: Flashback-aligned Beijing (SURD off). Use model_surd.yaml to enable SURD.",
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

    eval_cfg = cfg.get("eval", {})
    # Train: all prefixes (default). Val/test: last-POI-only for Massive-STEPS Acc@1 parity.
    eval_prefixes = bool(eval_cfg.get("use_all_prefixes", False))
    drop_cold = drop_unseen_user_poi_enabled(cfg)
    train_samples = build_split_samples(frames["train"], vocab, "train", data_cfg)
    val_samples = build_split_samples(
        frames["val"], vocab, "val", data_cfg, use_all_prefixes=eval_prefixes
    )
    test_samples = build_split_samples(
        frames["test"], vocab, "test", data_cfg, use_all_prefixes=eval_prefixes
    )
    n_val_before, n_test_before = len(val_samples), len(test_samples)
    train_users, train_pois = seen_user_poi_ids_from_train_frame(frames["train"], vocab)
    val_samples = maybe_drop_unseen_user_poi(
        val_samples, train_user_ids=train_users, train_poi_ids=train_pois, enabled=drop_cold
    )
    test_samples = maybe_drop_unseen_user_poi(
        test_samples, train_user_ids=train_users, train_poi_ids=train_pois, enabled=drop_cold
    )

    # carve：train 内再切 — model_train 训推荐；estimation 只给 Phase A 估 SURD
    # Baseline-only (use_surd=false): holdout still carved but Phase A is skipped.
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
            "backbone": str(cfg.get("model", {}).get("backbone", "flashback")),
            "num_pois": vocab.num_pois,
            "num_users": vocab.num_users,
            "n_train": len(model_train),
            "n_estimation": len(estimation),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
            "n_val_before_cold_filter": n_val_before,
            "n_test_before_cold_filter": n_test_before,
            "drop_unseen_user_poi": drop_cold,
            "eval_use_all_prefixes": eval_prefixes,
            "use_surd": bool(cfg.get("train", {}).get("use_surd", True)),
            "setting": data_cfg.get("setting", "transductive"),
            "train_counts": train_counts.tolist(),  # 原始 N_y，不是 prior
        },
    )
    print(
        log_block(
            f"data  {city}",
            [
                f"POIs={vocab.num_pois}  users={vocab.num_users}",
                f"train={len(model_train)}  est={len(estimation)}  "
                f"val={len(val_samples)}  test={len(test_samples)}",
                f"train_prefixes={bool(data_cfg.get('use_all_prefixes', True))}  "
                f"eval_prefixes={eval_prefixes}  "
                f"drop_cold={drop_cold}  "
                f"use_surd={bool(cfg.get('train', {}).get('use_surd', True))}  "
                f"backbone={cfg.get('model', {}).get('backbone', 'flashback')}  "
                f"rnn={cfg.get('model', {}).get('rnn_type')}  "
                f"hidden={cfg.get('model', {}).get('hidden_dim')}",
                f"device={device}",
            ],
        )
    )

    loaders = build_loaders(
        {
            "train": model_train,  # 推荐主训
            "estimator_fit": model_train,  # Phase A：在这批上拟合子集头
            "estimation": estimation,  # Phase A：在 holdout 上估 MI/SURD（不反传）
            "val": val_samples,
            "test": test_samples,
        },
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    # Graph-Flashback builds count graphs from full official-train samples
    # (before carve) unless transition_pkl / interact_pkl are set.
    model = build_surd_model(vocab, cfg["model"], train_samples=train_samples)
    trainer = AlternatingSURDTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],  # 每 epoch 可选打 test（见 eval.eval_test_every_epoch）
        estimator_fit_loader=loaders["estimator_fit"],
        estimation_loader=loaders["estimation"],  # 来自 carve 的 holdout
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
    print(
        log_block(
            "TEST",
            [format_metrics(test_metrics), f"outputs → {output_dir}"],
        )
    )


if __name__ == "__main__":
    main()
