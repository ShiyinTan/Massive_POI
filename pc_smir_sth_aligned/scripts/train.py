#!/usr/bin/env python3
"""Train PC-SMIR on Massive-STEPS official splits.

训练前会调用一次 ``prepare_artifacts``（与 ``scripts/prepare_data.py`` 相同函数）：
  - 若已跑过 prepare_data，这里通常直接复用 artifacts（不重算）。
  - 若没跑过，这里会现场准备，这样 ``train.py`` 可单独作为端到端入口。
用 ``--force-prep`` 才强制重建；默认不要把“两个脚本都调 prepare”理解成做了两遍重活。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_smir_sth_aligned.evaluation.metrics import format_metrics_table
from pc_smir_sth_aligned.pipeline import build_dataloaders, build_model, prepare_artifacts
from pc_smir_sth_aligned.training.trainer import Trainer
from pc_smir_sth_aligned.utils.config import load_config, print_config, save_config
from pc_smir_sth_aligned.utils.seed import set_seed


def _resolve(cfg):
    for key in ("data_dir",):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(ROOT / p)
    for key in ("artifact_dir", "run_dir"):
        p = Path(cfg["output"][key])
        if not p.is_absolute():
            cfg["output"][key] = str(ROOT / p)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--force-prep", action="store_true")
    args = parser.parse_args()

    cfg = _resolve(load_config([ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.config]))
    print_config(cfg, title="Resolved train config")
    set_seed(int(cfg.get("seed", 0)))

    # Same prepare_artifacts as prepare_data.py: ensure offline artifacts exist (cached if present).
    # 与 prepare_data.py 共用准备函数：产物已在则秒返回；--force-prep 才重建。
    paths = prepare_artifacts(cfg, force=args.force_prep)
    train_loader, val_loader, test_loader, vocab, surd_dim, catalog = build_dataloaders(cfg, paths)
    model = build_model(cfg, vocab, surd_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    trainer = Trainer(model, optim, device, cfg["train"], run_dir)
    best = trainer.fit(
        train_loader,
        val_loader,
        epochs=int(cfg["train"]["epochs"]),
        select_metric=str(cfg["train"].get("select_metric", "recall@10")),
        test_loader=test_loader,
    )
    # Reload best checkpoint and report final val + test (selection still on val).
    ckpt = torch.load(best, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics = trainer.evaluate(val_loader, desc="val")
    test_metrics = trainer.evaluate(test_loader, desc="test")
    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(
        format_metrics_table(
            {"val": val_metrics, "test": test_metrics},
            title=f"Best checkpoint (epoch {ckpt.get('epoch', '?')})  (Acc@K = Recall@K)",
        )
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()