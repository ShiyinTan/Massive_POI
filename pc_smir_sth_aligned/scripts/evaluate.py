#!/usr/bin/env python3
"""Evaluate a PC-SMIR checkpoint on the official validation/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_smir_sth_aligned.pipeline import build_dataloaders, build_model, prepare_artifacts
from pc_smir_sth_aligned.training.trainer import Trainer
from pc_smir_sth_aligned.utils.config import load_config
from pc_smir_sth_aligned.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config([ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.config])
    for key in ("data_dir",):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(ROOT / p)
    for key in ("artifact_dir", "run_dir"):
        p = Path(cfg["output"][key])
        if not p.is_absolute():
            cfg["output"][key] = str(ROOT / p)

    set_seed(int(cfg.get("seed", 0)))
    paths = prepare_artifacts(cfg, force=False)
    train_loader, val_loader, test_loader, vocab, surd_dim, _ = build_dataloaders(cfg, paths)
    model = build_model(cfg, vocab, surd_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=1e-3), device, cfg["train"], Path(cfg["output"]["run_dir"]))
    loader = test_loader if args.split == "test" else val_loader
    metrics = trainer.evaluate(loader)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()