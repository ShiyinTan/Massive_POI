#!/usr/bin/env python3
"""Download official Massive-STEPS splits and build leakage-safe examples + SURD profiles.

本脚本的唯一工作是调用 ``prepare_artifacts``（见 ``pc_smir/pipeline.py``）：
离线产出 vocab / examples / SURD 等，不训练模型。

为何 train.py 也会再调一次 prepare_artifacts？
  - 这里：方便单独准备/检查/强制重建（``--force``）产物。
  - train.py：保证“只跑训练命令”也能自给自足；若产物已在，函数直接命中缓存返回。
  两次调用不是重复重算，而是“可拆分入口 + 训练入口自包含”。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_smir_sth_aligned.pipeline import prepare_artifacts
from pc_smir_sth_aligned.utils.config import load_config, print_config, save_config
from pc_smir_sth_aligned.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="YAML configs, merged left-to-right")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_paths = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.config]
    cfg = load_config(cfg_paths)
    # Resolve relative data/output paths against package root.
    for key in ("data_dir",):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(ROOT / p)
    for key in ("artifact_dir", "run_dir"):
        p = Path(cfg["output"][key])
        if not p.is_absolute():
            cfg["output"][key] = str(ROOT / p)

    print_config(cfg, title="Resolved prepare config")
    set_seed(int(cfg.get("seed", 0)))
    # Standalone prep entry: build or refresh offline artifacts only (no training).
    # 独立准备入口：只构建/刷新离线产物，不训练。
    paths = prepare_artifacts(cfg, force=args.force)
    save_config(cfg, Path(cfg["output"]["artifact_dir"]) / "config_used.yaml")
    print("Prepared artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()