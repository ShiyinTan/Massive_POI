#!/usr/bin/env python3
"""Download official Massive-STEPS splits and build leakage-safe examples + SURD profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_smir.pipeline import prepare_artifacts
from pc_smir.utils.config import load_config, save_config
from pc_smir.utils.seed import set_seed


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

    set_seed(int(cfg.get("seed", 0)))
    paths = prepare_artifacts(cfg, force=args.force)
    save_config(cfg, Path(cfg["output"]["artifact_dir"]) / "config_used.yaml")
    print("Prepared artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()