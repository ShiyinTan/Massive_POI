#!/usr/bin/env python3
"""Convenience entrypoint: prepare → train → evaluate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    configs = sys.argv[1:] or ["configs/base.yaml", "configs/dataset_beijing.yaml", "configs/smoke.yaml"]
    cmd_prep = [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--config", *configs, "--force"]
    cmd_train = [sys.executable, str(ROOT / "scripts/train.py"), "--config", *configs]
    print("Running:", " ".join(cmd_prep))
    subprocess.check_call(cmd_prep, cwd=ROOT)
    print("Running:", " ".join(cmd_train))
    subprocess.check_call(cmd_train, cwd=ROOT)


if __name__ == "__main__":
    main()