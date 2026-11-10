from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(paths: list[str | Path]) -> dict[str, Any]:
    """Merge YAML configs left-to-right (later files override)."""
    cfg: dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            chunk = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, chunk)
    return cfg


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)