"""YAML config loading with shallow/deep merges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: List[str | Path]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            # Also try relative to package configs/
            alt = Path(__file__).resolve().parents[1] / "configs" / path.name
            if alt.is_file():
                path = alt
            else:
                raise FileNotFoundError(p)
        with path.open("r", encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overlay)
    return cfg