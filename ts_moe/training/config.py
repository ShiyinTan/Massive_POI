"""YAML config with defaults from the spec's initial experimental setup."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 42,
    "data": {
        "city": "beijing",
        "data_dir": None,
        "synthetic": False,
        "max_history": 50,
        "min_history": 1,
        "n_users": 12,
        "n_venues": 10,
        "length": 6,
    },
    "model": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "expert_variant": 21,
        "expert_groups": None,
        "top_k": 4,
        "temperature": 1.0,
        "guidance_scale": 0.1,
        "routing_mode": "surd",
        "max_len": 50,
    },
    "loss": {
        "lambda_route": 0.1,
        "lambda_balance": 0.01,
        "lambda_shared": 0.0,
    },
    "surd": {
        "estimator": "neural",
        "n_folds": 5,
        "alpha": 1.0,
        "kappa": 10.0,
        "eps": 1e-8,
        "n_bins": 8,
        "eval_fraction": 0.2,
        "max_len": 50,
        "neural": {
            "d_model": 32,
            "epochs": 5,
            "batch_size": 256,
            "lr": 1.0e-3,
            "neg_samples": 64,
            "full_softmax_limit": 4096,
        },
    },
    "train": {
        "batch_size": 32,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "epochs": 20,
        "num_negatives": 32,
        "grad_clip": 1.0,
    },
    "eval": {
        "ks": [1, 5, 10, 20],
        "candidate_strategy": "two_stage",
        "recall_m": 100,
        "chunk_size": 256,
    },
    "output_dir": "ts_moe/outputs",
}


def _merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return deepcopy(DEFAULTS)
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping")
    return _merge(DEFAULTS, loaded)
