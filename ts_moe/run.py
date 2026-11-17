"""Shared data, SURD-prior, and seeding entry points for the scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .dataset.build import Vocab, frames_to_examples
from .dataset.load import load_city_frames
from .dataset.synthetic import make_synthetic_frames
from .surd.atoms import expert_names, select_experts
from .surd.prior import PriorBundle, estimate_cross_fit_priors, slice_prior


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(config: dict) -> tuple[Vocab, dict, str]:
    data = config["data"]
    if data.get("synthetic"):
        frames = make_synthetic_frames(
            n_users=int(data.get("n_users", 12)),
            n_venues=int(data.get("n_venues", 10)),
            length=int(data.get("length", 6)),
            seed=int(config["seed"]),
        )
        city = "synthetic"
    else:
        frames = load_city_frames(str(data["city"]), data.get("data_dir"))
        city = str(data["city"])
    vocab, splits = frames_to_examples(
        frames,
        max_history=int(data["max_history"]),
        min_history=int(data["min_history"]),
        n_folds=int(config["surd"]["n_folds"]),
        seed=int(config["seed"]),
    )
    if not splits["train"]:
        raise RuntimeError("Training split produced no next-POI examples.")
    return vocab, splits, city


def _surd_config(config: dict, device: torch.device) -> dict:
    cfg = dict(config["surd"])
    cfg["seed"] = int(config["seed"])
    cfg["device"] = str(device)
    cfg["max_len"] = int(config["model"]["max_len"])
    cfg["neural"] = dict(config["surd"]["neural"])
    return cfg


def _cache_tag(config: dict, city: str) -> str:
    surd = config["surd"]
    neural = surd["neural"]
    payload = {
        "city": city,
        "estimator": surd["estimator"],
        "n_folds": surd["n_folds"],
        "alpha": surd["alpha"],
        "kappa": surd["kappa"],
        "n_bins": surd["n_bins"],
        "eval_fraction": surd["eval_fraction"],
        "seed": config["seed"],
        "neural_epochs": neural["epochs"],
        "neural_d": neural["d_model"],
    }
    return json.dumps(payload, sort_keys=True)


def prior_cache_path(config: dict, city: str) -> Path:
    return Path(config["output_dir"]) / f"surd_prior_{city}_{config['surd']['estimator']}.npz"


def estimate_active_prior(
    config: dict,
    vocab: Vocab,
    train_examples,
    city: str,
    device: torch.device,
    *,
    reestimate: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return fold priors and the full-train prior sliced to the active experts."""
    path = prior_cache_path(config, city)
    tag = _cache_tag(config, city)
    bundle = None
    if path.is_file() and not reestimate:
        with np.load(path, allow_pickle=False) as payload:
            saved = payload["meta"].item() if "meta" in payload else ""
            if saved == tag:
                bundle = PriorBundle(
                    fold_priors=payload["fold_priors"],
                    full_prior=payload["full_prior"],
                    expert_names=[str(x) for x in payload["expert_names"].tolist()],
                )
    if bundle is None:
        print(f"Estimating offline SURD prior ({config['surd']['estimator']})...", flush=True)
        bundle = estimate_cross_fit_priors(
            train_examples,
            vocab.n_pois,
            vocab.n_users,
            _surd_config(config, device),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            fold_priors=bundle.fold_priors,
            full_prior=bundle.full_prior,
            expert_names=np.array(bundle.expert_names),
            meta=np.array(tag),
        )
        print(f"Cached SURD prior at {path}", flush=True)

    indices, specs = select_experts(
        variant=int(config["model"]["expert_variant"]),
        groups=config["model"].get("expert_groups"),
    )
    names = expert_names(specs)
    eps = float(config["surd"]["eps"])
    return slice_prior(bundle.fold_priors, indices, eps), slice_prior(bundle.full_prior, indices, eps), names


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
