"""Estimate and cache the offline POI-conditioned SURD prior.

    python -m ts_moe.scripts.estimate_surd --config ts_moe/configs/base.yaml
"""

from __future__ import annotations

import argparse

from ..run import estimate_active_prior, load_splits, resolve_device, set_seed
from ..training.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline target-conditioned SURD prior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reestimate", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    vocab, splits, city = load_splits(config)
    fold_priors, full_prior, names = estimate_active_prior(
        config,
        vocab,
        splits["train"],
        city,
        device,
        reestimate=args.reestimate,
    )
    print(
        f"city={city} pois={vocab.n_pois} experts={len(names)} "
        f"fold_priors={tuple(fold_priors.shape)} full_prior={tuple(full_prior.shape)}"
    )


if __name__ == "__main__":
    main()
