"""Train TS-MoE: offline SURD prior, then joint CE + routing KL + load balance.

    python -m ts_moe.scripts.train --config ts_moe/configs/smoke.yaml
    python -m ts_moe.scripts.train --config ts_moe/configs/base.yaml --routing-mode plain
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..model.ts_moe import TSMoE
from ..run import estimate_active_prior, load_splits, resolve_device, set_seed
from ..training.config import load_config
from ..training.loop import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TS-MoE next-POI recommendation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reestimate-surd", action="store_true")
    parser.add_argument("--routing-mode", choices=["surd", "plain", "random", "fixed"], default=None)
    parser.add_argument("--expert-variant", type=int, choices=[4, 10, 21], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.routing_mode:
        config["model"]["routing_mode"] = args.routing_mode
    if args.expert_variant:
        config["model"]["expert_variant"] = args.expert_variant
        config["model"]["expert_groups"] = None
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    print(f"device={device} routing={config['model']['routing_mode']} experts={config['model']['expert_variant']}", flush=True)
    vocab, splits, city = load_splits(config)
    fold_priors, full_prior, names = estimate_active_prior(
        config,
        vocab,
        splits["train"],
        city,
        device,
        reestimate=args.reestimate_surd,
    )
    model = TSMoE(vocab, config)
    trainer = Trainer(
        model,
        vocab,
        config,
        fold_priors,
        full_prior,
        splits["train"],
        splits.get("val"),
        device,
    )
    out_dir = Path(config["output_dir"])
    checkpoint = out_dir / f"ts_moe_{city}_{config['model']['routing_mode']}.pt"
    trainer.fit(checkpoint, expert_names=names)
    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
