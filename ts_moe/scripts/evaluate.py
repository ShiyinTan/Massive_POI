"""Evaluate a TS-MoE checkpoint on the test split.

    python -m ts_moe.scripts.evaluate --checkpoint ts_moe/outputs/ts_moe_synthetic_surd.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..inference.predict import predict_examples
from ..run import load_splits, resolve_device, set_seed
from ..training.loop import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TS-MoE Recall / NDCG / routing")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="JSON path for the metric report")
    args = parser.parse_args()
    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint, device)
    config = payload["config"]
    set_seed(int(config["seed"]))
    _, splits, city = load_splits(config)
    train_targets = np.array([ex.target for ex in splits["train"]], dtype=np.int64)
    metrics = predict_examples(
        args.checkpoint,
        splits["test"],
        train_targets,
        device=device,
        batch_size=int(config["train"]["batch_size"]),
    )
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    output = Path(args.output) if args.output else Path(config["output_dir"]) / f"eval_{city}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
