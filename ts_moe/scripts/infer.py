"""Recommend the next POI for one observed trail.

    python -m ts_moe.scripts.infer --checkpoint ts_moe/outputs/ts_moe_beijing_surd.pt --history_csv trail.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ..inference.predict import predict_history_frame
from ..run import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="TS-MoE next-POI inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history_csv", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.history_csv)
    result = predict_history_frame(
        args.checkpoint,
        frame,
        top_k=args.top_k,
        device=resolve_device(args.device),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
