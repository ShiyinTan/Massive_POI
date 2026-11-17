"""Run a trained TS-MoE checkpoint.

``predict_examples`` scores next-POI candidates with the full-train SURD
prior stored in the checkpoint. ``predict_history_frame`` treats every row
of a one-trail frame as observed history and recommends the next venue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..dataset.build import Example, Vocab, build_examples, make_collate
from ..evaluation.metrics import evaluate_loader
from ..model.ts_moe import TSMoE
from ..training.loop import ExampleDataset, build_model_from_checkpoint, load_checkpoint


def predict_examples(
    checkpoint: str | Path,
    examples: list[Example],
    train_targets: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> dict:
    device = device or torch.device("cpu")
    payload = load_checkpoint(checkpoint, device)
    model, vocab, prior = build_model_from_checkpoint(payload, device)
    model.eval()
    loader = DataLoader(
        ExampleDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(vocab, int(payload["config"]["model"]["max_len"])),
    )
    return evaluate_loader(model, loader, prior, train_targets, payload["config"])


def predict_history_frame(
    checkpoint: str | Path,
    frame: pd.DataFrame,
    *,
    top_k: int = 10,
    device: torch.device | None = None,
) -> dict:
    """Recommend the next POI given one observed trail (no target column used)."""
    device = device or torch.device("cpu")
    payload = load_checkpoint(checkpoint, device)
    vocab = Vocab.from_state_dict(payload["vocab"])
    model, _, prior = build_model_from_checkpoint(payload, device)
    model.eval()
    work = frame.copy()
    if "trail_id" not in work.columns:
        work["trail_id"] = "query"
    if "user_id" not in work.columns:
        raise ValueError("history frame needs user_id")
    # Append a dummy last row so ``build_examples`` emits the full prefix.
    # The dummy target is dropped; we only keep the example whose history
    # length equals the real trail.
    dummy = work.iloc[[-1]].copy()
    known = next(iter(vocab.poi_to_idx))
    dummy["venue_id"] = known
    dummy["timestamp"] = pd.to_datetime(dummy["timestamp"]).iloc[0] + pd.Timedelta(hours=1)
    extended = pd.concat([work, dummy], ignore_index=True)
    from ..dataset.build import _prepare_frame

    prepared = _prepare_frame(extended)
    examples = build_examples(
        prepared,
        vocab,
        max_history=int(payload["config"]["model"]["max_len"]),
        min_history=1,
        n_folds=1,
        seed=0,
        assign_folds=False,
    )
    if not examples:
        raise ValueError("history is empty or the user/venues are outside the training vocab")
    example = examples[-1]
    if len(example.poi) != len(work):
        # The dummy row is the target of the last real prefix, so history
        # length equals the original trail. If some venues were unknown they
        # are id 0 and still count as steps.
        pass
    batch = make_collate(vocab, int(payload["config"]["model"]["max_len"]))([example])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.rank_batch(
            batch,
            prior,
            strategy=str(payload["config"]["eval"]["candidate_strategy"]),
            recall_m=int(payload["config"]["eval"]["recall_m"]),
            chunk_size=int(payload["config"]["eval"]["chunk_size"]),
        )
    scores = out["scores"][0].detach().cpu().numpy()
    candidates = out["candidates"][0].detach().cpu().numpy()
    pi = out["pi"][0].detach().cpu().numpy()
    order = np.argsort(-scores, kind="mergesort")[:top_k]
    idx_to_poi = {idx: poi for poi, idx in vocab.poi_to_idx.items()}
    names = payload.get("expert_names") or []
    rows = []
    for rank, pos in enumerate(order, start=1):
        poi_idx = int(candidates[pos])
        rows.append(
            {
                "rank": rank,
                "poi_id": idx_to_poi.get(poi_idx, str(poi_idx)),
                "score": float(scores[pos]),
                "routing": {
                    names[e] if e < len(names) else str(e): float(pi[pos, e]) for e in range(pi.shape[1])
                },
            }
        )
    return {"recommendations": rows}
