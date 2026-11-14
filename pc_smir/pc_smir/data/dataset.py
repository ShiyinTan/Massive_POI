"""Leakage-safe next-POI example construction from official trail splits.

从官方 trail 划分构造无泄漏的 next-POI 样本。

职责边界 / Ownership:
  - 本模块拥有 example 生命周期：trail → ``build_prefix_example`` →
    ``build_examples`` → ``NextPOIDataset`` / collate。
  - Q1–Q4 **离散化**（SURD 用）在 ``pc_smir.features.sources.SourceDiscretizer``，
    不混在本文件。

Pipeline timing / 调用时机:
  prepare  (prepare_artifacts):
      trails_from_split → build_examples  → 序列化到 examples.pkl
  train/eval (build_dataloaders):
      NextPOIDataset + DataLoader(collate_fn=collate_variable_candidates)
      → Trainer 每个 step 取一个 batch

Beijing official train 规模（注释中的数字）:
  ~400 trails → ~400 examples；catalog≈498 POIs；smoke max_len=12/20。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pc_smir.data.vocab import VocabMaps, haversine_m, spatial_cell_id
from pc_smir.features.stable_user import stable_user_features
from pc_smir.features.temporal import query_time_features


def trails_from_split(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group check-in rows into trails with ≥2 events."""
    trails = []
    for (user_id, trail_id), group in df.groupby(["user_id", "trail_id"], sort=False):
        group = group.sort_values(["timestamp", "venue_id"])
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        trails.append({"user_id": int(user_id), "trail_id": str(trail_id), "rows": rows})
    return trails


def _log1p_hour_gaps(times: list) -> list[float]:
    """Consecutive inter-check-in gaps in log1p(hours); first event gap=0."""
    gaps = [0.0]
    for i in range(1, len(times)):
        gap = max(0.0, (times[i] - times[i - 1]).total_seconds() / 3600.0)
        gaps.append(float(np.log1p(gap)))
    return gaps


def build_prefix_example(
    user_id: int,
    trail_id: str,
    hist_rows: list[dict],
    target_venue_id: int,
    vocab: VocabMaps,
    max_len: int,
) -> dict | None:
    """Build one next-POI example from trail prefix (excludes target event).

    Prediction timestamp = last observed check-in (not target timestamp).
    """
    if len(hist_rows) < 1:
        return None
    hist_rows = hist_rows[-max_len:]
    poi_ids = [vocab.encode_poi(r["venue_id"]) for r in hist_rows]
    cat_ids = [vocab.encode_cat(r["venue_category"]) for r in hist_rows]
    coords = [(float(r["latitude"]), float(r["longitude"])) for r in hist_rows]
    times = [r["timestamp"] for r in hist_rows]
    return {
        "user_id": int(user_id),
        "trail_id": str(trail_id),
        "history_poi_ids": poi_ids,
        "history_categories": cat_ids,
        "history_coordinates": coords,
        "history_cells": [spatial_cell_id(a, b, vocab) for a, b in coords],
        "history_time_gaps": _log1p_hour_gaps(times),
        "history_hours": [int(t.hour) for t in times],
        "history_dows": [int(t.dayofweek) for t in times],
        "query_time": times[-1],
        "stable_user_features": stable_user_features(poi_ids, cat_ids, coords, vocab),
        "target_poi_id": int(vocab.encode_poi(target_venue_id)),
        "target_venue_raw": int(target_venue_id),
        "last_lat": coords[-1][0],
        "last_lon": coords[-1][1],
        "vocab": vocab,
        "source_available": np.ones(4, dtype=np.float32),
    }


def train_seen_user_poi_sets(train_df: pd.DataFrame) -> tuple[set[int], set[str]]:
    """Raw user_id / venue_id sets observed in the training split."""
    users = {int(u) for u in train_df["user_id"].tolist()}
    pois = {str(v) for v in train_df["venue_id"].tolist()}
    return users, pois


def history_rows_before_target(
    all_df: pd.DataFrame,
    user_id: int,
    trail_id: str,
    target_ts: Any,
    target_venue_id: int,
    max_len: int,
) -> list[dict]:
    """All of a user's check-ins strictly before the target event (cross-trail)."""
    udf = all_df.loc[all_df["user_id"].astype(int) == int(user_id)].sort_values(
        ["timestamp", "venue_id", "trail_id"]
    )
    rows = udf.to_dict("records")
    trail_key = str(trail_id)
    target_idx = next(
        (
            i
            for i, r in enumerate(rows)
            if (
                str(r["trail_id"]) == trail_key
                and r["timestamp"] == target_ts
                and int(r["venue_id"]) == int(target_venue_id)
            )
        ),
        None,
    )
    if target_idx is None or target_idx < 1:
        return []
    hist = rows[:target_idx]
    return hist[-max_len:] if len(hist) > max_len else hist


def _passes_sthgcn_filters(
    user_id: int,
    target_venue_id: Any,
    train_user_ids: set[int] | None,
    train_venue_ids: set[str] | None,
) -> bool:
    if train_user_ids is not None and user_id not in train_user_ids:
        return False
    if train_venue_ids is not None and str(target_venue_id) not in train_venue_ids:
        return False
    return True


def _history_for_trail(
    trail: dict[str, Any],
    target_row: dict,
    *,
    sthgcn_protocol: bool,
    all_df: pd.DataFrame | None,
    max_len: int,
) -> list[dict]:
    """Same-trail prefix, or cross-trail temporal history under STHGCN protocol."""
    rows = trail["rows"]
    if sthgcn_protocol and all_df is not None:
        return history_rows_before_target(
            all_df,
            user_id=int(trail["user_id"]),
            trail_id=str(trail["trail_id"]),
            target_ts=target_row["timestamp"],
            target_venue_id=int(target_row["venue_id"]),
            max_len=max_len,
        )
    return rows[:-1]


def build_examples(
    df: pd.DataFrame,
    vocab: VocabMaps,
    max_len: int,
    *,
    sthgcn_protocol: bool = False,
    all_df: pd.DataFrame | None = None,
    train_user_ids: set[int] | None = None,
    train_venue_ids: set[str] | None = None,
) -> list[dict]:
    """Turn each trail into one next-POI dict example (Python, not tensors yet).

    When ``sthgcn_protocol=True`` (eval/val/test path):
      - drop sample if user ∉ train or target POI ∉ train
      - if ``all_df`` given, history = all user check-ins before target (cross-trail)
    """
    examples = []
    for trail in trails_from_split(df):
        rows = trail["rows"]
        target_row = rows[-1]
        user_id = int(trail["user_id"])

        if sthgcn_protocol and not _passes_sthgcn_filters(
            user_id, target_row["venue_id"], train_user_ids, train_venue_ids
        ):
            continue

        hist = _history_for_trail(
            trail, target_row, sthgcn_protocol=sthgcn_protocol, all_df=all_df, max_len=max_len
        )
        ex = build_prefix_example(
            user_id=user_id,
            trail_id=trail["trail_id"],
            hist_rows=hist,
            target_venue_id=int(target_row["venue_id"]),
            vocab=vocab,
            max_len=max_len,
        )
        if ex is not None:
            examples.append(ex)
    return examples


def _right_align_history(ex: dict, vocab: VocabMaps, max_len: int) -> dict[str, np.ndarray]:
    """Right-align history arrays to ``max_len`` with left pad=0 / mask=0."""
    L = max_len
    poi = np.zeros(L, dtype=np.int64)
    cat = np.zeros(L, dtype=np.int64)
    hours = np.zeros(L, dtype=np.int64)
    dows = np.zeros(L, dtype=np.int64)
    gaps = np.zeros(L, dtype=np.float32)
    cells = np.zeros(L, dtype=np.int64)
    coords = np.zeros((L, 2), dtype=np.float32)
    mask = np.zeros(L, dtype=np.float32)

    hist_len = min(len(ex["history_poi_ids"]), L)
    start = L - hist_len
    src0 = len(ex["history_poi_ids"]) - hist_len
    for i in range(hist_len):
        j = start + i
        src = src0 + i
        poi[j] = ex["history_poi_ids"][src]
        cat[j] = ex["history_categories"][src]
        hours[j] = ex["history_hours"][src]
        dows[j] = ex["history_dows"][src]
        gaps[j] = ex["history_time_gaps"][src]
        cells[j] = ex["history_cells"][src] % 5000
        la, lo = ex["history_coordinates"][src]
        coords[j, 0] = (la - vocab.lat_mean) / vocab.lat_std
        coords[j, 1] = (lo - vocab.lon_mean) / vocab.lon_std
        mask[j] = 1.0

    return {
        "history_poi_ids": poi,
        "history_categories": cat,
        "history_hours": hours,
        "history_dows": dows,
        "history_time_gaps": gaps,
        "history_cells": cells,
        "history_coordinates": coords,
        "history_mask": mask,
    }


def _candidate_meta_row(ex: dict, vocab: VocabMaps, pid: int) -> list[float]:
    lat, lon = vocab.poi_coords.get(pid, (ex["last_lat"], ex["last_lon"]))
    dist = haversine_m(ex["last_lat"], ex["last_lon"], lat, lon)
    pop = float(vocab.train_poi_counts.get(pid, 0))
    cat_id = float(vocab.poi_category.get(pid, 1))
    return [
        np.log1p(dist),
        np.log1p(pop),
        cat_id / max(vocab.n_cats, 1),
        (lat - vocab.lat_mean) / vocab.lat_std,
        (lon - vocab.lon_mean) / vocab.lon_std,
    ]


def _candidate_feature_tables(
    ex: dict,
    candidates: list[int],
    vocab: VocabMaps,
    surd_table: dict[int, np.ndarray],
    surd_valid: dict[int, float],
    surd_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build candidate metadata / SURD / validity arrays for one example."""
    cand_meta = []
    cand_surd = []
    cand_valid = []
    for pid in candidates:
        cand_meta.append(_candidate_meta_row(ex, vocab, pid))
        feat = surd_table.get(pid)
        if feat is None:
            cand_surd.append(np.zeros(surd_dim, dtype=np.float32))
            cand_valid.append(0.0)
        else:
            cand_surd.append(np.asarray(feat, dtype=np.float32))
            cand_valid.append(float(surd_valid.get(pid, 1.0)))
    return (
        np.asarray(cand_meta, dtype=np.float32),
        np.asarray(cand_surd, dtype=np.float32),
        np.asarray(cand_valid, dtype=np.float32),
    )


class NextPOIDataset(Dataset):
    """Pad one example into model tensors + build candidate set."""

    def __init__(
        self,
        examples: list[dict],
        vocab: VocabMaps,
        max_len: int,
        catalog_poi_ids: list[int],
        surd_table: dict[int, np.ndarray] | None = None,
        surd_valid: dict[int, float] | None = None,
        surd_dim: int = 1,
        neg_samples: int | None = None,
        seed: int = 0,
        exclude_oov_targets: bool = True,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.catalog = [p for p in catalog_poi_ids if p != 0]
        self.surd_table = surd_table or {}
        self.surd_valid = surd_valid or {}
        self.surd_dim = surd_dim
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)
        self.examples = [
            ex
            for ex in examples
            if not (
                exclude_oov_targets
                and (ex["target_poi_id"] == 1 or ex["target_poi_id"] not in self.catalog)
            )
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def _candidate_set(self, target_id: int) -> list[int]:
        if self.neg_samples is None or self.neg_samples <= 0:
            return list(self.catalog)
        negs = []
        while len(negs) < self.neg_samples:
            p = int(self.rng.choice(self.catalog))
            if p != target_id:
                negs.append(p)
        return [target_id] + negs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        hist = _right_align_history(ex, self.vocab, self.max_len)

        candidates = self._candidate_set(ex["target_poi_id"])
        if ex["target_poi_id"] not in candidates:
            candidates[0] = ex["target_poi_id"]
        target_index = candidates.index(ex["target_poi_id"])

        cand_meta, cand_surd, cand_valid = _candidate_feature_tables(
            ex, candidates, self.vocab, self.surd_table, self.surd_valid, self.surd_dim
        )

        return {
            "history_poi_ids": torch.tensor(hist["history_poi_ids"]),
            "history_categories": torch.tensor(hist["history_categories"]),
            "history_hours": torch.tensor(hist["history_hours"]),
            "history_dows": torch.tensor(hist["history_dows"]),
            "history_time_gaps": torch.tensor(hist["history_time_gaps"]),
            "history_cells": torch.tensor(hist["history_cells"]),
            "history_coordinates": torch.tensor(hist["history_coordinates"]),
            "history_mask": torch.tensor(hist["history_mask"]),
            "query_time_features": torch.tensor(query_time_features(ex["query_time"])),
            "stable_user_features": torch.tensor(ex["stable_user_features"]),
            "source_available": torch.tensor(ex["source_available"]),
            "candidate_poi_ids": torch.tensor(np.asarray(candidates, dtype=np.int64)),
            "candidate_metadata": torch.tensor(cand_meta),
            "candidate_surd": torch.tensor(cand_surd),
            "candidate_surd_valid": torch.tensor(cand_valid),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "target_poi_id": torch.tensor(ex["target_poi_id"], dtype=torch.long),
        }


_FIXED_COLLATE_KEYS = [
    "history_poi_ids",
    "history_categories",
    "history_hours",
    "history_dows",
    "history_time_gaps",
    "history_cells",
    "history_coordinates",
    "history_mask",
    "query_time_features",
    "stable_user_features",
    "source_available",
    "target_index",
    "target_poi_id",
]


def collate_variable_candidates(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad candidate axis to max-C in the batch; add ``candidate_mask``."""
    out: dict[str, torch.Tensor] = {
        k: torch.stack([b[k] for b in batch], dim=0) for k in _FIXED_COLLATE_KEYS
    }
    max_c = max(b["candidate_poi_ids"].numel() for b in batch)
    bsz = len(batch)
    surd_dim = batch[0]["candidate_surd"].shape[-1]
    meta_dim = batch[0]["candidate_metadata"].shape[-1]

    cand_ids = torch.zeros(bsz, max_c, dtype=torch.long)
    cand_meta = torch.zeros(bsz, max_c, meta_dim, dtype=torch.float32)
    cand_surd = torch.zeros(bsz, max_c, surd_dim, dtype=torch.float32)
    cand_valid = torch.zeros(bsz, max_c, dtype=torch.float32)
    cand_mask = torch.zeros(bsz, max_c, dtype=torch.float32)
    for i, b in enumerate(batch):
        c = b["candidate_poi_ids"].numel()
        cand_ids[i, :c] = b["candidate_poi_ids"]
        cand_meta[i, :c] = b["candidate_metadata"]
        cand_surd[i, :c] = b["candidate_surd"]
        cand_valid[i, :c] = b["candidate_surd_valid"]
        cand_mask[i, :c] = 1.0
    out["candidate_poi_ids"] = cand_ids
    out["candidate_metadata"] = cand_meta
    out["candidate_surd"] = cand_surd
    out["candidate_surd_valid"] = cand_valid
    out["candidate_mask"] = cand_mask
    return out
