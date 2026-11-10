"""Leakage-safe next-POI example construction from official trail splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pc_smir.data.vocab import VocabMaps, haversine_m
from pc_smir.features.sources import build_prefix_example
from pc_smir.features.temporal import query_time_features


def trails_from_split(df: pd.DataFrame) -> list[dict[str, Any]]:
    trails = []
    for (user_id, trail_id), group in df.groupby(["user_id", "trail_id"], sort=False):
        group = group.sort_values(["timestamp", "venue_id"])
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        trails.append({"user_id": int(user_id), "trail_id": str(trail_id), "rows": rows})
    return trails


def build_examples(df: pd.DataFrame, vocab: VocabMaps, max_len: int) -> list[dict]:
    examples = []
    for trail in trails_from_split(df):
        rows = trail["rows"]
        hist = rows[:-1]
        target = rows[-1]["venue_id"]
        ex = build_prefix_example(
            user_id=trail["user_id"],
            trail_id=trail["trail_id"],
            hist_rows=hist,
            target_venue_id=int(target),
            vocab=vocab,
            max_len=max_len,
        )
        if ex is not None:
            examples.append(ex)
    return examples


class NextPOIDataset(Dataset):
    """Dataset returning padded tensors + candidate slots filled by collate/sampler."""

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
        self.examples = []
        for ex in examples:
            if exclude_oov_targets and ex["target_poi_id"] == 1:
                continue
            if ex["target_poi_id"] not in self.catalog and exclude_oov_targets:
                continue
            self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def _candidate_set(self, target_id: int) -> list[int]:
        if self.neg_samples is None or self.neg_samples <= 0:
            return list(self.catalog)
        # Sampled training candidates: target + negatives from train catalog.
        negs = []
        catalog = self.catalog
        while len(negs) < self.neg_samples:
            p = int(self.rng.choice(catalog))
            if p != target_id:
                negs.append(p)
        return [target_id] + negs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        L = self.max_len
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
        for i in range(hist_len):
            j = start + i
            src = len(ex["history_poi_ids"]) - hist_len + i
            poi[j] = ex["history_poi_ids"][src]
            cat[j] = ex["history_categories"][src]
            hours[j] = ex["history_hours"][src]
            dows[j] = ex["history_dows"][src]
            gaps[j] = ex["history_time_gaps"][src]
            cells[j] = ex["history_cells"][src] % 5000  # bounded embedding table
            la, lo = ex["history_coordinates"][src]
            coords[j, 0] = (la - self.vocab.lat_mean) / self.vocab.lat_std
            coords[j, 1] = (lo - self.vocab.lon_mean) / self.vocab.lon_std
            mask[j] = 1.0

        candidates = self._candidate_set(ex["target_poi_id"])
        # Ensure target present.
        if ex["target_poi_id"] not in candidates:
            candidates[0] = ex["target_poi_id"]
        target_index = candidates.index(ex["target_poi_id"])

        cand_meta = []
        cand_surd = []
        cand_valid = []
        for pid in candidates:
            lat, lon = self.vocab.poi_coords.get(pid, (ex["last_lat"], ex["last_lon"]))
            dist = haversine_m(ex["last_lat"], ex["last_lon"], lat, lon)
            pop = float(self.vocab.train_poi_counts.get(pid, 0))
            cat_id = float(self.vocab.poi_category.get(pid, 1))
            cand_meta.append(
                [
                    np.log1p(dist),
                    np.log1p(pop),
                    cat_id / max(self.vocab.n_cats, 1),
                    (lat - self.vocab.lat_mean) / self.vocab.lat_std,
                    (lon - self.vocab.lon_mean) / self.vocab.lon_std,
                ]
            )
            feat = self.surd_table.get(pid)
            if feat is None:
                cand_surd.append(np.zeros(self.surd_dim, dtype=np.float32))
                cand_valid.append(0.0)
            else:
                cand_surd.append(np.asarray(feat, dtype=np.float32))
                cand_valid.append(float(self.surd_valid.get(pid, 1.0)))

        return {
            "history_poi_ids": torch.tensor(poi),
            "history_categories": torch.tensor(cat),
            "history_hours": torch.tensor(hours),
            "history_dows": torch.tensor(dows),
            "history_time_gaps": torch.tensor(gaps),
            "history_cells": torch.tensor(cells),
            "history_coordinates": torch.tensor(coords),
            "history_mask": torch.tensor(mask),
            "query_time_features": torch.tensor(query_time_features(ex["query_time"])),
            "stable_user_features": torch.tensor(ex["stable_user_features"]),
            "source_available": torch.tensor(ex["source_available"]),
            "candidate_poi_ids": torch.tensor(np.asarray(candidates, dtype=np.int64)),
            "candidate_metadata": torch.tensor(np.asarray(cand_meta, dtype=np.float32)),
            "candidate_surd": torch.tensor(np.asarray(cand_surd, dtype=np.float32)),
            "candidate_surd_valid": torch.tensor(np.asarray(cand_valid, dtype=np.float32)),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "target_poi_id": torch.tensor(ex["target_poi_id"], dtype=torch.long),
        }


def collate_variable_candidates(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad candidate axis to max-C in batch (needed for sampled softmax)."""
    keys_fixed = [
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
    out: dict[str, torch.Tensor] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_fixed}
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