"""Leakage-safe next-POI example construction from official trail splits.

从官方 trail 划分构造无泄漏的 next-POI 样本。

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

from pc_smir.data.vocab import VocabMaps, haversine_m
from pc_smir.features.sources import build_prefix_example
from pc_smir.features.temporal import query_time_features


def trails_from_split(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group check-in rows into trails with ≥2 events.

    作用 / Role:
      按 (user_id, trail_id) 分组、按时间排序；长度 <2 的轨迹丢弃
      （无法形成 history+target）。

    输入: 官方 split CSV 读入的 DataFrame（如 beijing train 1044 行）。
    输出: list[{"user_id", "trail_id", "rows": [row_dict, ...]}]
    北京例: train → 约 400 条 trail。

    何时调用: 仅 prepare 阶段，经 ``build_examples`` 间接调用。
    """
    trails = []
    for (user_id, trail_id), group in df.groupby(["user_id", "trail_id"], sort=False):
        group = group.sort_values(["timestamp", "venue_id"])
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        trails.append({"user_id": int(user_id), "trail_id": str(trail_id), "rows": rows})
    return trails


def build_examples(df: pd.DataFrame, vocab: VocabMaps, max_len: int) -> list[dict]:
    """Turn each trail into one next-POI dict example (Python, not tensors yet).

    作用 / Role:
      history = trail 除最后一次外的签到；target = 最后一次 venue_id。
      经 ``build_prefix_example`` 生成泄漏安全特征（预测时刻=历史最后一次时间）。

    输出样例字段（北京某 trail，hist_len=5；仍是 Python list/ndarray）:
      history_poi_ids / categories / hours / dows / time_gaps / cells / coordinates
      query_time, stable_user_features[8], source_available[4]=1,
      target_poi_id (模型 int), trail_id, user_id, last_lat/lon, ...

    北京规模: train≈400, validation≈58, test≈115 examples（max_len=20）。

    何时调用: prepare_artifacts 对 train/val/test 各调用一次并写入 examples.pkl。
    train/eval 不再重建，只反序列化。
    """
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
    """Pad one example into model tensors + build candidate set.

    作用 / Role:
      把 prepare 阶段的 dict example 变成单条样本张量；右侧对齐填充到 max_len；
      构造候选 POI（训练可采样负例；评估用全量 train catalog）。

    单条 ``__getitem__`` 输出形状（北京 smoke: max_len=L=20, neg_samples=32 → C=33）:
      history_poi_ids       [L]      int64   # 左 pad=0，真实历史靠右
      history_categories    [L]
      history_hours/dows    [L]
      history_time_gaps     [L]      float32
      history_cells         [L]      # % 5000 以适配 embedding 表
      history_coordinates   [L, 2]   # 已用 train mean/std 标准化
      history_mask          [L]      # 1=真实事件, 0=pad
      query_time_features   [7]
      stable_user_features  [8]
      source_available      [4]      # Q1–Q4；北京样本通常全 1
      candidate_poi_ids     [C]
      candidate_metadata    [C, 5]   # log1p(dist), log1p(pop), cat_norm, lat_n, lon_n
      candidate_surd        [C, F_s]
      candidate_surd_valid  [C]
      target_index          []       # target 在候选轴上的下标（采样时多为 0）
      target_poi_id         []

    北京 catalog≈498；eval 时 C≈498（全量），train 采样时 C=1+neg_samples。

    何时调用: build_dataloaders（train/eval）里构造 Dataset；每个 batch 由 DataLoader
    反复调用 ``__getitem__``。prepare 阶段不实例化本类。
    """

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
        """Build candidate id list for one example.

        neg_samples is None → 全量 train catalog（北京≈498，用于 val/test）。
        neg_samples=K → [target] + K 个负例（训练 sampled softmax）。
        """
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
        """Materialize one training/eval sample as tensors (see class docstring)."""
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
    """Pad candidate axis to max-C in the batch / 把候选维 pad 到 batch 内最大 C。

    作用 / Role:
      历史等固定键直接 stack → [B, ...]。
      候选相关键 pad 到 max_c，并增加 candidate_mask[B, C]（1=真实候选）。

    北京例:
      train batch_size=4, neg_samples=32 → 通常 C=33，batch 形状 [4, 33, ...]
      eval 全量 catalog → C≈498，batch 形状 [B, 498, ...]

    何时调用: DataLoader 的 collate_fn（仅 train/eval）；prepare 不用。
    """
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
