"""Per-POI SURD profile estimation, shrinkage, normalization, cross-fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pc_smir.surd.estimator import ESTIMATOR_NAME, ESTIMATOR_VERSION, estimate_surd

SOURCE_NAMES = ["Q1", "Q2", "Q3", "Q4"]
DEFAULT_PAIRS = [
    ("Q1", "Q2"),
    ("Q1", "Q3"),
    ("Q1", "Q4"),
    ("Q2", "Q3"),
    ("Q2", "Q4"),
    ("Q3", "Q4"),
]
DEFAULT_TRIPLES = [("Q1", "Q2", "Q3"), ("Q1", "Q3", "Q4")]


def _feature_order(pairs=None, triples=None) -> list[str]:
    pairs = pairs or DEFAULT_PAIRS
    triples = triples or DEFAULT_TRIPLES
    keys = [f"unique_{q.lower()}" for q in SOURCE_NAMES]
    keys += [f"synergy_{a.lower()}{b.lower()}" for a, b in pairs]
    keys += [f"synergy_{a.lower()}{b.lower()}{c.lower()}" for a, b, c in triples]
    # Per-atom channels: robust_z, sign, log1p_se, reliability, valid  => handled later
    return keys


@dataclass
class ProfileArtifacts:
    """Versioned SURD profile store keyed by POI id (model index)."""

    poi_features: dict[int, np.ndarray]
    poi_valid: dict[int, float]
    poi_raw: dict[int, dict[str, Any]]
    feature_names: list[str]
    channel_names: list[str]
    norm_median: np.ndarray
    norm_iqr: np.ndarray
    global_prior: np.ndarray
    estimator: str = ESTIMATOR_NAME
    version: str = ESTIMATOR_VERSION
    config: dict | None = None

    @property
    def feature_dim(self) -> int:
        # channels per atom: robust_z, sign, log1p(se), reliability, valid
        return len(self.feature_names) * 5

    def lookup(self, poi_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        valid = []
        for pid in poi_ids:
            if pid in self.poi_features:
                feats.append(self.poi_features[pid])
                valid.append(self.poi_valid.get(pid, 1.0))
            else:
                feats.append(np.zeros(self.feature_dim, dtype=np.float32))
                valid.append(0.0)
        return np.stack(feats, axis=0), np.asarray(valid, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "estimator": self.estimator,
            "version": self.version,
            "feature_names": self.feature_names,
            "channel_names": self.channel_names,
            "norm_median": self.norm_median.tolist(),
            "norm_iqr": self.norm_iqr.tolist(),
            "global_prior": self.global_prior.tolist(),
            "config": self.config,
            "poi_valid": {str(k): float(v) for k, v in self.poi_valid.items()},
            "poi_features": {str(k): v.tolist() for k, v in self.poi_features.items()},
            "poi_raw": {str(k): v for k, v in self.poi_raw.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "ProfileArtifacts":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            poi_features={int(k): np.asarray(v, dtype=np.float32) for k, v in payload["poi_features"].items()},
            poi_valid={int(k): float(v) for k, v in payload["poi_valid"].items()},
            poi_raw={int(k): v for k, v in payload["poi_raw"].items()},
            feature_names=payload["feature_names"],
            channel_names=payload["channel_names"],
            norm_median=np.asarray(payload["norm_median"], dtype=np.float64),
            norm_iqr=np.asarray(payload["norm_iqr"], dtype=np.float64),
            global_prior=np.asarray(payload["global_prior"], dtype=np.float64),
            estimator=payload.get("estimator", ESTIMATOR_NAME),
            version=payload.get("version", ESTIMATOR_VERSION),
            config=payload.get("config"),
        )


def _atom_vector(result: dict, pairs, triples) -> np.ndarray:
    syn = result["synergy"]
    vals = [float(result["unique"][q]) for q in SOURCE_NAMES]
    for a, b in pairs:
        key = (a, b)
        if key in syn:
            vals.append(float(syn[key]))
        else:
            vals.append(float(syn.get(f"{a}_{b}", 0.0)))
    for a, b, c in triples:
        key = (a, b, c)
        if key in syn:
            vals.append(float(syn[key]))
        else:
            vals.append(float(syn.get(f"{a}_{b}_{c}", 0.0)))
    return np.asarray(vals, dtype=np.float64)


def _channels(atom_vals: np.ndarray, se: np.ndarray, reliability: float, valid: float, median, iqr, clip: float = 5.0) -> np.ndarray:
    robust_z = (atom_vals - median) / np.clip(iqr, 1e-6, None)
    robust_z = np.clip(robust_z, -clip, clip)
    sign = np.sign(atom_vals)
    log_se = np.log1p(np.maximum(se, 0.0))
    rel = np.full_like(atom_vals, reliability)
    val = np.full_like(atom_vals, valid)
    return np.stack([robust_z, sign, log_se, rel, val], axis=-1).reshape(-1).astype(np.float32)


def estimate_profiles_for_rows(
    target_poi_ids: np.ndarray,
    source_state_matrix: dict[str, np.ndarray],
    catalog_poi_ids: list[int],
    config: dict | None = None,
) -> ProfileArtifacts:
    cfg = {
        "alpha": 0.5,
        "pairs": DEFAULT_PAIRS,
        "triples": DEFAULT_TRIPLES,
        "min_positive": 2,
        "kappa": 10.0,
        "n_bootstrap": 5,
        "clip": 5.0,
    }
    if config:
        cfg.update(config)
    pairs = [tuple(p) for p in cfg["pairs"]]
    triples = [tuple(t) for t in cfg["triples"]]
    feature_names = _feature_order(pairs, triples)
    channel_names = ["robust_z", "sign", "log1p_se", "reliability", "valid"]

    n = len(target_poi_ids)
    states = {k: np.asarray(v).astype(np.int64) for k, v in source_state_matrix.items()}

    # First pass: raw atoms for POIs with support.
    raw_atoms: dict[int, np.ndarray] = {}
    raw_meta: dict[int, dict] = {}
    ses: dict[int, np.ndarray] = {}

    pos_counts = {int(p): int((target_poi_ids == p).sum()) for p in catalog_poi_ids}
    eligible = [p for p in catalog_poi_ids if pos_counts.get(p, 0) >= int(cfg["min_positive"])]

    estimator_cfg = {
        "alpha": cfg["alpha"],
        "pairs": pairs,
        "triples": triples,
    }

    for p in eligible:
        y = (target_poi_ids == p).astype(np.int64)
        result = estimate_surd(y, states, SOURCE_NAMES, estimator_cfg)
        atoms = _atom_vector(result, pairs, triples)
        # Bootstrap SE (stratified by keeping positives).
        boot = []
        rng = np.random.default_rng(abs(hash((int(p), n))) % (2**32))
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        for b in range(int(cfg["n_bootstrap"])):
            if len(neg_idx) == 0:
                break
            take_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            take_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([take_pos, take_neg])
            yb = y[idx]
            sb = {k: v[idx] for k, v in states.items()}
            rb = estimate_surd(yb, sb, SOURCE_NAMES, estimator_cfg)
            boot.append(_atom_vector(rb, pairs, triples))
        se = np.std(np.stack(boot, axis=0), axis=0) if boot else np.zeros_like(atoms)
        raw_atoms[p] = atoms
        ses[p] = se
        raw_meta[p] = {
            "n_total": int(n),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "atoms": atoms.tolist(),
            "se": se.tolist(),
            "redundancy": {f"{a}_{b}": float(result["redundancy"][(a, b)]) for a, b in pairs},
        }

    if raw_atoms:
        mat = np.stack(list(raw_atoms.values()), axis=0)
        global_prior = np.median(mat, axis=0)
        median = np.median(mat, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
    else:
        global_prior = np.zeros(len(feature_names), dtype=np.float64)
        median = global_prior.copy()
        iqr = np.ones_like(median)

    kappa = float(cfg["kappa"])
    poi_features: dict[int, np.ndarray] = {}
    poi_valid: dict[int, float] = {}
    poi_raw: dict[int, Any] = {}

    for p in catalog_poi_ids:
        n_pos = pos_counts.get(p, 0)
        reliability = n_pos / (n_pos + kappa) if n_pos > 0 else 0.0
        if p in raw_atoms:
            shrunk = reliability * raw_atoms[p] + (1.0 - reliability) * global_prior
            valid = 1.0 if n_pos >= int(cfg["min_positive"]) else 0.0
            se = ses[p]
            feat = _channels(shrunk, se, reliability, valid, median, iqr, clip=float(cfg["clip"]))
            poi_features[p] = feat
            poi_valid[p] = valid
            poi_raw[p] = {
                **raw_meta[p],
                "shrunk": shrunk.tolist(),
                "reliability": reliability,
                "valid": valid,
                "prior_level": "global",
            }
        else:
            feat = _channels(global_prior, np.zeros_like(global_prior), reliability, 0.0, median, iqr, clip=float(cfg["clip"]))
            poi_features[p] = feat
            poi_valid[p] = 0.0
            poi_raw[p] = {
                "n_total": int(n),
                "n_positive": int(n_pos),
                "n_negative": int(n - n_pos),
                "reliability": reliability,
                "valid": 0.0,
                "prior_level": "global_or_unknown",
            }

    return ProfileArtifacts(
        poi_features=poi_features,
        poi_valid=poi_valid,
        poi_raw=poi_raw,
        feature_names=feature_names,
        channel_names=channel_names,
        norm_median=median,
        norm_iqr=iqr,
        global_prior=global_prior,
        config=cfg,
    )


def crossfit_profiles(
    example_poi_targets: np.ndarray,
    source_state_matrix: dict[str, np.ndarray],
    catalog_poi_ids: list[int],
    n_folds: int = 5,
    config: dict | None = None,
    timestamps: np.ndarray | None = None,
) -> tuple[ProfileArtifacts, list[ProfileArtifacts], np.ndarray]:
    """Forward-chaining / block cross-fit profiles for training rows.

    被 ``prepare_artifacts`` 调用。对 catalog 中每个 POI p，在给定行集合上估计
    Y_p 相对 Q1–Q4 的 SURD 原子（经 ``estimate_profiles_for_rows`` → ``estimate_surd``）。

    输入 / Inputs:
      example_poi_targets: [N] 每条 train example 的 target 模型 POI id（北京 N≈400）
      source_state_matrix: {"Q1".."Q4"} → [N] 离散状态
      catalog_poi_ids: 要建 profile 的 POI 列表（北京≈498）
      timestamps: 若给定，按时间切块做前向交叉拟合

    返回 / Returns:
      - full_train_profiles: 用全部 train 估的表 → val/test 候选查找
      - fold_profiles: 长度 n_folds；第 k 折不含（或不晚于）折 k 的标签
      - fold_id per example: 训练时选用哪张折表
    """
    n = len(example_poi_targets)
    if timestamps is not None:
        order = np.argsort(timestamps)
    else:
        order = np.arange(n)
    fold_id = np.full(n, -1, dtype=np.int64)
    # Contiguous time blocks.
    block = int(np.ceil(n / n_folds))
    for i, idx in enumerate(order):
        fold_id[idx] = min(n_folds - 1, i // block)

    fold_profiles: list[ProfileArtifacts] = []
    for k in range(n_folds):
        # Prefer strictly earlier folds when timestamps provided; else leave-one-fold-out.
        if timestamps is not None:
            train_mask = fold_id < k
            if train_mask.sum() < max(20, n // (n_folds * 2)):
                train_mask = fold_id != k
        else:
            train_mask = fold_id != k
        sub_targets = example_poi_targets[train_mask]
        sub_states = {name: arr[train_mask] for name, arr in source_state_matrix.items()}
        fold_profiles.append(
            estimate_profiles_for_rows(sub_targets, sub_states, catalog_poi_ids, config=config)
        )

    full = estimate_profiles_for_rows(example_poi_targets, source_state_matrix, catalog_poi_ids, config=config)
    return full, fold_profiles, fold_id


def source_transform_hash(discretizer_cfg: dict) -> str:
    blob = json.dumps(discretizer_cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]