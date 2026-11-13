"""End-to-end helpers: prepare examples, SURD profiles, dataloaders, model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pc_smir.data.dataset import (
    NextPOIDataset,
    build_examples,
    collate_variable_candidates,
    train_seen_user_poi_sets,
)
from pc_smir.data.splits import load_official_splits, split_manifest
from pc_smir.data.vocab import VocabMaps, fit_vocab
from pc_smir.features.sources import SourceDiscretizer
from pc_smir.models.pcsmir import PCSMIR
from pc_smir.surd.profiles import ProfileArtifacts, crossfit_profiles


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _artifact_paths(art_dir: Path) -> dict[str, Path]:
    return {
        "vocab": art_dir / "vocab.json",
        "manifest": art_dir / "split_manifest.json",
        "examples": art_dir / "examples.pkl",
        "discretizer": art_dir / "discretizer.pkl",
        "surd_full": art_dir / "surd_profiles_full.json",
        "surd_folds": art_dir / "surd_fold_profiles.pkl",
        "meta": art_dir / "prep_meta.json",
    }


def _cached_sthgcn_protocol(meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        return bool(json.load(f).get("sthgcn_protocol", False))


def _can_reuse_artifacts(paths: dict[str, Path], force: bool, sthgcn_protocol: bool) -> bool:
    """Reuse cache when examples+SURD exist, force is off, and protocol matches meta."""
    if force or not paths["examples"].exists() or not paths["surd_full"].exists():
        return False
    return _cached_sthgcn_protocol(paths["meta"]) == sthgcn_protocol


def _with_vocab(ex: dict, vocab: VocabMaps) -> dict:
    out = dict(ex)
    out["vocab"] = vocab
    return out


def _strip_vocab_refs(examples: dict[str, list[dict]]) -> None:
    for split_ex in examples.values():
        for ex in split_ex:
            ex.pop("vocab", None)


def _states_for_examples(
    ex_list: list[dict],
    vocab: VocabMaps,
    disc: SourceDiscretizer,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Map examples → (target_poi_ids, {Q1..Q4: state_ids}, query_timestamps)."""
    targets: list[int] = []
    mats: dict[str, list[int]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    times: list[float] = []
    for ex in ex_list:
        st = disc.transform_example(_with_vocab(ex, vocab))
        for k in mats:
            mats[k].append(st[k])
        targets.append(ex["target_poi_id"])
        times.append(ex["query_time"].timestamp())
    return (
        np.asarray(targets, dtype=np.int64),
        {k: np.asarray(v, dtype=np.int64) for k, v in mats.items()},
        np.asarray(times, dtype=np.float64),
    )


def _fit_discretizer(
    train_examples: list[dict],
    vocab: VocabMaps,
    surd_cfg: dict[str, Any],
) -> SourceDiscretizer:
    train_for_disc = [_with_vocab(ex, vocab) for ex in train_examples]
    return SourceDiscretizer(
        n_clusters=int(surd_cfg.get("n_clusters", 8)),
        hour_bins=int(surd_cfg.get("hour_bins", 8)),
    ).fit(train_for_disc)


def _build_split_examples(
    splits: dict[str, Any],
    vocab: VocabMaps,
    max_len: int,
    sthgcn_protocol: bool,
) -> dict[str, list[dict]]:
    """Build train/val/test examples; val/test honor ``sthgcn_protocol`` when set."""
    train_users, train_venues = train_seen_user_poi_sets(splits["train"])
    all_df = (
        pd.concat([splits["train"], splits["validation"], splits["test"]], ignore_index=True)
        if sthgcn_protocol
        else None
    )
    eval_kwargs = {
        "sthgcn_protocol": sthgcn_protocol,
        "all_df": all_df,
        "train_user_ids": train_users if sthgcn_protocol else None,
        "train_venue_ids": train_venues if sthgcn_protocol else None,
    }
    return {
        "train": build_examples(splits["train"], vocab, max_len),
        "validation": build_examples(splits["validation"], vocab, max_len, **eval_kwargs),
        "test": build_examples(splits["test"], vocab, max_len, **eval_kwargs),
    }


def _attach_train_fold_ids(train_examples: list[dict], fold_id: np.ndarray) -> None:
    for i, ex in enumerate(train_examples):
        ex["fold_id"] = int(fold_id[i])


def _write_prep_meta(
    path: Path,
    *,
    city: str,
    examples: dict[str, list[dict]],
    vocab: VocabMaps,
    full: ProfileArtifacts,
    sthgcn_protocol: bool,
) -> None:
    meta = {
        "city": city,
        "n_train_examples": len(examples["train"]),
        "n_val_examples": len(examples["validation"]),
        "n_test_examples": len(examples["test"]),
        "n_pois": vocab.n_pois,
        "n_cats": vocab.n_cats,
        "surd_dim": full.feature_dim,
        "estimator": full.estimator,
        "prediction_timestamp": "last_observed_checkin_in_trail_prefix",
        "candidate_policy": "training_catalog_pois",
        "split_protocol": "massive_steps_official_trail_split",
        "sthgcn_protocol": sthgcn_protocol,
        "eval_protocol": (
            "sthgcn_aligned_remove_unseen_user_poi_temporal_visibility"
            if sthgcn_protocol
            else "last_of_trail_full_train_catalog_oov_excluded"
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Prepare pipeline
# ---------------------------------------------------------------------------


def prepare_artifacts(cfg: dict[str, Any], force: bool = False) -> dict[str, Path]:
    """Offline data prep: splits → vocab/catalog → examples → SURD profiles.

    Idempotent when artifacts exist, ``force=False``, and ``eval.sthgcn_protocol``
    matches the cached ``prep_meta.json``.
    """
    city = cfg["data"]["city"]
    data_dir = Path(cfg["data"]["data_dir"])
    art_dir = Path(cfg["output"]["artifact_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(art_dir)
    sthgcn_protocol = bool(cfg.get("eval", {}).get("sthgcn_protocol", False))

    if _can_reuse_artifacts(paths, force, sthgcn_protocol):
        return paths

    # Stage 1 — official splits + audit manifest
    splits = load_official_splits(city, data_dir, download=bool(cfg["data"].get("download", True)))
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(split_manifest(splits), f, indent=2)

    # Stage 2 — train-only vocab + next-POI examples
    max_len = int(cfg["data"].get("max_len", 20))
    vocab = fit_vocab(splits["train"], spatial_cell_size_m=float(cfg["data"].get("spatial_cell_size_m", 500)))
    vocab.save(paths["vocab"])
    examples = _build_split_examples(splits, vocab, max_len, sthgcn_protocol)
    _strip_vocab_refs(examples)

    # Stage 3 — discrete Q1–Q4 states on train rows only
    surd_cfg = dict(cfg.get("surd", {}))
    disc = _fit_discretizer(examples["train"], vocab, surd_cfg)
    train_targets, train_states, train_times = _states_for_examples(examples["train"], vocab, disc)
    catalog = sorted({p for p in vocab.id2poi if p >= 2})

    # Stage 4 — offline SURD profiles (full + cross-fit folds)
    full, folds, fold_id = crossfit_profiles(
        train_targets,
        train_states,
        catalog,
        n_folds=int(surd_cfg.get("n_folds", 5)),
        config=surd_cfg,
        timestamps=train_times if surd_cfg.get("temporal_crossfit", True) else None,
    )
    full.save(paths["surd_full"])
    with open(paths["surd_folds"], "wb") as f:
        pickle.dump({"folds": folds, "fold_id": fold_id}, f)
    with open(paths["discretizer"], "wb") as f:
        pickle.dump(disc, f)

    # Stage 5 — annotate train rows + persist
    _attach_train_fold_ids(examples["train"], fold_id)
    with open(paths["examples"], "wb") as f:
        pickle.dump({"examples": examples, "catalog": catalog, "surd_dim": full.feature_dim}, f)
    _write_prep_meta(
        paths["meta"],
        city=city,
        examples=examples,
        vocab=vocab,
        full=full,
        sthgcn_protocol=sthgcn_protocol,
    )
    return paths


# ---------------------------------------------------------------------------
# Dataloaders + model
# ---------------------------------------------------------------------------


def _mean_fold_surd_tables(
    catalog: list[int],
    folds: list[ProfileArtifacts],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Conservative train lookup: mean of fold profiles that contain each POI."""
    train_feats: dict[int, np.ndarray] = {}
    train_valid: dict[int, float] = {}
    for pid in catalog:
        mats = [fp.poi_features[pid] for fp in folds if pid in fp.poi_features]
        vals = [fp.poi_valid.get(pid, 0.0) for fp in folds if pid in fp.poi_features]
        if not mats:
            continue
        train_feats[pid] = np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)
        train_valid[pid] = float(np.mean(vals))
    return train_feats, train_valid


def _make_dataset(
    examples: list[dict],
    vocab: VocabMaps,
    max_len: int,
    catalog: list[int],
    surd_feats: dict[int, np.ndarray],
    surd_valid: dict[int, float],
    surd_dim: int,
    *,
    neg_samples: int | None,
    seed: int,
) -> NextPOIDataset:
    return NextPOIDataset(
        examples,
        vocab,
        max_len,
        catalog,
        surd_table=surd_feats,
        surd_valid=surd_valid,
        surd_dim=surd_dim,
        neg_samples=neg_samples,
        seed=seed,
    )


def _make_loader(dataset: NextPOIDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_candidates,
        num_workers=0,
    )


def build_dataloaders(cfg: dict[str, Any], paths: dict[str, Path]):
    """Load prepared artifacts → train/val/test DataLoaders.

    Train candidate SURD uses the mean of fold profiles; val/test use the full
    train-estimated profile table.
    """
    vocab = VocabMaps.load(paths["vocab"])
    with open(paths["examples"], "rb") as f:
        blob = pickle.load(f)
    examples = blob["examples"]
    catalog = blob["catalog"]

    surd_full = ProfileArtifacts.load(paths["surd_full"])
    with open(paths["surd_folds"], "rb") as f:
        folds: list[ProfileArtifacts] = pickle.load(f)["folds"]

    train_feats, train_valid = _mean_fold_surd_tables(catalog, folds)
    full_feats, full_valid = surd_full.poi_features, surd_full.poi_valid
    surd_dim = surd_full.feature_dim

    max_len = int(cfg["data"].get("max_len", 20))
    neg = cfg["data"].get("train_neg_samples", None)
    train_bs = int(cfg["train"].get("batch_size", 16))
    eval_bs = int(cfg["train"].get("eval_batch_size", 8))
    seed = int(cfg.get("seed", 0))

    train_ds = _make_dataset(
        examples["train"], vocab, max_len, catalog, train_feats, train_valid, surd_dim,
        neg_samples=neg, seed=seed,
    )
    val_ds = _make_dataset(
        examples["validation"], vocab, max_len, catalog, full_feats, full_valid, surd_dim,
        neg_samples=None, seed=0,
    )
    test_ds = _make_dataset(
        examples["test"], vocab, max_len, catalog, full_feats, full_valid, surd_dim,
        neg_samples=None, seed=0,
    )

    return (
        _make_loader(train_ds, train_bs, shuffle=True),
        _make_loader(val_ds, eval_bs, shuffle=False),
        _make_loader(test_ds, eval_bs, shuffle=False),
        vocab,
        surd_dim,
        catalog,
    )


def build_model(cfg: dict[str, Any], vocab: VocabMaps, surd_dim: int) -> PCSMIR:
    mcfg = cfg.get("model", {})
    pairs = [tuple(p) for p in mcfg.get("pairs", [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])]
    triples = [tuple(t) for t in mcfg.get("triples", [[0, 1, 2], [0, 2, 3]])]
    return PCSMIR(
        n_pois=vocab.n_pois,
        n_cats=vocab.n_cats,
        d=int(mcfg.get("d", 64)),
        meta_dim=int(mcfg.get("meta_dim", 5)),
        surd_dim=surd_dim,
        stable_dim=int(mcfg.get("stable_dim", 8)),
        dropout=float(mcfg.get("dropout", 0.1)),
        pairs=pairs,
        triples=triples,
        use_bilinear=bool(mcfg.get("use_bilinear", True)),
        router_temperature=float(mcfg.get("router_temperature", 1.0)),
        ablation=str(mcfg.get("ablation", "full")),
    )
