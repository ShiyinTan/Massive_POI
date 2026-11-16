"""Trail windows, vocabularies, and padded batches.

History for a prediction at position ``t`` is the trail prefix ``[:t]``.
The target check-in is never part of the sequence. Each trail is its own
sequence (Massive-STEPS trips are not concatenated). Fold ids are a stable
hash of ``trail_id``, so every window of a trip stays in one cross-fit fold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class Example:
    user: int
    poi: list[int]
    hour: list[int]
    dow: list[int]
    lat: list[float]
    lon: list[float]
    target: int
    fold: int
    trail_id: str


@dataclass
class Vocab:
    user_to_idx: dict[str, int]
    poi_to_idx: dict[str, int]
    cat_to_idx: dict[str, int]
    poi_lat: np.ndarray
    poi_lon: np.ndarray
    poi_cat: np.ndarray
    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float
    n_users: int
    n_pois: int
    n_cats: int

    def state_dict(self) -> dict:
        return {
            "user_to_idx": self.user_to_idx,
            "poi_to_idx": self.poi_to_idx,
            "cat_to_idx": self.cat_to_idx,
            "poi_lat": self.poi_lat.tolist(),
            "poi_lon": self.poi_lon.tolist(),
            "poi_cat": self.poi_cat.tolist(),
            "lat_mean": self.lat_mean,
            "lat_std": self.lat_std,
            "lon_mean": self.lon_mean,
            "lon_std": self.lon_std,
            "n_users": self.n_users,
            "n_pois": self.n_pois,
            "n_cats": self.n_cats,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "Vocab":
        return cls(
            user_to_idx={str(k): int(v) for k, v in payload["user_to_idx"].items()},
            poi_to_idx={str(k): int(v) for k, v in payload["poi_to_idx"].items()},
            cat_to_idx={str(k): int(v) for k, v in payload["cat_to_idx"].items()},
            poi_lat=np.asarray(payload["poi_lat"], dtype=np.float64),
            poi_lon=np.asarray(payload["poi_lon"], dtype=np.float64),
            poi_cat=np.asarray(payload["poi_cat"], dtype=np.int64),
            lat_mean=float(payload["lat_mean"]),
            lat_std=float(payload["lat_std"]),
            lon_mean=float(payload["lon_mean"]),
            lon_std=float(payload["lon_std"]),
            n_users=int(payload["n_users"]),
            n_pois=int(payload["n_pois"]),
            n_cats=int(payload["n_cats"]),
        )


def _day_of_week(stamps: pd.Series) -> pd.Series:
    accessor = stamps.dt
    dow = accessor.day_of_week if hasattr(accessor, "day_of_week") else accessor.dayofweek
    return dow.fillna(0).astype(int).clip(0, 6)


def _fold_id(trail_id: str, n_folds: int, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{trail_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def _std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std if std > 1e-6 else 1.0


def build_vocab(train_df: pd.DataFrame) -> Vocab:
    """Fit ids and coordinate normalization on the training split only."""
    users = sorted(train_df["user_id"].astype(str).unique())
    venues = sorted(train_df["venue_id"].astype(str).unique())
    cats = sorted(train_df["venue_category"].astype(str).unique())
    user_to_idx = {user: i for i, user in enumerate(users)}
    poi_to_idx = {venue: i + 1 for i, venue in enumerate(venues)}  # 0 = pad
    cat_to_idx = {cat: i + 1 for i, cat in enumerate(cats)}  # 0 = unknown

    lat = train_df["latitude"].to_numpy(dtype=np.float64)
    lon = train_df["longitude"].to_numpy(dtype=np.float64)
    lat_mean, lon_mean = float(lat.mean()), float(lon.mean())
    lat_std, lon_std = _std(lat), _std(lon)

    n_pois = len(venues) + 1
    poi_lat = np.zeros(n_pois, dtype=np.float64)
    poi_lon = np.zeros(n_pois, dtype=np.float64)
    poi_cat = np.zeros(n_pois, dtype=np.int64)
    grouped = train_df.groupby(train_df["venue_id"].astype(str))
    for venue, part in grouped:
        idx = poi_to_idx[str(venue)]
        poi_lat[idx] = (float(part["latitude"].mean()) - lat_mean) / lat_std
        poi_lon[idx] = (float(part["longitude"].mean()) - lon_mean) / lon_std
        poi_cat[idx] = cat_to_idx.get(str(part["venue_category"].iloc[0]), 0)

    return Vocab(
        user_to_idx=user_to_idx,
        poi_to_idx=poi_to_idx,
        cat_to_idx=cat_to_idx,
        poi_lat=poi_lat,
        poi_lon=poi_lon,
        poi_cat=poi_cat,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std,
        n_users=len(users),
        n_pois=n_pois,
        n_cats=len(cats) + 1,
    )


def build_examples(
    frame: pd.DataFrame,
    vocab: Vocab,
    *,
    max_history: int,
    min_history: int,
    n_folds: int,
    seed: int,
    assign_folds: bool,
) -> list[Example]:
    """One next-POI example per check-in after ``min_history`` prefix steps.

    Targets whose venue is outside the training vocabulary are dropped.
    Unknown history venues become the padding id and are masked only when
    the whole step is padding; an unknown venue id 0 is a real token here
    and is a limitation of train-only ids (rare on the official splits).
    """
    examples: list[Example] = []
    work = frame
    if "hour" not in work.columns:
        work = work.copy()
        stamps = pd.to_datetime(work["timestamp"])
        work["hour"] = stamps.dt.hour.fillna(0).astype(int).clip(0, 23)
        work["dow"] = _day_of_week(stamps)

    for trail_id, part in work.groupby("trail_id", sort=False):
        part = part.sort_values("timestamp")
        users = part["user_id"].astype(str).tolist()
        venues = part["venue_id"].astype(str).tolist()
        hours = part["hour"].tolist()
        dows = part["dow"].tolist()
        lats = part["latitude"].tolist()
        lons = part["longitude"].tolist()
        fold = _fold_id(str(trail_id), n_folds, seed) if assign_folds else -1
        for t in range(min_history, len(part)):
            target_venue = venues[t]
            if target_venue not in vocab.poi_to_idx:
                continue
            user_key = users[t - 1]
            if user_key not in vocab.user_to_idx:
                continue
            start = max(0, t - max_history)
            examples.append(
                Example(
                    user=vocab.user_to_idx[user_key],
                    poi=[vocab.poi_to_idx.get(v, 0) for v in venues[start:t]],
                    hour=[int(h) for h in hours[start:t]],
                    dow=[int(d) for d in dows[start:t]],
                    lat=[float(v) for v in lats[start:t]],
                    lon=[float(v) for v in lons[start:t]],
                    target=vocab.poi_to_idx[target_venue],
                    fold=fold,
                    trail_id=str(trail_id),
                )
            )
    return examples


def frames_to_examples(
    frames: dict[str, pd.DataFrame],
    *,
    max_history: int = 50,
    min_history: int = 1,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[Vocab, dict[str, list[Example]]]:
    prepared = {name: _prepare_frame(frame) for name, frame in frames.items()}
    vocab = build_vocab(prepared["train"])
    splits = {}
    for name, frame in prepared.items():
        splits[name] = build_examples(
            frame,
            vocab,
            max_history=max_history,
            min_history=min_history,
            n_folds=n_folds,
            seed=seed,
            assign_folds=(name == "train"),
        )
    return vocab, splits


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "venue_category" not in work.columns:
        work["venue_category"] = "UNK"
    stamps = pd.to_datetime(work["timestamp"])
    work["timestamp"] = stamps
    work["hour"] = stamps.dt.hour.fillna(0).astype(int).clip(0, 23)
    work["dow"] = _day_of_week(stamps)
    return work


def collate_examples(
    examples: Sequence[Example],
    *,
    max_len: int,
    lat_mean: float,
    lat_std: float,
    lon_mean: float,
    lon_std: float,
) -> dict[str, torch.Tensor]:
    """Right-align histories so position 0 is the oldest kept check-in."""
    batch = len(examples)
    poi = torch.zeros(batch, max_len, dtype=torch.long)
    hour = torch.zeros(batch, max_len, dtype=torch.long)
    dow = torch.zeros(batch, max_len, dtype=torch.long)
    loc = torch.zeros(batch, max_len, 2, dtype=torch.float32)
    pad = torch.ones(batch, max_len, dtype=torch.bool)
    user = torch.zeros(batch, dtype=torch.long)
    target = torch.zeros(batch, dtype=torch.long)
    fold = torch.full((batch,), -1, dtype=torch.long)
    for i, example in enumerate(examples):
        length = min(len(example.poi), max_len)
        start = max_len - length
        poi[i, start:] = torch.tensor(example.poi[-length:], dtype=torch.long)
        hour[i, start:] = torch.tensor(example.hour[-length:], dtype=torch.long)
        dow[i, start:] = torch.tensor(example.dow[-length:], dtype=torch.long)
        lat = (np.asarray(example.lat[-length:], dtype=np.float64) - lat_mean) / lat_std
        lon = (np.asarray(example.lon[-length:], dtype=np.float64) - lon_mean) / lon_std
        loc[i, start:, 0] = torch.tensor(lat, dtype=torch.float32)
        loc[i, start:, 1] = torch.tensor(lon, dtype=torch.float32)
        pad[i, start:] = False
        user[i] = example.user
        target[i] = example.target
        fold[i] = example.fold
    return {
        "user": user,
        "poi": poi,
        "hour": hour,
        "dow": dow,
        "loc": loc,
        "pad": pad,
        "target": target,
        "fold": fold,
    }


def make_collate(vocab: Vocab, max_len: int):
    return partial(
        collate_examples,
        max_len=max_len,
        lat_mean=vocab.lat_mean,
        lat_std=vocab.lat_std,
        lon_mean=vocab.lon_mean,
        lon_std=vocab.lon_std,
    )
