"""Train-only vocabularies and preprocessing statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAD_ID = 0
OOV_ID = 1


@dataclass
class VocabMaps:
    poi2id: dict[str, int] = field(default_factory=dict)
    id2poi: dict[int, str] = field(default_factory=dict)
    cat2id: dict[str, int] = field(default_factory=dict)
    id2cat: dict[int, str] = field(default_factory=dict)
    poi_category: dict[int, int] = field(default_factory=dict)  # poi_id -> cat_id
    poi_coords: dict[int, tuple[float, float]] = field(default_factory=dict)
    train_poi_counts: dict[int, int] = field(default_factory=dict)
    lat_mean: float = 0.0
    lat_std: float = 1.0
    lon_mean: float = 0.0
    lon_std: float = 1.0
    spatial_cell_size_m: float = 500.0
    # Spatial origin for cell hashing (train mins).
    lat_origin: float = 0.0
    lon_origin: float = 0.0
    n_pois: int = 2
    n_cats: int = 2

    def encode_poi(self, venue_id: Any) -> int:
        key = str(venue_id)
        return self.poi2id.get(key, OOV_ID)

    def encode_cat(self, category: Any) -> int:
        key = str(category) if pd.notna(category) else "<missing>"
        return self.cat2id.get(key, OOV_ID)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        # JSON keys must be strings.
        payload["id2poi"] = {str(k): v for k, v in self.id2poi.items()}
        payload["id2cat"] = {str(k): v for k, v in self.id2cat.items()}
        payload["poi_category"] = {str(k): v for k, v in self.poi_category.items()}
        payload["poi_coords"] = {str(k): list(v) for k, v in self.poi_coords.items()}
        payload["train_poi_counts"] = {str(k): v for k, v in self.train_poi_counts.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "VocabMaps":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls()
        obj.poi2id = payload["poi2id"]
        obj.id2poi = {int(k): v for k, v in payload["id2poi"].items()}
        obj.cat2id = payload["cat2id"]
        obj.id2cat = {int(k): v for k, v in payload["id2cat"].items()}
        obj.poi_category = {int(k): int(v) for k, v in payload["poi_category"].items()}
        obj.poi_coords = {int(k): (float(v[0]), float(v[1])) for k, v in payload["poi_coords"].items()}
        obj.train_poi_counts = {int(k): int(v) for k, v in payload["train_poi_counts"].items()}
        for key in [
            "lat_mean",
            "lat_std",
            "lon_mean",
            "lon_std",
            "spatial_cell_size_m",
            "lat_origin",
            "lon_origin",
            "n_pois",
            "n_cats",
        ]:
            setattr(obj, key, payload[key])
        return obj


def fit_vocab(train_df: pd.DataFrame, spatial_cell_size_m: float = 500.0) -> VocabMaps:
    """Fit ID maps and normalization stats on training split only."""
    vocab = VocabMaps(spatial_cell_size_m=spatial_cell_size_m)
    pois = sorted(train_df["venue_id"].astype(str).unique().tolist())
    cats = sorted(train_df["venue_category"].fillna("<missing>").astype(str).unique().tolist())

    vocab.poi2id = {"<pad>": PAD_ID, "<oov>": OOV_ID}
    for i, p in enumerate(pois, start=2):
        vocab.poi2id[p] = i
    vocab.id2poi = {i: p for p, i in vocab.poi2id.items()}

    vocab.cat2id = {"<pad>": PAD_ID, "<oov>": OOV_ID}
    for i, c in enumerate(cats, start=2):
        vocab.cat2id[c] = i
    vocab.id2cat = {i: c for c, i in vocab.cat2id.items()}

    # POI metadata from training rows (last occurrence wins for coords/category).
    for _, row in train_df.iterrows():
        pid = vocab.encode_poi(row["venue_id"])
        cid = vocab.encode_cat(row["venue_category"])
        vocab.poi_category[pid] = cid
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            vocab.poi_coords[pid] = (float(row["latitude"]), float(row["longitude"]))

    counts = train_df["venue_id"].astype(str).value_counts()
    vocab.train_poi_counts = {vocab.encode_poi(k): int(v) for k, v in counts.items()}

    lat = train_df["latitude"].astype(float)
    lon = train_df["longitude"].astype(float)
    vocab.lat_mean = float(lat.mean())
    vocab.lat_std = float(lat.std() or 1.0)
    vocab.lon_mean = float(lon.mean())
    vocab.lon_std = float(lon.std() or 1.0)
    vocab.lat_origin = float(lat.min())
    vocab.lon_origin = float(lon.min())
    vocab.n_pois = len(vocab.poi2id)
    vocab.n_cats = len(vocab.cat2id)
    return vocab


def spatial_cell_id(lat: float, lon: float, vocab: VocabMaps) -> int:
    """Deterministic spatial cell from train-fitted origin and resolution."""
    # Approximate meters: 1 deg lat ~ 111_320 m; lon scaled by cos(lat).
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(0.2, float(np.cos(np.deg2rad(lat))))
    dy = (lat - vocab.lat_origin) * meters_per_deg_lat
    dx = (lon - vocab.lon_origin) * meters_per_deg_lon
    cell_y = int(np.floor(dy / vocab.spatial_cell_size_m))
    cell_x = int(np.floor(dx / vocab.spatial_cell_size_m))
    # Map to non-negative int; reserve 0 for pad/missing.
    return abs(cell_y) * 100_003 + abs(cell_x) + 2


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))