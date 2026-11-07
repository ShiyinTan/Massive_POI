"""Load Massive-STEPS check-ins and build next-POI samples without future leakage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "trail_id",
    "user_id",
    "venue_id",
    "timestamp",
)


@dataclass
class Vocab:
    """Transductive POI / user vocabulary (ids remapped to contiguous ints)."""

    poi2idx: Dict[int, int]
    idx2poi: List[int]
    user2idx: Dict[Any, int]
    category2idx: Dict[str, int]
    poi_coords: np.ndarray  # [num_pois, 2] lat/lon; NaN if unknown
    poi_category_idx: np.ndarray  # [num_pois]

    @property
    def num_pois(self) -> int:
        return len(self.idx2poi)

    @property
    def num_users(self) -> int:
        return len(self.user2idx)

    @property
    def num_categories(self) -> int:
        return len(self.category2idx)


@dataclass
class NextPOISample:
    sample_id: int
    user_id: int
    trail_id: str
    history_poi_ids: List[int]
    history_timestamps: List[pd.Timestamp]
    history_lats: List[float]
    history_lons: List[float]
    target_poi_id: int
    target_timestamp: pd.Timestamp
    target_time_features: np.ndarray  # [8] hour/dow sin-cos + weekend + month sin-cos
    spatial_features: np.ndarray  # fixed-dim spatial context (no future info)
    user_features: np.ndarray  # optional Q4-ready stats (not used as 4th SURD source in v1)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    # Try relative to package, then repo root (parent of surd_guided_poi).
    here = Path(__file__).resolve().parents[1]
    repo = here.parent
    for base in (here, repo, repo / "Massive-STEPS"):
        cand = base / path
        if cand.is_file():
            return cand
    return p


def load_checkins(
    checkins_csv: Optional[str] = None,
    hf_dataset_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load Massive-STEPS check-in CSV from local path or HuggingFace Hub."""
    path: Optional[Path] = None
    if checkins_csv:
        path = _resolve_path(checkins_csv)
        if not path.is_file():
            path = None

    if path is None and hf_dataset_id and hf_filename:
        from huggingface_hub import hf_hub_download

        local_dir = cache_dir or "outputs/data_cache"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=hf_dataset_id,
            filename=hf_filename,
            repo_type="dataset",
            local_dir=local_dir,
        )
        path = Path(downloaded)

    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Could not locate check-ins CSV. Provide data.checkins_csv or "
            "data.hf_dataset_id + data.hf_filename."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Check-ins CSV missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "venue_id", "user_id", "trail_id"])
    df["venue_id"] = df["venue_id"].astype(int)
    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan
    # Fallback to city center when venue coords are missing (Massive-STEPS practice).
    if "venue_city_latitude" in df.columns:
        df["latitude"] = df["latitude"].fillna(df["venue_city_latitude"])
    if "venue_city_longitude" in df.columns:
        df["longitude"] = df["longitude"].fillna(df["venue_city_longitude"])
    if "venue_category" not in df.columns:
        df["venue_category"] = "UNK"
    df["venue_category"] = df["venue_category"].fillna("UNK").astype(str)
    df = df.sort_values(["user_id", "trail_id", "timestamp"]).reset_index(drop=True)
    return df


def build_vocab(df: pd.DataFrame) -> Vocab:
    """Build contiguous vocab from the full dataframe (transductive POI ids)."""
    pois = sorted(df["venue_id"].unique().tolist())
    poi2idx = {p: i for i, p in enumerate(pois)}
    users = sorted(df["user_id"].unique().tolist(), key=lambda x: str(x))
    user2idx = {u: i for i, u in enumerate(users)}
    cats = sorted(df["venue_category"].unique().tolist())
    category2idx = {c: i for i, c in enumerate(cats)}

    coords = np.full((len(pois), 2), np.nan, dtype=np.float64)
    cat_idx = np.zeros(len(pois), dtype=np.int64)
    for _, row in df.drop_duplicates("venue_id").iterrows():
        i = poi2idx[int(row["venue_id"])]
        coords[i, 0] = float(row["latitude"]) if pd.notna(row["latitude"]) else np.nan
        coords[i, 1] = float(row["longitude"]) if pd.notna(row["longitude"]) else np.nan
        cat_idx[i] = category2idx[str(row["venue_category"])]
    return Vocab(
        poi2idx=poi2idx,
        idx2poi=pois,
        user2idx=user2idx,
        category2idx=category2idx,
        poi_coords=coords,
        poi_category_idx=cat_idx,
    )


def _time_features(ts: pd.Timestamp) -> np.ndarray:
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    month = ts.month
    weekend = 1.0 if dow >= 5 else 0.0
    return np.array(
        [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0),
            math.cos(2 * math.pi * dow / 7.0),
            weekend,
            math.sin(2 * math.pi * (month - 1) / 12.0),
            math.cos(2 * math.pi * (month - 1) / 12.0),
            hour / 24.0,
        ],
        dtype=np.float32,
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), [lat1, lon1, lat2, lon2])):
        return float("nan")
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _spatial_features(
    hist_lats: Sequence[float],
    hist_lons: Sequence[float],
    hist_ts: Sequence[pd.Timestamp],
) -> np.ndarray:
    """Fixed-dim spatial context from history only (no candidate leakage)."""
    last_lat, last_lon = hist_lats[-1], hist_lons[-1]
    valid = [
        (la, lo)
        for la, lo in zip(hist_lats, hist_lons)
        if not (isinstance(la, float) and math.isnan(la)) and not (isinstance(lo, float) and math.isnan(lo))
    ]
    if valid:
        mean_lat = float(np.mean([v[0] for v in valid]))
        mean_lon = float(np.mean([v[1] for v in valid]))
        std_lat = float(np.std([v[0] for v in valid])) if len(valid) > 1 else 0.0
        std_lon = float(np.std([v[1] for v in valid])) if len(valid) > 1 else 0.0
    else:
        mean_lat = mean_lon = std_lat = std_lon = 0.0

    step_dists = []
    for i in range(1, len(hist_lats)):
        d = _haversine_km(hist_lats[i - 1], hist_lons[i - 1], hist_lats[i], hist_lons[i])
        if not math.isnan(d):
            step_dists.append(d)
    mean_step = float(np.mean(step_dists)) if step_dists else 0.0
    max_step = float(np.max(step_dists)) if step_dists else 0.0
    last_lat_f = 0.0 if (isinstance(last_lat, float) and math.isnan(last_lat)) else float(last_lat)
    last_lon_f = 0.0 if (isinstance(last_lon, float) and math.isnan(last_lon)) else float(last_lon)
    # Displacement from mean history location to last location.
    disp = _haversine_km(mean_lat, mean_lon, last_lat_f, last_lon_f)
    if math.isnan(disp):
        disp = 0.0
    span_h = 0.0
    if len(hist_ts) >= 2:
        span_h = max(0.0, (hist_ts[-1] - hist_ts[0]).total_seconds() / 3600.0)
    missing_coord = 1.0 if (isinstance(last_lat, float) and math.isnan(last_lat)) else 0.0
    return np.array(
        [
            last_lat_f / 90.0,
            last_lon_f / 180.0,
            mean_lat / 90.0,
            mean_lon / 180.0,
            std_lat,
            std_lon,
            math.log1p(mean_step),
            math.log1p(max_step),
            math.log1p(disp),
            math.log1p(span_h),
            float(len(hist_lats)),
            missing_coord,
        ],
        dtype=np.float32,
    )


def _user_features(user_rows: pd.DataFrame, upto_time: pd.Timestamp) -> np.ndarray:
    """Lightweight user stats from check-ins strictly before the prediction time."""
    past = user_rows[user_rows["timestamp"] < upto_time]
    n = len(past)
    if n == 0:
        return np.zeros(6, dtype=np.float32)
    n_unique = past["venue_id"].nunique()
    n_trails = past["trail_id"].nunique()
    cats = past["venue_category"].value_counts(normalize=True)
    entropy = float(-(cats * np.log(cats + 1e-12)).sum()) if len(cats) else 0.0
    hours = past["timestamp"].dt.hour
    return np.array(
        [
            math.log1p(n),
            math.log1p(n_unique),
            math.log1p(n_trails),
            float(n_unique) / float(n),
            entropy,
            float(hours.mean()) / 24.0,
        ],
        dtype=np.float32,
    )


def build_samples(
    df: pd.DataFrame,
    vocab: Vocab,
    max_history_len: int = 20,
    min_history_len: int = 1,
    use_all_prefixes: bool = True,
) -> List[NextPOISample]:
    """Construct next-POI samples from trails; features use only past context."""
    samples: List[NextPOISample] = []
    sid = 0
    # Pre-index per user for user_features.
    user_groups = {u: g for u, g in df.groupby("user_id", sort=False)}

    for trail_id, g in df.groupby("trail_id", sort=False):
        g = g.sort_values("timestamp")
        if len(g) < min_history_len + 1:
            continue
        poi_ids = [vocab.poi2idx[int(v)] for v in g["venue_id"].tolist()]
        ts_list = list(g["timestamp"])
        lats = [float(x) if pd.notna(x) else float("nan") for x in g["latitude"].tolist()]
        lons = [float(x) if pd.notna(x) else float("nan") for x in g["longitude"].tolist()]
        user_raw = g["user_id"].iloc[0]
        user_id = vocab.user2idx[user_raw]
        user_df = user_groups[user_raw]

        targets = range(min_history_len, len(g)) if use_all_prefixes else [len(g) - 1]
        for t in targets:
            hist_start = max(0, t - max_history_len)
            hist_pois = poi_ids[hist_start:t]
            hist_ts = ts_list[hist_start:t]
            hist_lats = lats[hist_start:t]
            hist_lons = lons[hist_start:t]
            if len(hist_pois) < min_history_len:
                continue
            target_ts = ts_list[t]
            samples.append(
                NextPOISample(
                    sample_id=sid,
                    user_id=user_id,
                    trail_id=str(trail_id),
                    history_poi_ids=hist_pois,
                    history_timestamps=hist_ts,
                    history_lats=hist_lats,
                    history_lons=hist_lons,
                    target_poi_id=poi_ids[t],
                    target_timestamp=target_ts,
                    target_time_features=_time_features(target_ts),
                    spatial_features=_spatial_features(hist_lats, hist_lons, hist_ts),
                    user_features=_user_features(user_df, target_ts),
                )
            )
            sid += 1
    return samples


def compute_poi_prior(samples: Sequence[NextPOISample], num_pois: int) -> np.ndarray:
    """Empirical P(Y=y) on the given samples (N_y / N), never 1/K by default."""
    counts = np.zeros(num_pois, dtype=np.float64)
    for s in samples:
        counts[s.target_poi_id] += 1.0
    total = counts.sum()
    if total <= 0:
        raise ValueError("Empty sample list for prior.")
    return counts / total


def poi_sample_counts(samples: Sequence[NextPOISample], num_pois: int) -> np.ndarray:
    counts = np.zeros(num_pois, dtype=np.int64)
    for s in samples:
        counts[s.target_poi_id] += 1
    return counts