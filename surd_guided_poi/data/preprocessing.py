"""Build next-POI samples from Massive-STEPS official split CSVs.

Uses the same trail grouping / coordinate fallback conventions as
``Massive-STEPS/Flashback/dataloader.py``. Vocabulary is **transductive**
across train+val+test venue_ids (Flashback builds poi2id the same way).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Vocab:
    poi2idx: Dict[int, int]
    idx2poi: List[int]
    user2idx: Dict[Any, int]
    category2idx: Dict[str, int]
    poi_coords: np.ndarray
    poi_category_idx: np.ndarray

    @property
    def num_pois(self) -> int:
        return len(self.idx2poi)

    @property
    def num_users(self) -> int:
        return len(self.user2idx)


@dataclass
class NextPOISample:
    """One supervised next-POI example / 一条训练样本（不是原始 check-in 行）.

    Diff from raw CSV / 与原始 CSV 的差别：
      raw: 多行轨迹 check-in
      sample: 截取 history → 预测下一个 ``target_poi_id``

    Toy example / 示意（trail 有 3 次打卡，预测第 3 次）::

        history_poi_ids = [0, 3]          # vocab indices, not raw venue_id
        history_unix    = [1577854800.0, 1577865600.0]
        target_poi_id   = 5
        target_time_features.shape = (8,)   # hour/dow/...
        spatial_features.shape     = (12,)
    """

    sample_id: int
    user_id: int  # vocab user index
    trail_id: str
    history_poi_ids: List[int]
    history_timestamps: List[pd.Timestamp]
    history_lats: List[float]
    history_lons: List[float]
    # Unix seconds for Flashback temporal kernel (same as Flashback dataloader).
    history_unix: List[float]
    target_poi_id: int
    target_timestamp: pd.Timestamp
    target_time_features: np.ndarray
    spatial_features: np.ndarray
    user_features: np.ndarray
    split: str = "train"


def build_vocab_from_frames(frames: Mapping[str, pd.DataFrame]) -> Vocab:
    """Transductive vocab over all official splits (Flashback-compatible)."""
    all_df = pd.concat(list(frames.values()), ignore_index=True)
    pois = sorted(all_df["venue_id"].unique().tolist())
    poi2idx = {p: i for i, p in enumerate(pois)}
    users = sorted(all_df["user_id"].unique().tolist(), key=lambda x: str(x))
    user2idx = {u: i for i, u in enumerate(users)}
    cats = sorted(all_df["venue_category"].astype(str).unique().tolist())
    category2idx = {c: i for i, c in enumerate(cats)}

    coords = np.full((len(pois), 2), np.nan, dtype=np.float64)
    cat_idx = np.zeros(len(pois), dtype=np.int64)
    for _, row in all_df.drop_duplicates("venue_id").iterrows():
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
    dow = int(ts.day_of_week)
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


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return float("nan")
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _spatial_features(hist_lats, hist_lons, hist_ts) -> np.ndarray:
    last_lat, last_lon = hist_lats[-1], hist_lons[-1]
    valid = [
        (la, lo)
        for la, lo in zip(hist_lats, hist_lons)
        if not (isinstance(la, float) and math.isnan(la))
        and not (isinstance(lo, float) and math.isnan(lo))
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


def build_samples_from_split(
    df: pd.DataFrame,
    vocab: Vocab,
    split_name: str,
    *,
    max_history_len: int = 20,
    min_history_len: int = 1,
    use_all_prefixes: bool = True,
    user_history_df: Optional[pd.DataFrame] = None,
) -> List[NextPOISample]:
    """CSV rows → list[NextPOISample] for one official split.

    Grouping / 分组: ``groupby([user_id, trail_id])`` (same as Flashback).

    Length rules / 长度规则（默认 config）::

        min_history_len = 1   # need ≥1 history POI → trail needs ≥2 check-ins
        max_history_len = 20  # truncate history to last 20 (NOT a user filter)

    Does **not** drop users whose total check-ins < 20.

    Toy trail (3 check-ins) with ``use_all_prefixes=True`` → 2 samples::

        check-ins: [vA, vB, vC]
        sample0: history=[vA]     → target=vB
        sample1: history=[vA,vB]  → target=vC
    """
    samples: List[NextPOISample] = []
    sid = 0
    history_source = df if user_history_df is None else user_history_df
    user_groups = {u: g for u, g in history_source.groupby("user_id", sort=False)}

    for (user_raw, trail_id), g in df.groupby(["user_id", "trail_id"], sort=False):
        g = g.sort_values("timestamp")
        if len(g) < min_history_len + 1:
            continue
        poi_ids = [vocab.poi2idx[int(v)] for v in g["venue_id"].tolist()]
        ts_list = list(g["timestamp"])
        unix = [float(ts.value // 10**9) for ts in ts_list]  # Flashback: ns → seconds
        lats = [float(x) if pd.notna(x) else float("nan") for x in g["latitude"].tolist()]
        lons = [float(x) if pd.notna(x) else float("nan") for x in g["longitude"].tolist()]
        user_id = vocab.user2idx[user_raw]
        user_df = user_groups.get(user_raw, g)

        targets = range(min_history_len, len(g)) if use_all_prefixes else [len(g) - 1]
        for t in targets:
            hist_start = max(0, t - max_history_len)
            hist_pois = poi_ids[hist_start:t]
            hist_ts = ts_list[hist_start:t]
            hist_unix = unix[hist_start:t]
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
                    history_unix=hist_unix,
                    target_poi_id=poi_ids[t],
                    target_timestamp=target_ts,
                    target_time_features=_time_features(target_ts),
                    spatial_features=_spatial_features(hist_lats, hist_lons, hist_ts),
                    user_features=_user_features(user_df, target_ts),
                    split=split_name,
                )
            )
            sid += 1
    return samples


def compute_poi_prior(samples: Sequence[NextPOISample], num_pois: int) -> np.ndarray:
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


# Back-compat aliases used by older scripts/tests.
def build_vocab(df: pd.DataFrame) -> Vocab:
    return build_vocab_from_frames({"all": df})


def build_samples(df, vocab, max_history_len=20, min_history_len=1, use_all_prefixes=True):
    return build_samples_from_split(
        df,
        vocab,
        split_name="custom",
        max_history_len=max_history_len,
        min_history_len=min_history_len,
        use_all_prefixes=use_all_prefixes,
    )


def load_checkins(*args, **kwargs):
    raise RuntimeError(
        "Custom single-CSV loading was removed. Use "
        "surd_guided_poi.data.massive_steps.load_official_split_frames "
        "to follow Massive-STEPS official train/validation/test splits."
    )