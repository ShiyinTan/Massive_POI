"""Source-state construction for SURD estimation (discrete, train-fitted)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from pc_smir.data.vocab import VocabMaps, haversine_m, spatial_cell_id
from pc_smir.features.stable_user import stable_user_features
from pc_smir.features.temporal import day_type, hour_bin


@dataclass
class SourceDiscretizer:
    """Train-only KMeans / bin transforms for SURD source states."""

    n_clusters: int = 8
    hour_bins: int = 8
    q1_model: Any | None = None
    q3_model: Any | None = None
    q4_model: Any | None = None
    q4_mean: np.ndarray | None = None
    q4_std: np.ndarray | None = None
    missing_state: int = -1

    def fit(self, examples: list[dict]) -> "SourceDiscretizer":
        q1 = np.stack([self._q1_vector(ex) for ex in examples], axis=0)
        q3 = np.stack([self._q3_vector(ex) for ex in examples], axis=0)
        q4 = np.stack([ex["stable_user_features"] for ex in examples], axis=0)
        self.q4_mean = q4.mean(axis=0)
        self.q4_std = np.clip(q4.std(axis=0), 1e-6, None)
        q4n = (q4 - self.q4_mean) / self.q4_std

        k = min(self.n_clusters, max(2, len(examples) // 10))
        self.q1_model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(q1)
        self.q3_model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(q3)
        self.q4_model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(q4n)
        return self

    def transform_example(self, ex: dict) -> dict[str, int]:
        if self.q1_model is None:
            raise RuntimeError("SourceDiscretizer must be fit before transform")
        q1 = int(self.q1_model.predict(self._q1_vector(ex)[None, :])[0])
        q2 = hour_bin(ex["query_time"], self.hour_bins) * 2 + day_type(ex["query_time"])
        q3 = int(self.q3_model.predict(self._q3_vector(ex)[None, :])[0])
        q4v = (ex["stable_user_features"] - self.q4_mean) / self.q4_std
        q4 = int(self.q4_model.predict(q4v[None, :])[0])
        return {"Q1": q1, "Q2": int(q2), "Q3": q3, "Q4": q4}

    def _q1_vector(self, ex: dict) -> np.ndarray:
        pois = ex["history_poi_ids"]
        cats = ex["history_categories"]
        # Preference summary: normalized category histogram + recency-weighted POI hash features.
        cat_hist = np.zeros(32, dtype=np.float64)
        for c in cats:
            cat_hist[int(c) % 32] += 1.0
        if cat_hist.sum() > 0:
            cat_hist /= cat_hist.sum()
        recent = np.zeros(16, dtype=np.float64)
        for i, p in enumerate(pois[-16:]):
            recent[i] = (int(p) % 97) / 97.0
        return np.concatenate([cat_hist, recent]).astype(np.float64)

    def _q3_vector(self, ex: dict) -> np.ndarray:
        coords = ex["history_coordinates"]
        if not coords:
            return np.zeros(6, dtype=np.float64)
        lat, lon = coords[-1]
        cell = spatial_cell_id(lat, lon, ex["vocab"])
        moves = []
        for i in range(1, len(coords)):
            moves.append(haversine_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
        med = float(np.median(moves)) if moves else 0.0
        return np.asarray(
            [
                lat,
                lon,
                (cell % 1000) / 1000.0,
                np.log1p(med),
                float(len(coords)),
                float(len({(round(a, 3), round(b, 3)) for a, b in coords})),
            ],
            dtype=np.float64,
        )


def build_prefix_example(
    user_id: int,
    trail_id: str,
    hist_rows: list[dict],
    target_venue_id: int,
    vocab: VocabMaps,
    max_len: int,
) -> dict | None:
    """Build one next-POI example from trail prefix (excludes target event)."""
    if len(hist_rows) < 1:
        return None
    hist_rows = hist_rows[-max_len:]
    poi_ids = [vocab.encode_poi(r["venue_id"]) for r in hist_rows]
    cat_ids = [vocab.encode_cat(r["venue_category"]) for r in hist_rows]
    coords = [(float(r["latitude"]), float(r["longitude"])) for r in hist_rows]
    times = [r["timestamp"] for r in hist_rows]
    # Prediction timestamp = last observed check-in (not target timestamp).
    query_time = times[-1]
    time_gaps = [0.0]
    for i in range(1, len(times)):
        gap = max(0.0, (times[i] - times[i - 1]).total_seconds() / 3600.0)
        time_gaps.append(float(np.log1p(gap)))

    cells = [spatial_cell_id(a, b, vocab) for a, b in coords]
    stable = stable_user_features(poi_ids, cat_ids, coords, vocab)
    target_id = vocab.encode_poi(target_venue_id)
    return {
        "user_id": int(user_id),
        "trail_id": str(trail_id),
        "history_poi_ids": poi_ids,
        "history_categories": cat_ids,
        "history_coordinates": coords,
        "history_cells": cells,
        "history_time_gaps": time_gaps,
        "history_hours": [int(t.hour) for t in times],
        "history_dows": [int(t.dayofweek) for t in times],
        "query_time": query_time,
        "stable_user_features": stable,
        "target_poi_id": int(target_id),
        "target_venue_raw": int(target_venue_id),
        "last_lat": coords[-1][0],
        "last_lon": coords[-1][1],
        "vocab": vocab,
        "source_available": np.ones(4, dtype=np.float32),
    }