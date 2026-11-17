"""Load Massive-STEPS official next-POI split CSVs.

The project ``data`` directory is a symlink to ``Massive-STEPS/data``.
Official files, when present, live at::

    data/<city>/<city>_checkins_{train,validation,test}.csv

Columns follow ``Massive-STEPS/src/preprocess_std.py``: ``trail_id``,
``user_id``, ``venue_id``, ``latitude``, ``longitude``, ``timestamp``,
and optionally ``venue_category`` plus city-center coordinates.
This loader does not download data. The checked-in Massive-STEPS tree
ships check-in statistics and boundary geojson, not the split tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data"

SPLIT_SUFFIX = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    path = DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def split_csv_path(data_dir: Path, city: str, split: str) -> Path:
    slug = city.strip().lower().replace("-", "_").replace(" ", "_")
    filename = f"{slug}_checkins_{SPLIT_SUFFIX[split]}.csv"
    direct = data_dir / filename
    nested = data_dir / slug / filename
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(
        f"Missing Massive-STEPS split CSV for city={slug!r} split={split!r}. "
        f"Looked for {direct} and {nested}. "
        "Place the official {{city}}_checkins_{{train,validation,test}}.csv "
        "files under the symlinked data directory "
        "(ts_moe/data → ../Massive-STEPS/data)."
    )


def _normalize(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"trail_id", "user_id", "venue_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame = df.copy()
    if "latitude" not in frame.columns:
        frame["latitude"] = pd.NA
    if "longitude" not in frame.columns:
        frame["longitude"] = pd.NA
    if "venue_city_latitude" in frame.columns:
        frame["latitude"] = frame["latitude"].fillna(frame["venue_city_latitude"])
    if "venue_city_longitude" in frame.columns:
        frame["longitude"] = frame["longitude"].fillna(frame["venue_city_longitude"])
    if "venue_category" not in frame.columns:
        frame["venue_category"] = "UNK"
    frame["venue_category"] = frame["venue_category"].fillna("UNK").astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "venue_id", "user_id", "trail_id"])
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    # A missing coordinate becomes the trail median, then 0. The model still
    # sees a location channel; city-center fill above covers the official files.
    if frame["latitude"].isna().any():
        frame["latitude"] = frame["latitude"].fillna(frame["latitude"].median())
    if frame["longitude"].isna().any():
        frame["longitude"] = frame["longitude"].fillna(frame["longitude"].median())
    frame["latitude"] = frame["latitude"].fillna(0.0)
    frame["longitude"] = frame["longitude"].fillna(0.0)
    return frame.sort_values(["user_id", "trail_id", "timestamp"]).reset_index(drop=True)


def load_split_frame(data_dir: Path, city: str, split: str) -> pd.DataFrame:
    path = split_csv_path(data_dir, city, split)
    return _normalize(pd.read_csv(path), path)


def load_city_frames(city: str, data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    root = resolve_data_dir(data_dir)
    return {
        "train": load_split_frame(root, city, "train"),
        "val": load_split_frame(root, city, "val"),
        "test": load_split_frame(root, city, "test"),
    }
