"""Massive-STEPS official dataset paths and Flashback-compatible loading.

Follows conventions used by ``Massive-STEPS/Flashback`` and
``Massive-STEPS/GETNext``:

* Check-in schema from ``preprocess_std.py`` / HF city CSVs
* Official split files: ``{city}_checkins_{train,validation,test}.csv``
  (trail-level 7:1:2, user_id-stratified — see Massive-STEPS README /
  ``src/create_next_poi_dataset.py``)
* Coordinate fallback: venue lat/lon ← city center when missing
  (same as ``Flashback/dataloader.py``)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FLASHBACK_DIR = REPO_ROOT / "Massive-STEPS" / "Flashback"
MASSIVE_STEPS_DATA = REPO_ROOT / "Massive-STEPS" / "data"

CITY_HF_IDS = {
    "beijing": "CRUISEResearchGroup/Massive-STEPS-Beijing",
    "tokyo": "CRUISEResearchGroup/Massive-STEPS-Tokyo",
    "shanghai": "CRUISEResearchGroup/Massive-STEPS-Shanghai",
    "melbourne": "CRUISEResearchGroup/Massive-STEPS-Melbourne",
    "sydney": "CRUISEResearchGroup/Massive-STEPS-Sydney",
    "moscow": "CRUISEResearchGroup/Massive-STEPS-Moscow",
    "new_york": "CRUISEResearchGroup/Massive-STEPS-New-York",
    "sao_paulo": "CRUISEResearchGroup/Massive-STEPS-Sao-Paulo",
    "jakarta": "CRUISEResearchGroup/Massive-STEPS-Jakarta",
    "bandung": "CRUISEResearchGroup/Massive-STEPS-Bandung",
    "istanbul": "CRUISEResearchGroup/Massive-STEPS-Istanbul",
    "kuwait_city": "CRUISEResearchGroup/Massive-STEPS-Kuwait-City",
    "palembang": "CRUISEResearchGroup/Massive-STEPS-Palembang",
    "petaling_jaya": "CRUISEResearchGroup/Massive-STEPS-Petaling-Jaya",
    "tangerang": "CRUISEResearchGroup/Massive-STEPS-Tangerang",
}

SPLIT_SUFFIX = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def ensure_flashback_on_path() -> Path:
    """Put Massive-STEPS/Flashback on sys.path so we can import its modules."""
    path = str(FLASHBACK_DIR.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return FLASHBACK_DIR


def city_slug(city: str) -> str:
    return city.strip().lower().replace("-", "_").replace(" ", "_")


def split_filename(city: str, split: str) -> str:
    slug = city_slug(city)
    suffix = SPLIT_SUFFIX[split]
    return f"{slug}_checkins_{suffix}.csv"


def resolve_split_csv(
    city: str,
    split: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
    download: bool = True,
) -> Path:
    """
    Resolve official Massive-STEPS split CSV.

    Search order:
      1. ``data_dir / filename`` if provided
      2. ``Massive-STEPS/data/{city}/filename``
      3. ``cache_dir / filename``
      4. HuggingFace Hub download into cache_dir (same IDs as Massive-STEPS README)
    """
    slug = city_slug(city)
    filename = split_filename(slug, split)
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir) / filename)
    candidates.append(MASSIVE_STEPS_DATA / slug / filename)
    if cache_dir:
        candidates.append(Path(cache_dir) / filename)

    for path in candidates:
        if path.is_file():
            return path

    if not download:
        raise FileNotFoundError(
            f"Official split CSV not found for city={slug} split={split}: {filename}"
        )

    repo_id = hf_dataset_id or CITY_HF_IDS.get(slug)
    if not repo_id:
        raise FileNotFoundError(
            f"No HF dataset id for city={slug}. Pass data.hf_dataset_id or place {filename} locally."
        )
    from huggingface_hub import hf_hub_download

    out = Path(cache_dir) if cache_dir else Path("outputs/data_cache") / slug
    out.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(out),
    )
    return Path(downloaded)


def resolve_official_splits(
    city: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Return paths for train / validation / test official CSVs."""
    return {
        "train": resolve_split_csv(
            city, "train", data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
        ),
        "val": resolve_split_csv(
            city, "val", data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
        ),
        "test": resolve_split_csv(
            city, "test", data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
        ),
    }


def load_official_split_frames(
    city: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and lightly normalize the three official Massive-STEPS split CSVs."""
    paths = resolve_official_splits(
        city, data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
    )
    frames = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        required = {"trail_id", "user_id", "venue_id", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        if "latitude" not in df.columns:
            df["latitude"] = pd.NA
        if "longitude" not in df.columns:
            df["longitude"] = pd.NA
        # Flashback convention: fill missing venue coords with city center.
        if "venue_city_latitude" in df.columns:
            df["latitude"] = df["latitude"].fillna(df["venue_city_latitude"])
        if "venue_city_longitude" in df.columns:
            df["longitude"] = df["longitude"].fillna(df["venue_city_longitude"])
        if "venue_category" not in df.columns:
            df["venue_category"] = "UNK"
        df["venue_category"] = df["venue_category"].fillna("UNK").astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "venue_id", "user_id", "trail_id"])
        df["venue_id"] = df["venue_id"].astype(int)
        frames[name] = df.sort_values(["user_id", "trail_id", "timestamp"]).reset_index(drop=True)
    return frames


def load_flashback_poi_dataloader(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    *,
    max_users: int = 0,
    min_checkins: int = 0,
):
    """Instantiate Massive-STEPS Flashback ``PoiDataloader`` on official splits."""
    ensure_flashback_on_path()
    from dataloader import PoiDataloader  # type: ignore  # noqa: WPS433

    loader = PoiDataloader(max_users=max_users, min_checkins=min_checkins)
    loader.read(str(train_csv), str(val_csv), str(test_csv))
    return loader


def flashback_vocab_sizes(loader) -> Tuple[int, int]:
    return int(loader.locations()), int(loader.user_count())