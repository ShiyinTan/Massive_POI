"""Massive-STEPS official split CSVs and Flashback-compatible loading.

Fixed repo layout (no path discovery)::

    <Massive_POI>/
      Massive-STEPS/Flashback/   # network.py, dataloader.py, dataset.py
      Massive-STEPS/data/        # optional local city CSVs
      surd_guided_poi/

Conventions match ``Massive-STEPS/Flashback`` / ``GETNext``:

* Official files: ``{city}_checkins_{train,validation,test}.csv``
  (trail-level ~7:1:2, user_id-stratified)
* Missing venue lat/lon filled from city-center columns
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed relative paths from this file: surd_guided_poi/data/ → repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
FLASHBACK_DIR = REPO_ROOT / "Massive-STEPS" / "Flashback"
MASSIVE_STEPS_DATA = REPO_ROOT / "Massive-STEPS" / "data"

# Flashback modules use bare imports (``from network import ...``).
# Register that folder once; callers do not search or configure paths.
_FLASHBACK_PATH = str(FLASHBACK_DIR)
if _FLASHBACK_PATH not in sys.path:
    sys.path.insert(0, _FLASHBACK_PATH)

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

# Config/API key "val" maps to Massive-STEPS filename suffix "validation".
SPLIT_SUFFIX = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def city_slug(city: str) -> str:
    return city.strip().lower().replace("-", "_").replace(" ", "_")


def split_filename(city: str, split: str) -> str:
    return f"{city_slug(city)}_checkins_{SPLIT_SUFFIX[split]}.csv"


def resolve_split_csv(
    city: str,
    split: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
    download: bool = True,
) -> Path:
    """Locate one official split CSV.

    Order: ``data_dir`` → ``Massive-STEPS/data/{city}/`` → ``cache_dir`` → HF download.
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
            f"No HF dataset id for city={slug}. "
            f"Pass data.hf_dataset_id or place {filename} locally."
        )
    from huggingface_hub import hf_hub_download

    out = Path(cache_dir) if cache_dir else Path("outputs/data_cache") / slug
    out.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(out),
        )
    )


def resolve_official_splits(
    city: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Paths for train / val / test official CSVs."""
    kwargs = dict(data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id)
    return {
        "train": resolve_split_csv(city, "train", **kwargs),
        "val": resolve_split_csv(city, "val", **kwargs),
        "test": resolve_split_csv(city, "test", **kwargs),
    }


def _normalize_split_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Light cleanup aligned with Flashback/dataloader.py."""
    required = {"trail_id", "user_id", "venue_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")

    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA
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
    return df.sort_values(["user_id", "trail_id", "timestamp"]).reset_index(drop=True)


def load_official_split_frames(
    city: str,
    *,
    data_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    hf_dataset_id: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and normalize the three official Massive-STEPS split CSVs."""
    paths = resolve_official_splits(
        city, data_dir=data_dir, cache_dir=cache_dir, hf_dataset_id=hf_dataset_id
    )
    return {name: _normalize_split_frame(pd.read_csv(path), path) for name, path in paths.items()}


def load_flashback_poi_dataloader(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    *,
    max_users: int = 0,
    min_checkins: int = 0,
):
    """Optional: build Flashback's ``PoiDataloader`` (class in Massive-STEPS/Flashback/dataloader.py).

    Prefer ``load_official_split_frames`` + ``build_samples_from_split`` for the
    SURD training path; this helper is for direct Flashback-style loading.
    """
    # Import after module-level FLASHBACK_DIR registration above.
    from dataloader import PoiDataloader  # type: ignore

    loader = PoiDataloader(max_users=max_users, min_checkins=min_checkins)
    loader.read(str(train_csv), str(val_csv), str(test_csv))
    return loader


def flashback_vocab_sizes(loader) -> Tuple[int, int]:
    return int(loader.locations()), int(loader.user_count())
