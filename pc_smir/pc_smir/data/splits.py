"""Massive-STEPS official split loading / 官方划分加载。

Official partitions are trail-level CSVs produced by
`Massive-STEPS/src/create_next_poi_dataset.py`:

  - {city}_checkins_train.csv
  - {city}_checkins_validation.csv
  - {city}_checkins_test.csv

Construction (upstream): stratified by user_id, test_size=0.2 then
val_size=0.125 of remaining, random_state=42. This package never
re-splits; it only consumes those official files (local or HuggingFace).

两函数分工（do not confuse）:
  - ``load_official_splits``: 真正读入划分数据（DataFrame），供训练/特征使用。
  - ``split_manifest``: 只从已加载的 DataFrame 汇总元数据（计数、协议说明），
    用于审计/落盘 JSON；不读盘、不重新划分、不改数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

SplitName = Literal["train", "validation", "test"]

CITY_TO_HF: dict[str, str] = {
    "bandung": "CRUISEResearchGroup/Massive-STEPS-Bandung",
    "beijing": "CRUISEResearchGroup/Massive-STEPS-Beijing",
    "istanbul": "CRUISEResearchGroup/Massive-STEPS-Istanbul",
    "jakarta": "CRUISEResearchGroup/Massive-STEPS-Jakarta",
    "kuwait_city": "CRUISEResearchGroup/Massive-STEPS-Kuwait-City",
    "melbourne": "CRUISEResearchGroup/Massive-STEPS-Melbourne",
    "moscow": "CRUISEResearchGroup/Massive-STEPS-Moscow",
    "new_york": "CRUISEResearchGroup/Massive-STEPS-New-York",
    "palembang": "CRUISEResearchGroup/Massive-STEPS-Palembang",
    "petaling_jaya": "CRUISEResearchGroup/Massive-STEPS-Petaling-Jaya",
    "sao_paulo": "CRUISEResearchGroup/Massive-STEPS-Sao-Paulo",
    "shanghai": "CRUISEResearchGroup/Massive-STEPS-Shanghai",
    "sydney": "CRUISEResearchGroup/Massive-STEPS-Sydney",
    "tangerang": "CRUISEResearchGroup/Massive-STEPS-Tangerang",
    "tokyo": "CRUISEResearchGroup/Massive-STEPS-Tokyo",
}

REQUIRED_COLUMNS = [
    "trail_id",
    "user_id",
    "venue_id",
    "latitude",
    "longitude",
    "venue_category",
    "venue_category_id_code",
    "venue_city_latitude",
    "venue_city_longitude",
    "timestamp",
]


def city_slug(city: str) -> str:
    return city.strip().lower().replace("-", "_").replace(" ", "_")


def split_filenames(city: str) -> dict[SplitName, str]:
    slug = city_slug(city)
    return {
        "train": f"{slug}_checkins_train.csv",
        "validation": f"{slug}_checkins_validation.csv",
        "test": f"{slug}_checkins_test.csv",
    }


def ensure_official_splits(city: str, data_dir: str | Path, download: bool = True) -> Path:
    """Ensure official train/validation/test CSVs exist under data_dir.

    If missing and download=True, fetch from the city's HuggingFace dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = split_filenames(city)
    missing = [name for name in files.values() if not (data_dir / name).exists()]
    if not missing:
        return data_dir
    if not download:
        raise FileNotFoundError(
            f"Missing official split files in {data_dir}: {missing}. "
            "Pass download=True or place Massive-STEPS CSVs locally."
        )
    slug = city_slug(city)
    if slug not in CITY_TO_HF:
        raise KeyError(f"Unknown city '{city}'. Known: {sorted(CITY_TO_HF)}")
    from huggingface_hub import hf_hub_download

    repo_id = CITY_TO_HF[slug]
    for name in files.values():
        hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset", local_dir=str(data_dir))
    # Optional full CSV for audit; ignore if absent.
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{slug}_checkins.csv",
            repo_type="dataset",
            local_dir=str(data_dir),
        )
    except Exception:
        pass
    return data_dir


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing required column '{col}' in Massive-STEPS CSV")
    # Coord fallback matches Massive-STEPS Flashback dataloader convention.
    out["latitude"] = out["latitude"].fillna(out["venue_city_latitude"])
    out["longitude"] = out["longitude"].fillna(out["venue_city_longitude"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out = out.sort_values(["trail_id", "timestamp", "venue_id"]).reset_index(drop=True)
    # Drop exact duplicate rows.
    out = out.drop_duplicates(
        subset=["trail_id", "user_id", "venue_id", "timestamp", "latitude", "longitude"],
        keep="first",
    )
    # Flag invalid coordinates.
    valid = (
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
        & out["latitude"].notna()
        & out["longitude"].notna()
    )
    out = out.loc[valid].reset_index(drop=True)
    return out


def load_official_splits(city: str, data_dir: str | Path, download: bool = True) -> dict[SplitName, pd.DataFrame]:
    """Load Massive-STEPS official partitions as DataFrames. Never custom-random-splits.

    作用 / Role:
      从官方 CSV（必要时先下载）读入 train/validation/test 签到表，做轻量规范化
      （坐标回填、排序、去重、无效坐标过滤），并校验 trail_id 在各划分互不重叠。
      返回的是**真实数据**：``dict[split_name -> pd.DataFrame]``，后续建样本、词表、
      SURD 都依赖这里。

    与 ``split_manifest`` 的区别 / vs ``split_manifest``:
      本函数负责“加载数据”；``split_manifest`` 只在加载结果上做统计摘要，不读文件。
    """
    data_dir = ensure_official_splits(city, data_dir, download=download)
    files = split_filenames(city)
    splits: dict[SplitName, pd.DataFrame] = {}
    for split, name in files.items():
        df = pd.read_csv(data_dir / name)
        splits[split] = _normalize_frame(df)
    # Sanity: trail IDs must be disjoint across splits.
    trail_sets = {s: set(df["trail_id"].astype(str)) for s, df in splits.items()}
    if trail_sets["train"] & trail_sets["validation"]:
        raise RuntimeError("Official split integrity failed: train ∩ validation non-empty")
    if trail_sets["train"] & trail_sets["test"]:
        raise RuntimeError("Official split integrity failed: train ∩ test non-empty")
    if trail_sets["validation"] & trail_sets["test"]:
        raise RuntimeError("Official split integrity failed: validation ∩ test non-empty")
    return splits


def split_manifest(splits: dict[SplitName, pd.DataFrame]) -> dict:
    """Build a JSON-serializable summary of already-loaded official splits.

    作用 / Role:
      输入必须是 ``load_official_splits`` 已经返回的 DataFrame 字典；对本函数而言
      数据已在内存中。它只统计每个划分的行数/轨迹数/用户数/POI 数，并附上上游
      官方划分协议说明（脚本名与分层参数），方便写入 ``split_manifest.json`` 做
      可复现实验的审计记录。

    与 ``load_official_splits`` 的区别 / vs ``load_official_splits``:
      - ``load_official_splits`` → 读盘 + 规范化 + 完整性检查 → **数据**
      - ``split_manifest`` → 不读盘、不重新划分、不修改表 → **元数据摘要**
      典型用法：先 ``splits = load_official_splits(...)``，再
      ``json.dump(split_manifest(splits), ...)``。
    """
    return {
        split: {
            "n_rows": int(len(df)),
            "n_trails": int(df["trail_id"].nunique()),
            "n_users": int(df["user_id"].nunique()),
            "n_venues": int(df["venue_id"].nunique()),
            "protocol": "massive_steps_official_trail_split",
            "upstream_script": "Massive-STEPS/src/create_next_poi_dataset.py",
            "upstream_params": {
                "stratify": "user_id",
                "test_size": 0.2,
                "val_from_train_size": 0.125,
                "random_state": 42,
            },
        }
        for split, df in splits.items()
    }