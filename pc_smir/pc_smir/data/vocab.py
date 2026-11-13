"""Train-only vocabularies and preprocessing statistics / 仅训练集拟合的词表与预处理统计。

Typical Beijing (Massive-STEPS official train CSV) scale used in comments below
北京官方 train 规模（注释中的数字均来自此）:
  - beijing_checkins_train.csv ≈ 1044 rows, 498 venues, 123 categories
  - after fit_vocab: n_pois=500, n_cats=125  (含 pad/oov)
  - lat ≈ [39.72, 40.12], lon ≈ [116.19, 116.64]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# 特殊 ID / reserved model IDs (never assigned to a real Beijing venue/category).
PAD_ID = 0  # 序列填充；embedding padding_idx。例：history 右侧对齐时左侧 poi=0
OOV_ID = 1  # 训练集未见过的 venue/category（UNK）。例：encode_poi(999999) → 1
# Under eval.sthgcn_protocol, history-prefix OOV POIs also map here for embedding lookup;
# samples are not dropped solely for history OOV (only unseen *target* / *user* are).


@dataclass
class VocabMaps:
    """Container for train-fitted ID maps and spatial stats / 训练集词表与空间统计容器。

    Built once by ``fit_vocab(train_df)`` on Beijing train (or another city), then
    ``save``/``load`` for train & eval. Val/test must NOT refit these fields.
    由 ``fit_vocab`` 在 train 上拟合一次后复用；验证/测试禁止重新拟合。
    """

    # --- ID maps / 整数词表 ---
    # 原始 venue_id 字符串 → 模型整数。北京例：{"<pad>":0,"<oov>":1,"0":2,"10":3,...,"84":420,...}
    poi2id: dict[str, int] = field(default_factory=dict)
    # 反向映射，便于调试。北京例：{0:"<pad>", 1:"<oov>", 420:"84", ...}
    id2poi: dict[int, str] = field(default_factory=dict)
    # 类别名 → 模型整数。北京例：{"<pad>":0,"<oov>":1,"American Restaurant":2,...,"Metro Station":79,...}
    cat2id: dict[str, int] = field(default_factory=dict)
    # 反向类别映射。北京例：{79:"Metro Station", ...}
    id2cat: dict[int, str] = field(default_factory=dict)

    # --- POI side information from train rows / 来自 train 行的 POI 侧信息 ---
    # 模型 POI id → 类别 id（最后一次出现覆盖）。北京例：venue 161→Café 的编码
    poi_category: dict[int, int] = field(default_factory=dict)  # poi_id -> cat_id
    # 模型 POI id → (lat, lon)。北京例：某 Café≈(39.9829, 116.4928)
    poi_coords: dict[int, tuple[float, float]] = field(default_factory=dict)
    # 训练集访问次数（按模型 POI id）。北京最热门 venue_id=84 → encode 420，count=37
    train_poi_counts: dict[int, int] = field(default_factory=dict)

    # --- Coordinate normalization (train mean/std) / 坐标标准化参数 ---
    # 北京 train 实测约：lat_mean≈39.9388, lat_std≈0.0500
    lat_mean: float = 0.0
    lat_std: float = 1.0
    # 北京 train 实测约：lon_mean≈116.4575, lon_std≈0.0475
    lon_mean: float = 0.0
    lon_std: float = 1.0

    # --- Spatial grid / 空间格子 ---
    # 格子边长（米）。默认 500；北京 smoke/base 配置同此值
    spatial_cell_size_m: float = 500.0
    # 格子原点 = train 纬度/经度最小值。北京约 lat_origin≈39.7241, lon_origin≈116.1923
    lat_origin: float = 0.0
    lon_origin: float = 0.0

    # --- Sizes for embedding tables / embedding 表大小 ---
    # 北京：n_pois=500 (=498 真实 + pad + oov), n_cats=125 (=123 + pad + oov)
    n_pois: int = 2
    n_cats: int = 2

    def encode_poi(self, venue_id: Any) -> int:
        """Map raw venue_id → model int; unknown → OOV_ID.

        作用：把 Massive-STEPS CSV 里的 venue_id 编成 embedding 下标。
        北京例：
          encode_poi(84)     → 420   # train 中出现 37 次的热门 POI
          encode_poi(161)    → 某 ≥2 的 id（首行 Café）
          encode_poi(999999) → 1     # 训练集没有的 venue → OOV
        """
        key = str(venue_id)
        return self.poi2id.get(key, OOV_ID)

    def encode_cat(self, category: Any) -> int:
        """Map category name → model int; NaN→'<missing>'; unknown→OOV_ID.

        作用：编码 venue_category 字符串。
        北京例：
          encode_cat("Metro Station") → 79   # train 中最常见类别之一（93 次）
          encode_cat("Café")          → 某个 ≥2 的 id
          encode_cat(None)            → "<missing>" 的 id（若 train 出现过）或 OOV
          encode_cat("DefinitelyNotACategory") → 1
        """
        key = str(category) if pd.notna(category) else "<missing>"
        return self.cat2id.get(key, OOV_ID)

    def save(self, path: str | Path) -> None:
        """Persist this vocab to JSON / 将整份词表写成 JSON。

        北京流水线：prepare 后写入 artifacts/beijing[_smoke]/vocab.json，
        供后续 train/eval 复用，避免再次扫描 CSV。
        """
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
        """Reload a previously saved vocab / 读回 save 的 JSON。

        北京例：VocabMaps.load("artifacts/beijing_smoke/vocab.json")
        后 encode_poi(84) 仍为 420（与 fit 时一致）。
        """
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
    """Fit ID maps and normalization stats on training split only.

    作用 / Role:
      只在官方 train 划分上拟合词表与空间统计；val/test 禁止参与。
      北京：读 beijing_checkins_train.csv（≈1044 行）→ n_pois=500, n_cats=125。

    输入 / Inputs:
      train_df: 至少含 venue_id, venue_category, latitude, longitude
      spatial_cell_size_m: 空间格子边长（米），北京默认/配置多为 500

    输出 / Output:
      填好字段的 VocabMaps（见类字段注释中的北京数值）

    Pipeline: prepare_artifacts → fit_vocab(splits["train"]) → vocab.save(...)
    """
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
    """Deterministic spatial cell from train-fitted origin and resolution.

    作用 / Role:
      用 train 拟合的原点与格子边长，把经纬度量化成非负格子 ID（0 留给 pad；
      返回值从 2 起）。北京同城近邻点常落在同一 500m 格。

    北京例（fit 后 lat_origin≈39.7241, lon_origin≈116.1923, cell=500m）:
      首条 train 坐标 (39.98295, 116.49279) → spatial_cell_id ≈ 5700224
      纬度 +0.0045°（约 +500m）→ 格子变为 ≈ 5800227

    Pipeline: ``dataset.build_prefix_example`` 与 SURD Q3（``SourceDiscretizer``）调用。
    """
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
    """Great-circle distance in meters / 两点球面距离（米）。

    作用 / Role:
      与词表无关的几何工具；用于候选 POI 相对“最后观测位置”的距离特征，
      以及稳定用户画像中的位移/活动半径。

    北京例（train 前两行）:
      (39.98295, 116.49279) → (39.98305, 116.48868) ≈ 350.4 m
      同一点 → 0.0 m
    """
    r = 6371000.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
