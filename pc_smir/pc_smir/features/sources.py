"""Q1–Q4 source discretization for offline SURD estimation.

职责边界 / Ownership:
  - 本模块只负责「信息源 → 离散状态」：``SourceDiscretizer``。
  - next-POI **example 构造**（trail 前缀 → dict）在 ``pc_smir.data.dataset``，
    不放这里，避免 dataset 与 SURD 源离散化搅在一起。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from pc_smir.data.vocab import VocabMaps, haversine_m, spatial_cell_id
from pc_smir.features.temporal import day_type, hour_bin


@dataclass
class SourceDiscretizer:
    """Train-only KMeans / bin transforms for SURD source states.

    作用 / Role（被 ``pipeline.prepare_artifacts`` 调用）:
      把每条 next-POI example 的四个语义源压成**小基数离散状态**，供离线 SURD
      估计器使用；**不是**模型里 E1–E4 的连续向量。

    ------------------------------------------------------------------
    四个源分别是什么、怎么得到、为何这样设计
    ------------------------------------------------------------------

    **Q1 个人偏好 (Personal preference)**
      - 长什么样: 先得到连续摘要向量 ``_q1_vector`` ∈ R^{48}
        （32 维类别直方图 + 16 维近期 POI hash），再 KMeans → 整数簇 id。
      - 怎么得到: 只用 example 的 ``history_poi_ids`` / ``history_categories``
        （来自 dataset 的前缀，不含 target）。
      - 为何: SURD 需要低基数类别；直接用 embedding 会随训练漂移且维数过高。
        簇 id 把“相似偏好画像”归并，便于估 unique/synergy。

    **Q2 时间上下文 (Temporal)**
      - 长什么样: 单个整数 ``hour_bin * 2 + day_type``。
        ``hour_bin`` ∈ [0, hour_bins)（默认 8 段，每段约 3 小时）；
        ``day_type`` ∈ {0=工作日, 1=周末} → 合计最多 ``hour_bins*2`` 个状态
        （默认 16；北京 smoke 实测值域约 [0,15]）。
      - 怎么得到: 读 ``ex["query_time"]``（最后一次**历史**签到时间，不是 target）。
      - 为何: 时间本身已是低基数；无需 KMeans。保留 hour×周末 交互，
        对应“何时出门”这一源，与偏好/空间语义分开。

    **Q3 空间上下文 (Spatial)**
      - 长什么样: 连续摘要 ``_q3_vector`` ∈ R^{6}
        （末点 lat/lon、格子哈希、中位位移、历史长度、独特位置数），
        再 KMeans → 整数簇 id。
      - 怎么得到: ``history_coordinates`` + train 拟合的 ``spatial_cell_id``
        （需要 ``ex["vocab"]``）；**不用 target 坐标当当前位置**。
      - 为何: 原始经纬度几乎唯一，列联表会稀疏；聚类成区域/移动模式状态。

    **Q4 稳定用户画像 (Stable user)**
      - 长什么样: 直接用 example 里已算好的 ``stable_user_features`` ∈ R^{8}
        （访问量、多样性、熵、位移、半径等，见 ``features/stable_user.py``），
        用 train 的 mean/std 标准化后再 KMeans → 整数簇 id。
      - 怎么得到: dataset 在建 example 时写入该 8 维向量；本类只做标准化+聚类。
      - 为何: 与短时序列偏好（Q1）分开，表示更慢变的用户活跃/流动性档位。

    fit(train examples) → 学 Q1/Q3/Q4 的 KMeans 与 Q4 标准化；Q2 无固定规则。
    transform_example(ex) → {"Q1","Q2","Q3","Q4"} 整数状态。

    北京 smoke（n_clusters=4）对象字段示例:
      disc.q4_mean / q4_std: np.float32, shape (8,)
      disc.q*_model.n_clusters_: 4
      transform 一例: {"Q1": 1, "Q2": 6, "Q3": 2, "Q4": 1}
    """

    n_clusters: int = 8
    hour_bins: int = 8
    q1_model: Any | None = None
    q3_model: Any | None = None
    q4_model: Any | None = None
    q4_mean: np.ndarray | None = None
    q4_std: np.ndarray | None = None
    missing_state: int = -1

    def fit(self, examples: list[dict]) -> "SourceDiscretizer":
        """Fit cluster/bin maps on training examples only / 仅用 train 拟合。"""
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
        """Freeze train-fitted maps; emit one discrete state per source Q1–Q4.

        输出例: {"Q1": 2, "Q2": 5, "Q3": 1, "Q4": 0} —— 供 SURD 列联表，而非 embedding。
        """
        if self.q1_model is None:
            raise RuntimeError("SourceDiscretizer must be fit before transform")
        q1 = int(self.q1_model.predict(self._q1_vector(ex)[None, :])[0])
        # Q2: deterministic hour×day-type code (no cluster model).
        q2 = hour_bin(ex["query_time"], self.hour_bins) * 2 + day_type(ex["query_time"])
        q3 = int(self.q3_model.predict(self._q3_vector(ex)[None, :])[0])
        q4v = (ex["stable_user_features"] - self.q4_mean) / self.q4_std
        q4 = int(self.q4_model.predict(q4v[None, :])[0])
        return {"Q1": q1, "Q2": int(q2), "Q3": q3, "Q4": q4}

    def _q1_vector(self, ex: dict) -> np.ndarray:
        """Continuous Q1 summary before clustering / 聚类前的偏好摘要向量 (48,)."""
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
        """Continuous Q3 summary before clustering / 聚类前的空间摘要向量 (6,)."""
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
