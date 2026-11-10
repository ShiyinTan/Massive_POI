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

from pc_smir.data.vocab import haversine_m, spatial_cell_id
from pc_smir.features.temporal import day_type, hour_bin

# Q1 continuous summary sizes (before KMeans).
Q1_CAT_BINS = 32
Q1_POI_BINS = 64
# Multiplicative hash constant (Knuth) so consecutive poi_ids do not map to adjacent bins.
_POI_HASH_MULT = 2654435761


def _poi_bucket(poi_id: int, n_bins: int = Q1_POI_BINS) -> int:
    """Map a model POI id → fixed bag bucket (visit-based hashing, not semantic)."""
    return int((int(poi_id) * _POI_HASH_MULT) % n_bins)


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
      - 长什么样: ``_q1_vector`` ∈ R^{96} = 全历史类别袋 ``cat_hist[32]``
        + 全前缀 POI 访问袋 ``poi_bag[64]``（TF 占比），再 KMeans → 簇 id。
      - 怎么得到: 只用 example 的 ``history_poi_ids`` / ``history_categories``
        （来自 dataset 的前缀，不含 target）；**与访问顺序无关**。
      - 为何: 用“去过什么/去过多少次”的 bag 相似度，而不是轨迹槽位上的 id 指纹；
        SURD 需要低基数状态，KMeans 把相近访问画像归并。

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
        """Continuous Q1 summary before clustering / 聚类前的偏好摘要向量 (96,).

        两段都是**全前缀访问袋（bag / TF 占比）**，对顺序不敏感：
          ``cat_hist[32]`` + ``poi_bag[64]`` → KMeans → Q1 簇 id。

        设计动机 / Why (替换旧版 recent 槽位指纹):
          - 旧版把 ``poi_id % 97`` 填进时间槽，欧氏距离会把“编号碰巧相近”当相似，
            且强调顺序而非“去过谁/去过几次”。
          - 新版用固定桶上的**访问计数占比**：同 POI 多重访问抬高同一桶，
            相同访问多重集（可打乱顺序）→ 同一向量，更贴近 visit-based 相似。
          - id 数值本身仍无语义；桶只是压缩 catalog 的哈希地址。
          - KMeans 比较的是各 example 的 96 维袋向量，不是对 id 聚类。
          - 未来可加「轨迹」通道（如 last-k 有序 poi/cat 槽或时间衰减），
            与本 visit 袋并行；当前实现只做 visit-based，不做 dual-channel。
        """
        pois = ex["history_poi_ids"]
        cats = ex["history_categories"]

        # ----- cat_hist：全历史类别袋（归一化占比）-----
        # 桶 = category_id % 32：固定维数；比较的是各桶访问比例，不是 id 远近。
        cat_hist = np.zeros(Q1_CAT_BINS, dtype=np.float64)
        for c in cats:
            cat_hist[int(c) % Q1_CAT_BINS] += 1.0
        if cat_hist.sum() > 0:
            cat_hist /= cat_hist.sum()

        # ----- poi_bag：全前缀 POI 访问袋（hashed bag-of-POI, TF）-----
        # 每个历史 POI 贡献 +1 到哈希桶；多重访问累加；最后除以总次数 → 占比。
        # 乘法哈希打散连续 id，避免“相邻编号 → 相邻桶”的假结构。
        # 与顺序无关：同一多重集访问 → 同一 poi_bag。
        poi_bag = np.zeros(Q1_POI_BINS, dtype=np.float64)
        for p in pois:
            poi_bag[_poi_bucket(int(p))] += 1.0
        if poi_bag.sum() > 0:
            poi_bag /= poi_bag.sum()

        return np.concatenate([cat_hist, poi_bag]).astype(np.float64)

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
