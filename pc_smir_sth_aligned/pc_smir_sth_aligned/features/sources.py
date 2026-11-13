"""Q1–Q4 source discretization for offline SURD estimation.

职责边界 / Ownership:
  - 本模块只负责「信息源 → 离散状态」：``SourceDiscretizer``。
  - next-POI **example 构造**（trail 前缀 → dict）在 ``pc_smir.data.dataset``，
    不放这里，避免 dataset 与 SURD 源离散化搅在一起。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from pc_smir_sth_aligned.data.vocab import haversine_m, spatial_cell_id
from pc_smir_sth_aligned.features.temporal import day_type, hour_bin

# MinHash modulus (large prime) and default signature length.
MINHASH_PRIME = 2_147_483_647  # 2^31 - 1
Q1_N_MINHASH = 64


def _token_to_int(token: str) -> int:
    """Stable string → int for MinHash (process-independent)."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % MINHASH_PRIME


def build_visit_set(ex: dict) -> set[str]:
    """Full-prefix visit set for Q1 MinHash.

    选择 / Choice:
      - **主信号**: 历史中出现过的 POI（``p:{id}``），刻画“去过哪些点”。
      - **可选并入**: 历史类别（``c:{id}``），刻画“去过哪些类型”。
      两者用命名空间前缀避免 id 空间碰撞；用 **集合**（去重）对应 Jaccard 访问相似，
      不含 target，与顺序无关。
    """
    tokens: set[str] = set()
    for p in ex.get("history_poi_ids", []):
        tokens.add(f"p:{int(p)}")
    for c in ex.get("history_categories", []):
        tokens.add(f"c:{int(c)}")
    return tokens


def minhash_signature(
    tokens: set[str],
    a: np.ndarray,
    b: np.ndarray,
    prime: int = MINHASH_PRIME,
) -> np.ndarray:
    """Compute length-|a| MinHash signature in [0,1). Empty set → ones."""
    k = int(a.shape[0])
    if not tokens:
        return np.ones(k, dtype=np.float64)
    vals = np.fromiter((_token_to_int(t) for t in tokens), dtype=np.int64, count=len(tokens))
    vals = vals % prime
    # For each hash i: min over tokens of (a[i]*v + b[i]) % P
    # Broadcast: (k, 1) * (1, n) + (k, 1)
    av = (a.astype(np.int64)[:, None] * vals[None, :]) + b.astype(np.int64)[:, None]
    # Reduce mod P without relying on overflow wrapping
    av = np.mod(av, prime)
    sig_int = av.min(axis=1)
    return sig_int.astype(np.float64) / float(prime)


@dataclass
class SourceDiscretizer:
    """Train-only KMeans / bin transforms for SURD source states.

    作用 / Role（被 ``pipeline.prepare_artifacts`` 调用）:
      把每条 next-POI example 的四个语义源压成**小基数离散状态**，供离线 SURD
      估计器使用；**不是**模型里 E1–E4 的连续向量。

    ------------------------------------------------------------------
    四个源分别是什么、怎么得到、为何这样设计
    ------------------------------------------------------------------

    **Q1 个人偏好 (Personal preference) — MinHash + KMeans**
      - 访问集合: 全前缀 POI∪类别（见 ``build_visit_set``）。
      - MinHash: 固定 ``n_minhash`` 个哈希（``minhash_seed`` 仅在 fit 时采样 a,b），
        得到签名向量 ≈ Jaccard 相似的低维表示。
      - KMeans: 在 train 签名上聚类 → 离散 Q1 状态。
      - 含义: 聚的是“访问集合相近”的样本，不是 id 数值邻近，也不是轨迹顺序。

    **Q2 时间上下文 (Temporal)**
      - 单个整数 ``hour_bin * 2 + day_type``（固定规则，无 KMeans）。

    **Q3 空间上下文 (Spatial)**
      - ``_q3_vector`` ∈ R^{6} 再 KMeans。

    **Q4 稳定用户画像 (Stable user)**
      - ``stable_user_features[8]`` 标准化后 KMeans。

    fit(train examples) → 固定 MinHash 参数 + 学 Q1/Q3/Q4 的 KMeans；Q2 用固定规则。
    transform_example(ex) → {"Q1","Q2","Q3","Q4"} 整数状态。
    """

    n_clusters: int = 8
    hour_bins: int = 8
    n_minhash: int = Q1_N_MINHASH
    minhash_seed: int = 42
    q1_model: Any | None = None
    q3_model: Any | None = None
    q4_model: Any | None = None
    q4_mean: np.ndarray | None = None
    q4_std: np.ndarray | None = None
    # Train-only MinHash coefficients: h_i(x) = (a_i * x + b_i) mod P
    minhash_a: np.ndarray | None = None
    minhash_b: np.ndarray | None = None
    missing_state: int = -1

    def _ensure_minhash_params(self) -> None:
        """Sample MinHash (a,b) once from ``minhash_seed`` (train-only freeze)."""
        if self.minhash_a is not None and self.minhash_b is not None:
            return
        rng = np.random.default_rng(int(self.minhash_seed))
        # a ∈ [1, P-1], b ∈ [0, P-1]
        self.minhash_a = rng.integers(1, MINHASH_PRIME, size=int(self.n_minhash), dtype=np.int64)
        self.minhash_b = rng.integers(0, MINHASH_PRIME, size=int(self.n_minhash), dtype=np.int64)

    def fit(self, examples: list[dict]) -> "SourceDiscretizer":
        """Fit MinHash params + cluster/bin maps on training examples only."""
        self._ensure_minhash_params()
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
        """Freeze train-fitted maps; emit one discrete state per source Q1–Q4."""
        if self.q1_model is None:
            raise RuntimeError("SourceDiscretizer must be fit before transform")
        q1 = int(self.q1_model.predict(self._q1_vector(ex)[None, :])[0])
        q2 = hour_bin(ex["query_time"], self.hour_bins) * 2 + day_type(ex["query_time"])
        q3 = int(self.q3_model.predict(self._q3_vector(ex)[None, :])[0])
        q4v = (ex["stable_user_features"] - self.q4_mean) / self.q4_std
        q4 = int(self.q4_model.predict(q4v[None, :])[0])
        return {"Q1": q1, "Q2": int(q2), "Q3": q3, "Q4": q4}

    def _q1_vector(self, ex: dict) -> np.ndarray:
        """MinHash signature of the visit set → float vector for KMeans.

        步骤 / Steps:
          1) ``build_visit_set``: 全前缀 POI∪cat 去重集合（leakage-safe，无 target）
          2) 对每个元素做稳定 int 哈希，再对 k 个 (a,b) 取
             ``min_x (a*x+b) mod P``，得到长度 ``n_minhash`` 的签名
          3) 除以 P 缩到 [0,1)，供欧氏 KMeans（近似按 Jaccard 近邻分簇）

        北京 smoke: 签名维默认 64；空历史 → 全 1 占位签名。
        """
        self._ensure_minhash_params()
        assert self.minhash_a is not None and self.minhash_b is not None
        return minhash_signature(build_visit_set(ex), self.minhash_a, self.minhash_b)

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
