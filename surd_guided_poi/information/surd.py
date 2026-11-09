"""把 Specific MI 按官方 SURD 规则拆成 R / U / S（per-POI + 全局聚合）。

算法来源（适配为按 POI 状态输出）::
  Martínez-Sánchez et al., Nat Commun 15, 9296 (2024)
  https://github.com/ALD-Lab/SURD  (``surd.assign``)

相对官方包的两点适配
--------------------
1. 循环内**不**乘 ``P(Y=y)``，每个目标 POI 各自保留 R/U/S。
2. 全局 SURD 事后用 ``sum_y P(y) * specific_term(y)`` 聚合（见规格）。

模块调用关系（谁喂谁 / 谁调谁）
--------------------------------
上游（本文件外）::

    collect_subset_probs → estimate_specific_mi → enforce_subset_monotonicity
                                                         │
                                                         ▼
                                              assign_surd_from_specific_mi  ← 本文件主入口
                                                         │
                         ┌───────────────────────────────┼────────────────┐
                         ▼                               ▼                ▼
              build_poi_specific_weights      aggregate_global_surd   balanced_surd_summary
              （weights.py，训 L_U/L_S）         （按 prior 加权）        （等权平均，诊断用）

本文件内部::

    assign_surd_from_specific_mi
        ├─ _subset_tuples          # 生成 1-based 子集索引序
        ├─ subset_key              # 子集名 → 'Q1' / 'Q1_Q2' …
        └─ _order                  # 按 MI 升序（近平局按子集大小）排序

    _name_tuple_to_index_tuple     # 名称元组 → 1-based 索引（工具；当前主路径未用）

入口调用方：``AlternatingSURDTrainer.refresh_surd``（Phase A）、``tests/test_surd.py``。

观测语言：这些是**预测信息结构**，不是因果效应。
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..models.subset_estimators import SOURCE_NAMES, SUBSET_NAMES, subset_key


def _order(values: np.ndarray, sizes: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """按 specific MI 升序得到稳定排列下标；近平局时小子集优先。

    角色：``assign_surd_from_specific_mi`` 内部排序工具（改编自 ALD-Lab/SURD
    ``core._order``）。子集与其超集 MI 常相等时，必须让子集先入账，否则会被
    记成 synergy。

    被谁调用：仅 ``assign_surd_from_specific_mi``（每个 POI y 排两次）。
    """
    values = np.asarray(values, dtype=float)
    sizes = np.asarray(sizes)
    if values.size == 0:
        return np.arange(0)
    tol = rtol * max(float(np.abs(values).max()), 1.0)
    i0 = np.argsort(values, kind="stable")
    group = np.empty(values.size, dtype=np.int64)
    group[i0] = np.concatenate(([0], np.cumsum(np.diff(values[i0]) > tol)))
    return np.lexsort((np.arange(values.size), sizes, group))


def _subset_tuples(source_names: Sequence[str] = SOURCE_NAMES) -> List[Tuple[int, ...]]:
    """生成与官方 SURD 一致的 1-based 子集索引列表。

    例（3 源）::

        [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]

    角色：给 ``assign_surd_from_specific_mi`` 定 comb 顺序，与
    ``SUBSET_NAMES`` / ``specific_mi`` 的 ``subset_key`` 对齐。
    被谁调用：仅 ``assign_surd_from_specific_mi``。
    """
    n = len(source_names)
    inds = range(1, n + 1)
    return [c for k in inds for c in combinations(inds, k)]


def _name_tuple_to_index_tuple(name_tuple: Tuple[str, ...], source_names: Sequence[str]) -> Tuple[int, ...]:
    """``('Q1','Q2')`` → ``(1, 2)``（1-based）。

    角色：名称子集 ↔ 官方索引子集的转换工具。
    当前主训练路径未直接调用（``assign`` 走 ``_subset_tuples`` + ``SUBSET_NAMES``）。
    """
    mapping = {n: i + 1 for i, n in enumerate(source_names)}
    return tuple(mapping[n] for n in name_tuple)


def assign_surd_from_specific_mi(
    specific_mi: Mapping[str, np.ndarray],
    source_names: Sequence[str] = SOURCE_NAMES,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """对每个目标 POI 独立跑官方 SURD assign，得到 per-POI 的 U / R / S。

    角色（本模块主入口）
    --------------------
    把 ``estimate_specific_mi``（宜经 ``enforce_subset_monotonicity``）给出的
    ``\\tilde{i}(y; Q_A)`` 拆成 unique / redundant / synergistic。

    调用关系
    --------
    - **被谁调用**：``AlternatingSURDTrainer.refresh_surd``；测试 ``test_surd.py``。
    - **吃什么**：``specific_mi[subset_key]``，形状各 ``[K]``（K=num_pois）。
    - **内部调用**：``_subset_tuples``、``subset_key``、``_order``。
    - **结果被谁用**：
        - ``build_poi_specific_weights`` → 训练 ``L_U`` / ``L_S`` 查表权重
        - ``aggregate_global_surd`` / ``balanced_surd_summary`` → 全局/诊断汇总

    返回
    ----
    ``poi_surd[poi_id] = {
        "unique": {"Q1": …},           # 单源 I_R[(i,)]
        "redundant": {"Q1_Q2": …},     # 多源冗余
        "synergistic": {"Q1_Q2": …},   # I_S
    }``
    """
    source_names = tuple(source_names)
    nvars = len(source_names)
    index_combs = _subset_tuples(source_names)
    # Align specific MI arrays to index_combs order.
    name_combs = SUBSET_NAMES if source_names == SOURCE_NAMES else [
        tuple(source_names[i - 1] for i in c) for c in index_combs
    ]
    mats = []
    for names in name_combs:
        key = subset_key(names)
        if key not in specific_mi:
            raise KeyError(f"Missing specific MI for subset {key}")
        mats.append(np.asarray(specific_mi[key], dtype=np.float64))
    stacked = np.stack(mats, axis=0)  # [num_combs, K]
    K = stacked.shape[1]

    poi_surd: Dict[int, Dict[str, Dict[str, float]]] = {}
    inds = list(range(1, nvars + 1))

    for y in range(K):
        I1 = stacked[:, y].copy()
        sizes = np.array([len(c) for c in index_combs])
        i1 = _order(I1, sizes)
        lab = [index_combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])
        I1 = I1[i1]

        # Clip supersets that fall below the max of size-l subsets (official rule).
        if lens.max() >= 2:
            for l in range(1, int(lens.max())):
                inds_l2 = np.where(lens == l + 1)[0]
                if inds_l2.size == 0 or not np.any(lens == l):
                    continue
                Il1max = I1[lens == l].max()
                weak = inds_l2[I1[inds_l2] < Il1max]
                I1[weak] = 0.0

        i1 = _order(I1, lens)
        lab = [lab[i_] for i_ in i1]
        Di = np.diff(I1[i1], prepend=0.0)
        red_vars = list(inds)

        unique: Dict[str, float] = {n: 0.0 for n in source_names}
        redundant: Dict[str, float] = {}
        synergistic: Dict[str, float] = {}
        for i_, ll in enumerate(lab):
            info = float(Di[i_])  # POI-specific: do NOT multiply by P(y) here
            if len(ll) == 1:
                # Book to current redundancy pool; singleton pool => unique.
                key = tuple(red_vars)
                if len(key) == 1:
                    unique[source_names[key[0] - 1]] += info
                else:
                    rname = subset_key(tuple(source_names[j - 1] for j in key))
                    redundant[rname] = redundant.get(rname, 0.0) + info
                red_vars.remove(ll[0])
            else:
                sname = subset_key(tuple(source_names[j - 1] for j in ll))
                synergistic[sname] = synergistic.get(sname, 0.0) + info

        poi_surd[y] = {
            "unique": unique,
            "redundant": redundant,
            "synergistic": synergistic,
        }
    return poi_surd


def aggregate_global_surd(
    poi_surd: Mapping[int, Dict[str, Dict[str, float]]],
    poi_prior: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """按 ``P(y)`` 加权，把 per-POI SURD 聚成全局一份。

    角色：``sum_y P(y) * specific_term(y)``（规格要求的 population MI）。

    调用关系
    --------
    - **被谁调用**：``refresh_surd``（日志 / 落盘）；``test_surd.py``。
    - **吃什么**：``assign_surd_from_specific_mi`` 的 ``poi_surd`` + ``poi_prior[K]``。
    - **不调用**本文件其他函数；与 ``balanced_surd_summary`` 并列、权重不同。
    """
    prior = np.asarray(poi_prior, dtype=np.float64)
    unique = {n: 0.0 for n in SOURCE_NAMES}
    redundant: Dict[str, float] = {}
    synergistic: Dict[str, float] = {}
    for y, terms in poi_surd.items():
        w = float(prior[y]) if y < len(prior) else 0.0
        for k, v in terms["unique"].items():
            unique[k] = unique.get(k, 0.0) + w * v
        for k, v in terms["redundant"].items():
            redundant[k] = redundant.get(k, 0.0) + w * v
        for k, v in terms["synergistic"].items():
            synergistic[k] = synergistic.get(k, 0.0) + w * v
    return {"unique": unique, "redundant": redundant, "synergistic": synergistic}


def balanced_surd_summary(
    poi_surd: Mapping[int, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """对 ``poi_surd`` 里出现的 POI **等权**平均（不是 population MI）。

    角色：诊断 / 对照用的 macro 汇总；训练主路径当前走 ``aggregate_global_surd``。

    调用关系
    --------
    - **吃什么**：同 ``aggregate_global_surd`` 的 ``poi_surd``（来自 ``assign_…``）。
    - **与** ``aggregate_global_surd`` **并列**（一个按 prior，一个按 1/|POIs|）。
    """
    if not poi_surd:
        return {"unique": {}, "redundant": {}, "synergistic": {}}
    unique = {n: 0.0 for n in SOURCE_NAMES}
    redundant: Dict[str, float] = {}
    synergistic: Dict[str, float] = {}
    inv = 1.0 / len(poi_surd)
    for terms in poi_surd.values():
        for k, v in terms["unique"].items():
            unique[k] = unique.get(k, 0.0) + inv * v
        for k, v in terms["redundant"].items():
            redundant[k] = redundant.get(k, 0.0) + inv * v
        for k, v in terms["synergistic"].items():
            synergistic[k] = synergistic.get(k, 0.0) + inv * v
    return {"unique": unique, "redundant": redundant, "synergistic": synergistic}
