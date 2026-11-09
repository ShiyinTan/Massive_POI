"""Build detached POI-specific U/S guidance weights from SURD terms."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..models.subset_estimators import SOURCE_NAMES, SUBSET_NAMES, subset_key


def _normalize_nonneg(values: Mapping[str, float], eps: float) -> Dict[str, float]:
    """非负项归一化成「权重和为 1」；全 0 则全 0。"""
    clipped = {k: max(float(v), 0.0) for k, v in values.items()}
    total = sum(clipped.values())
    if total <= eps:
        return {k: 0.0 for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def build_poi_specific_weights(
    poi_surd: Mapping[int, Dict[str, Dict[str, float]]],
    poi_sample_counts: np.ndarray,
    poi_prior: Optional[np.ndarray] = None,
    min_poi_samples: int = 3,
    shrink_kappa: float = 10.0,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """把每个 POI 的 U/S 量变成训练用权重；样本太少的 POI 往全局权重靠。"""
    counts = np.asarray(poi_sample_counts, dtype=np.int64)
    K = len(counts)

    # 先算一份「全局平均」权重，给长尾 POI 当 fallback
    global_u = {n: 0.0 for n in SOURCE_NAMES}
    # 多源键同 subset_key：'Q1_Q2' / 'Q1_Q3' / 'Q2_Q3' / 'Q1_Q2_Q3'
    global_s: Dict[str, float] = {
        subset_key(s): 0.0 for s in SUBSET_NAMES if len(s) >= 2
    }
    total_w = 0.0
    for y, terms in poi_surd.items():
        w = float(counts[y]) if y < K else 0.0
        total_w += w
        for n, v in terms["unique"].items():
            global_u[n] += w * v
        for n, v in terms["synergistic"].items():
            global_s[n] = global_s.get(n, 0.0) + w * v
    if total_w > 0:
        global_u = {k: v / total_w for k, v in global_u.items()}
        global_s = {k: v / total_w for k, v in global_s.items()}
    global_u_w = _normalize_nonneg(global_u, eps)
    global_s_w = _normalize_nonneg(global_s, eps)

    unique_weights: Dict[int, Dict[str, float]] = {}
    synergy_weights: Dict[int, Dict[str, float]] = {}
    signed: Dict[int, Dict[str, Dict[str, float]]] = {}
    fallback: Dict[int, bool] = {}

    for y in range(K):
        terms = poi_surd.get(
            y,
            {
                "unique": {n: 0.0 for n in SOURCE_NAMES},
                "redundant": {},
                "synergistic": {subset_key(s): 0.0 for s in SUBSET_NAMES if len(s) >= 2},
            },
        )
        signed[y] = {
            "unique": dict(terms["unique"]),
            "redundant": dict(terms.get("redundant", {})),
            "synergistic": dict(terms.get("synergistic", {})),
        }
        u_raw = _normalize_nonneg(terms["unique"], eps)
        s_raw = _normalize_nonneg(terms.get("synergistic", {}), eps)
        # Ensure all multi-source keys exist.
        for s in SUBSET_NAMES:
            if len(s) >= 2:
                s_raw.setdefault(subset_key(s), 0.0)

        n_y = int(counts[y]) if y < K else 0  # 这里必须用整数 N_y，不能用 prior
        use_fallback = n_y < min_poi_samples  # 样本太少：别信这个 POI 自己的 SURD
        fallback[y] = use_fallback
        if use_fallback:
            rho = 0.0  # 全用全局
        else:
            rho = float(n_y) / float(n_y + shrink_kappa)  # 样本越多，越信自己

        unique_weights[y] = {
            n: rho * u_raw.get(n, 0.0) + (1.0 - rho) * global_u_w.get(n, 0.0) for n in SOURCE_NAMES
        }
        synergy_weights[y] = {
            k: rho * s_raw.get(k, 0.0) + (1.0 - rho) * global_s_w.get(k, 0.0) for k in global_s_w
        }
        unique_weights[y] = _normalize_nonneg(unique_weights[y], eps)
        synergy_weights[y] = _normalize_nonneg(synergy_weights[y], eps)

    fallback_frac = float(sum(1 for y in range(K) if fallback[y])) / max(K, 1)
    return {
        "unique_weights": unique_weights,
        "synergy_weights": synergy_weights,
        "signed": signed,
        "fallback": fallback,
        "fallback_fraction": fallback_frac,
        "sample_counts": counts,
        "global_unique_weights": global_u_w,
        "global_synergy_weights": global_s_w,
    }


def ema_update_weights(
    old: Optional[Dict[str, Any]],
    new: Dict[str, Any],
    beta: float = 0.5,
) -> Dict[str, Any]:
    """新旧 SURD 权重做指数滑动平均，避免每隔几轮权重跳太猛。"""
    if old is None or beta <= 0:
        return new
    beta = float(beta)
    out = dict(new)
    u: Dict[int, Dict[str, float]] = {}
    s: Dict[int, Dict[str, float]] = {}
    for y, nw in new["unique_weights"].items():
        ow = old.get("unique_weights", {}).get(y, nw)
        u[y] = {k: beta * ow.get(k, 0.0) + (1.0 - beta) * nw.get(k, 0.0) for k in nw}
        u[y] = _normalize_nonneg(u[y], 1e-12)
    for y, nw in new["synergy_weights"].items():
        ow = old.get("synergy_weights", {}).get(y, nw)
        s[y] = {k: beta * ow.get(k, 0.0) + (1.0 - beta) * nw.get(k, 0.0) for k in nw}
        s[y] = _normalize_nonneg(s[y], 1e-12)
    out["unique_weights"] = u
    out["synergy_weights"] = s
    return out


def weights_to_tensors(
    weights: Dict[str, Any],
    num_pois: int,
    device,
) -> Dict[str, "torch.Tensor"]:
    """字典权重 → 张量 [K, ...]，方便按 target_poi_id 查表；.detach() 保证不反传。"""
    import torch

    u_keys = list(SOURCE_NAMES)
    s_keys = [subset_key(s) for s in SUBSET_NAMES if len(s) >= 2]
    u = torch.zeros(num_pois, len(u_keys), dtype=torch.float32, device=device)
    s = torch.zeros(num_pois, len(s_keys), dtype=torch.float32, device=device)
    for y in range(num_pois):
        uw = weights["unique_weights"].get(y, {})
        sw = weights["synergy_weights"].get(y, {})
        for i, k in enumerate(u_keys):
            u[y, i] = float(uw.get(k, 0.0))
        for i, k in enumerate(s_keys):
            s[y, i] = float(sw.get(k, 0.0))
    return {
        "unique": u.detach(),  # SURD 估计本身不参与梯度
        "synergy": s.detach(),
        "unique_keys": u_keys,
        "synergy_keys": s_keys,
    }