"""POI-specific specific mutual information from subset probability estimates.

\tilde{i}(y; Q_A) = E_{q_A | Y=y}[ log P(Y=y | Q_A) / P(Y=y) ]

Estimated on an independent estimation/calibration split (not the same
samples used to fit the subset heads in the current Phase A round).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.subset_estimators import SUBSET_NAMES, subset_key


def _log_base(units: str):
    if units == "nats":
        return np.log, torch.log
    if units == "bits":
        return np.log2, lambda x: torch.log2(x)
    raise ValueError(f"Unknown units: {units}")


def estimate_specific_mi(
    subset_probs: Mapping[str, np.ndarray],
    labels: np.ndarray,
    poi_prior: np.ndarray,
    eps: float = 1e-12,
    units: str = "nats",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """在估计 holdout 上算每个 POI 的 Specific MI ``\\tilde{i}(y; Q_A)``。

    公式（对子集 A、标签 y）::

        \\tilde{i}(y; Q_A) = E[ log P_A(Y=y | Q_A) − log P(Y=y) | Y=y ]

    即：只在「真实下一站是 y」的估计样本上，对 log 概率比取平均。

    输入
    ----
    subset_probs:
        ``dict[subset_key → ndarray]``。键如 ``"Q1"`` / ``"Q1_Q2"`` / ``"Q1_Q2_Q3"``。
        每个值形状 ``[N, K]``：估计集上子集头给出的 ``P_A(Y | Q_A)``（行和≈1）。
        ``N``=估计样本数，``K``=POI 数。
    labels:
        形状 ``[N]`` 的 int：每条估计样本的真实 ``target_poi_id``（∈ ``0..K-1``）。
    poi_prior:
        形状 ``[K]``：训练先验 ``P(Y=y)=N_y/N``（来自 model-train，**不是**均匀 1/K）。
        用作分母；与 ``poi_sample_counts``（原始次数）不同。
    eps / units:
        概率下截断；``"nats"``（``log``）或 ``"bits"``（``log2``）。

    输出
    ----
    specific:
        ``dict[subset_key → ndarray[K]]``。``specific[key][y]`` = 该子集下 POI y 的
        Specific MI；估计集上从未出现的 y 保持 ``0``。
    counts:
        形状 ``[K]`` 的 int：估计集上各 y 出现次数（``bincount(labels)``），供后续
        权重 shrink / 诊断，不是 prior。

    注意：应在与拟合子集头不同的样本上调用（见 ``carve_estimator_holdout``）。
    """
    np_log, _ = _log_base(units)
    labels = np.asarray(labels, dtype=np.int64)
    prior = np.clip(np.asarray(poi_prior, dtype=np.float64), eps, None)
    K = prior.shape[0]
    N = labels.shape[0]
    counts = np.bincount(labels, minlength=K).astype(np.int64)

    specific: Dict[str, np.ndarray] = {}
    for key, probs in subset_probs.items():
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (N, K):
            raise ValueError(f"{key}: expected probs shape {(N, K)}, got {p.shape}")
        p = np.clip(p, eps, 1.0)
        # 只看真实标签那一格的概率：log P_A(y|Q_A) − log P(y)
        p_true = p[np.arange(N), labels]
        log_ratio = np_log(p_true) - np_log(prior[labels])
        mi = np.zeros(K, dtype=np.float64)
        for y in np.unique(labels):
            mask = labels == y
            mi[y] = float(log_ratio[mask].mean())  # 对「下一站是 y」的样本取平均
        # 估计集上没出现过的 POI 保持 0
        specific[key] = mi
    return specific, counts


def enforce_subset_monotonicity(
    specific_mi: Mapping[str, np.ndarray],
    subsets: Sequence[Tuple[str, ...]] = SUBSET_NAMES,
) -> Dict[str, np.ndarray]:
    """
    SURD requires i(y; Q_A) not to decrease when A grows.

    Follows the official ``surd.neural`` monotone projection: replace each
    value by the max over subsets of A (empty subset treated as 0).
    """
    out: Dict[str, np.ndarray] = {}
    for subset in sorted(subsets, key=len):
        key = subset_key(subset)  # 'Q1' / 'Q1_Q2' / ...；与 specific_mi 字典对齐
        if key not in specific_mi:
            continue
        v = np.maximum(np.asarray(specific_mi[key], dtype=np.float64), 0.0)
        for smaller, smaller_v in out.items():
            smaller_set = set(smaller.split("_"))  # 从键反推子集名
            if smaller_set < set(subset):
                v = np.maximum(v, smaller_v)
        out[key] = v
    # Preserve original key order for callers.
    return {subset_key(s): out[subset_key(s)] for s in subsets if subset_key(s) in out}


@torch.no_grad()
def collect_subset_probs(
    model,
    loader,
    device: torch.device,
    temperature: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run subset heads over a loader; return probs per subset and labels."""
    model.eval()
    all_logits: Dict[str, list] = {}
    labels: list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, compute_subsets=True)
        subset_logits: Dict[str, torch.Tensor] = out["subset_logits"]  # type: ignore
        for k, logits in subset_logits.items():
            all_logits.setdefault(k, []).append(logits.detach().cpu())
        labels.append(batch.target_poi_id.detach().cpu())
    label_arr = torch.cat(labels, dim=0).numpy()
    probs: Dict[str, np.ndarray] = {}
    for k, chunks in all_logits.items():
        logits = torch.cat(chunks, dim=0)
        if temperature and k in temperature:
            logits = logits / float(temperature[k])
        probs[k] = F.softmax(logits, dim=-1).numpy().astype(np.float64)
    return probs, label_arr