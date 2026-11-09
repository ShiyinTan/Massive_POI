"""POI-specific SURD-weighted auxiliary CE losses (U and S).

Important: the CE terms are *not* pure unique/synergy information.
SURD weights only guide how strongly each branch is supervised.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..models.subset_estimators import SOURCE_NAMES, SUBSET_NAMES, subset_key


def unique_auxiliary_loss(
    single_source_logits: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    unique_weight_table: torch.Tensor,
    source_order: Sequence[str] = SOURCE_NAMES,
) -> torch.Tensor:
    """按真实下一站 y 查 unique 权重，加权各单源头的交叉熵（权重不反传）。"""
    w = unique_weight_table[target]  # [B, m]：这一批样本各自该强调哪些单源
    loss = target.new_zeros(())
    for i, name in enumerate(source_order):
        logits = single_source_logits[name]
        nll = F.cross_entropy(logits, target, reduction="none")
        loss = loss + (w[:, i] * nll).mean()  # 权重大 → 更用力训这个源
    return loss


def synergy_auxiliary_loss(
    multi_source_logits: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    synergy_weight_table: torch.Tensor,
    subset_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    """按 y 查 synergy 权重，加权多源组合头 CE（同样不是「纯 synergy 信息量」）。"""
    if subset_keys is None:
        # 与多源头 / synergy 权重列顺序一致：'Q1_Q2', ...
        subset_keys = [subset_key(s) for s in SUBSET_NAMES if len(s) >= 2]
    w = synergy_weight_table[target]  # [B, num_multi]
    loss = target.new_zeros(())
    for i, key in enumerate(subset_keys):
        logits = multi_source_logits[key]
        nll = F.cross_entropy(logits, target, reduction="none")
        loss = loss + (w[:, i] * nll).mean()
    return loss