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
    """
    L_U = -1/N sum_n sum_i w_i^U(y_n) log P_i(y_n | Q_i)

    unique_weight_table: [num_pois, num_sources] detached.
    """
    w = unique_weight_table[target]  # [B, m]
    loss = target.new_zeros(())
    for i, name in enumerate(source_order):
        logits = single_source_logits[name]
        nll = F.cross_entropy(logits, target, reduction="none")
        loss = loss + (w[:, i] * nll).mean()
    return loss


def synergy_auxiliary_loss(
    multi_source_logits: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    synergy_weight_table: torch.Tensor,
    subset_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    """
    L_S = -1/N sum_n sum_{|A|>=2} w_A^S(y_n) log P_A(y_n | Q_A)
    """
    if subset_keys is None:
        subset_keys = [subset_key(s) for s in SUBSET_NAMES if len(s) >= 2]
    w = synergy_weight_table[target]  # [B, num_multi]
    loss = target.new_zeros(())
    for i, key in enumerate(subset_keys):
        logits = multi_source_logits[key]
        nll = F.cross_entropy(logits, target, reduction="none")
        loss = loss + (w[:, i] * nll).mean()
    return loss