"""Probability heads for every non-empty subset of the 3 information sources."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

SOURCE_NAMES = ("Q1", "Q2", "Q3")


def all_subsets(names: Sequence[str] = SOURCE_NAMES) -> List[Tuple[str, ...]]:
    """Non-empty subsets ordered by size then lexicographically (SURD key order)."""
    out: List[Tuple[str, ...]] = []
    n = len(names)
    for k in range(1, n + 1):
        out.extend(combinations(names, k))
    return out


SUBSET_NAMES: List[Tuple[str, ...]] = all_subsets()


def subset_key(subset: Iterable[str]) -> str:
    """把子集元组编成字典键字符串。

    例::

        ("Q1",)           → "Q1"
        ("Q1", "Q2")      → "Q1_Q2"
        ("Q1", "Q2", "Q3")→ "Q1_Q2_Q3"

    由有序元组用 ``"_"`` 拼接而成（顺序与 ``SOURCE_NAMES`` / ``all_subsets`` 一致）。
    用作：子集头 ``ModuleDict`` 名、specific-MI / SURD / 权重表的键、辅助损失里查 logits。
    """
    return "_".join(subset)


class SubsetHead(nn.Module):
    """单个子集头：拼接后的源向量 → POI logits。

    例：子集 {Q1,Q2} → 输入 [B, 2*hidden] → 输出 [B, K]。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_pois: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, in_dim] → [B, K]


class SubsetEstimators(nn.Module):
    """3 个源 → 7 个非空子集概率头（只看自己被授权的 h_i）。

    输入 ``source_hs = [h1, h2, h3]``，每个 ``[B, hidden]``。

    输出 dict（键 → logits）::

        "Q1"       → [B, K]
        "Q2"       → [B, K]
        "Q3"       → [B, K]
        "Q1_Q2"    → [B, K]   # 输入是 concat(h1,h2) = [B, 2*hidden]
        "Q1_Q3"    → [B, K]
        "Q2_Q3"    → [B, K]
        "Q1_Q2_Q3" → [B, K]   # concat 三维 → [B, 3*hidden]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_pois: int,
        subset_hidden_dim: int = 64,
        dropout: float = 0.1,
        source_names: Sequence[str] = SOURCE_NAMES,
    ):
        super().__init__()
        self.source_names = tuple(source_names)
        self.subsets = all_subsets(self.source_names)
        self.heads = nn.ModuleDict()
        name_to_idx = {n: i for i, n in enumerate(self.source_names)}
        self._subset_indices: Dict[str, List[int]] = {}
        for subset in self.subsets:
            key = subset_key(subset)  # 如 "Q1" / "Q1_Q2"；作 ModuleDict 键
            in_dim = hidden_dim * len(subset)
            self.heads[key] = SubsetHead(in_dim, subset_hidden_dim, num_pois, dropout)
            self._subset_indices[key] = [name_to_idx[s] for s in subset]

    def forward(self, source_hs: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(source_hs) != len(self.source_names):
            raise ValueError("source_hs length must match number of sources")
        out: Dict[str, torch.Tensor] = {}
        for key, idxs in self._subset_indices.items():
            parts = [source_hs[i] for i in idxs]
            x = torch.cat(parts, dim=-1)  # [B, |A|*hidden]
            out[key] = self.heads[key](x)  # [B, K]
        return out

    def single_source_logits(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """只取单源头：Q1/Q2/Q3 → 各 [B, K]（给 L_U）。"""
        return {k: logits[k] for k in self.source_names}

    def multi_source_logits(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """只取多源组合头：Q1_Q2 等 → 各 [B, K]（给 L_S）。"""
        return {k: v for k, v in logits.items() if "_" in k}