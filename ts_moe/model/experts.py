"""Unique, redundancy, and synergy experts.

Unique experts are two-layer GELU FFNs on one candidate-aware vector
(spec §5.1). Redundancy experts build a shared code from the product,
absolute difference, and sum, then reconstruct both inputs (spec §5.2).
Synergy experts use a residual Hadamard product (spec §5.3). The product
is an inductive bias; it is not itself the SURD synergy atom.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
    """``W2 GELU(W1 z)`` with hidden width ``2d``."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RedundancyExpert(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dec_i = nn.Linear(d_model, d_model)
        self.dec_j = nn.Linear(d_model, d_model)
        self.ffn = ExpertFFN(d_model)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(torch.cat([z_i * z_j, (z_i - z_j).abs(), z_i + z_j], dim=-1))
        hat_i = self.dec_i(shared)
        hat_j = self.dec_j(shared)
        recon = (z_i - hat_i).pow(2).mean() + (z_j - hat_j).pow(2).mean()
        return self.ffn(shared), recon


class SynergyExpert(nn.Module):
    def __init__(self, d_model: int, members: tuple[int, ...]):
        super().__init__()
        self.members = members
        self.mix = nn.Linear(d_model, d_model, bias=False)
        self.residual = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in members)
        self.ffn = ExpertFFN(d_model)

    def forward(self, zs: list[torch.Tensor], projected: list[torch.Tensor]) -> torch.Tensor:
        product = projected[self.members[0]]
        for index in self.members[1:]:
            product = product * projected[index]
        residual = self.residual[0](zs[self.members[0]])
        for linear, index in zip(self.residual[1:], self.members[1:]):
            residual = residual + linear(zs[index])
        # Spec §5.3: c_A = W_A (⊙_i W_i z_i) + Σ_i V_{A,i} z_i
        return self.ffn(self.mix(product) + residual)
