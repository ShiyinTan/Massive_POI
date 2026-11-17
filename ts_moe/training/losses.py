"""Spec §8 objective: next-POI CE + routing KL + SURD-matched load balance.

Sampled softmax uses the provided candidate set as the denominator. The
positive candidate is column 0. Routing KL is ``KL(stopgrad(π*) || π)`` on
the dense softmax, before top-k renormalization. Each candidate contributes
its own prior.

``L_route`` is applied only for ``routing_mode="surd"``. Plain MoE keeps the
same experts but balances toward a uniform load instead of the SURD prior.
Random and fixed routing do not train a gate, so both auxiliary terms are 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_softmax_ce(scores: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def routing_kl(pi_star: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    target = pi_star.detach().clamp_min(1e-8)
    target = target / target.sum(dim=-1, keepdim=True)
    pred = pi.clamp_min(1e-8)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    return (target * (target.log() - pred.log())).sum(dim=-1).mean()


def load_balance(pi: torch.Tensor, pi_star: torch.Tensor, mode: str) -> torch.Tensor:
    """``Σ_e (f_e - f*_e)^2`` with ``f`` averaged over the scored candidates."""
    if mode not in {"surd", "plain"}:
        return pi.new_zeros(())
    frequency = pi.mean(dim=(0, 1))
    if mode == "surd":
        target = pi_star.detach().mean(dim=(0, 1))
    else:
        target = torch.full_like(frequency, 1.0 / frequency.numel())
    return ((frequency - target) ** 2).sum()


def total_loss(
    scores: torch.Tensor,
    pi: torch.Tensor,
    pi_star: torch.Tensor,
    shared_loss: torch.Tensor,
    *,
    mode: str,
    lambda_route: float,
    lambda_balance: float,
    lambda_shared: float,
) -> dict[str, torch.Tensor]:
    ce = sampled_softmax_ce(scores)
    if mode == "surd":
        route = routing_kl(pi_star, pi)
    else:
        route = scores.new_zeros(())
    balance = load_balance(pi, pi_star, mode)
    loss = ce + float(lambda_route) * route + float(lambda_balance) * balance + float(lambda_shared) * shared_loss
    return {"loss": loss, "ce": ce.detach(), "route": route.detach(), "balance": balance.detach(), "shared": shared_loss.detach()}
