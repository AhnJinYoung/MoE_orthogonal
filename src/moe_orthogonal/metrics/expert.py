from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_orthogonal.aggregation.projection import (
    aggregate_standard,
    aggregate_top1_ortho,
    project_orthogonal,
)


@dataclass(frozen=True)
class LoadBalanceStats:
    fractions: torch.Tensor
    variance: torch.Tensor
    max_min_ratio: torch.Tensor


def expert_parallel_component_ratio(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean PCR for selected non-top1 expert outputs.

    Args:
        expert_outputs: [..., top_k, hidden]
    """
    if expert_outputs.shape[-2] <= 1:
        return expert_outputs.new_tensor(0.0)

    top1 = expert_outputs[..., :1, :]
    non_top1 = expert_outputs[..., 1:, :]
    ortho = project_orthogonal(non_top1, top1, eps=eps)
    parallel = non_top1 - ortho
    ratio = parallel.norm(dim=-1) / non_top1.norm(dim=-1).clamp_min(eps)
    return ratio.mean()


def removed_redundancy_energy(
    expert_outputs: torch.Tensor,
    gate_weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Relative energy removed by Top1-Ortho compared with standard aggregation."""
    standard = aggregate_standard(expert_outputs, gate_weights)
    ortho = aggregate_top1_ortho(expert_outputs, gate_weights, eps=eps)
    removed = standard - ortho
    return (removed.norm(dim=-1) / standard.norm(dim=-1).clamp_min(eps)).mean()


def cosine_similarity(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean pairwise cosine similarity among selected experts."""
    top_k = expert_outputs.shape[-2]
    if top_k <= 1:
        return expert_outputs.new_tensor(0.0)

    normalized = expert_outputs / expert_outputs.norm(dim=-1, keepdim=True).clamp_min(eps)
    sims = normalized @ normalized.transpose(-1, -2)
    mask = ~torch.eye(top_k, dtype=torch.bool, device=expert_outputs.device)
    return sims[..., mask].mean()


def load_balance_stats(
    topk_indices: torch.Tensor,
    num_experts: int,
    eps: float = 1e-6,
) -> LoadBalanceStats:
    """Compute expert load fractions and simple imbalance summaries."""
    flat = topk_indices.reshape(-1)
    counts = torch.bincount(flat, minlength=num_experts).to(dtype=torch.float32)
    fractions = counts / counts.sum().clamp_min(eps)
    nonzero_min = fractions[fractions > 0].min() if (fractions > 0).any() else fractions.new_tensor(eps)
    return LoadBalanceStats(
        fractions=fractions,
        variance=fractions.var(unbiased=False),
        max_min_ratio=fractions.max() / nonzero_min.clamp_min(eps),
    )
