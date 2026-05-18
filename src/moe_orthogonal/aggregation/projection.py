from __future__ import annotations

import torch


def _fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def project_orthogonal(
    vectors: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Remove the component of vectors parallel to anchor.

    Args:
        vectors: Tensor with hidden dimension last.
        anchor: Tensor broadcastable to vectors with hidden dimension last.
        eps: Numerical guard for zero-norm anchors.
    """
    out_dtype = vectors.dtype
    vectors_f = _fp32(vectors)
    anchor_f = _fp32(anchor)
    numerator = (vectors_f * anchor_f).sum(dim=-1, keepdim=True)
    denominator = (anchor_f * anchor_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    projected = numerator / denominator * anchor_f
    return (vectors_f - projected).to(out_dtype)


def aggregate_standard(
    expert_outputs: torch.Tensor,
    gate_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted top-k expert aggregation.

    Shape convention:
        expert_outputs: [..., top_k, hidden]
        gate_weights: [..., top_k]
    """
    return (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=-2)


def aggregate_top1_ortho(
    expert_outputs: torch.Tensor,
    gate_weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Top1-Ortho aggregation.

    The top-1 expert output is preserved. Other selected expert outputs are
    projected onto the subspace orthogonal to the top-1 output before gated sum.
    """
    if expert_outputs.shape[-2] < 1:
        raise ValueError("expert_outputs must contain at least one selected expert")

    top1 = expert_outputs[..., :1, :]
    if expert_outputs.shape[-2] == 1:
        adjusted = expert_outputs
    else:
        non_top1 = expert_outputs[..., 1:, :]
        non_top1_ortho = project_orthogonal(non_top1, top1, eps=eps)
        adjusted = torch.cat([top1, non_top1_ortho], dim=-2)
    return aggregate_standard(adjusted, gate_weights)


def gram_schmidt_ortho(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Orthogonalize selected expert outputs in top-k order."""
    if expert_outputs.shape[-2] < 1:
        raise ValueError("expert_outputs must contain at least one selected expert")

    components = []
    for idx in range(expert_outputs.shape[-2]):
        vector = expert_outputs[..., idx : idx + 1, :]
        for basis in components:
            vector = project_orthogonal(vector, basis, eps=eps)
        components.append(vector)
    return torch.cat(components, dim=-2)


def aggregate_gs_ortho(
    expert_outputs: torch.Tensor,
    gate_weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return aggregate_standard(gram_schmidt_ortho(expert_outputs, eps=eps), gate_weights)


def aggregate_res_ortho(
    expert_outputs: torch.Tensor,
    gate_weights: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Aggregate normally, then remove the component parallel to residual."""
    aggregated = aggregate_standard(expert_outputs, gate_weights)
    return project_orthogonal(aggregated, residual, eps=eps)
