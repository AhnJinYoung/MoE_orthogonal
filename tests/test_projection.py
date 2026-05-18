import torch

from moe_orthogonal.aggregation import (
    aggregate_standard,
    aggregate_top1_ortho,
    gram_schmidt_ortho,
    project_orthogonal,
)
from moe_orthogonal.metrics import expert_parallel_component_ratio, removed_redundancy_energy


def test_project_orthogonal_removes_anchor_component():
    anchor = torch.tensor([[[1.0, 0.0]]])
    vectors = torch.tensor([[[2.0, 3.0]]])

    ortho = project_orthogonal(vectors, anchor)

    assert torch.allclose(ortho, torch.tensor([[[0.0, 3.0]]]), atol=1e-6)
    assert torch.allclose((ortho * anchor).sum(dim=-1), torch.zeros(1, 1), atol=1e-6)


def test_standard_aggregation_matches_weighted_sum():
    expert_outputs = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    gate_weights = torch.tensor([[[0.25, 0.75]]])

    aggregated = aggregate_standard(expert_outputs, gate_weights)

    assert torch.allclose(aggregated, torch.tensor([[[0.25, 1.5]]]))


def test_top1_ortho_removes_non_top1_parallel_component():
    expert_outputs = torch.tensor([[[[1.0, 0.0], [2.0, 3.0]]]])
    gate_weights = torch.tensor([[[1.0, 1.0]]])

    aggregated = aggregate_top1_ortho(expert_outputs, gate_weights)

    assert torch.allclose(aggregated, torch.tensor([[[1.0, 3.0]]]), atol=1e-6)


def test_gram_schmidt_outputs_pairwise_orthogonal_vectors():
    expert_outputs = torch.tensor([[[[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]]]])

    ortho = gram_schmidt_ortho(expert_outputs)
    dots = ortho @ ortho.transpose(-1, -2)

    assert torch.allclose(dots[..., 0, 1], torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(dots[..., 0, 2], torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(dots[..., 1, 2], torch.zeros(1, 1), atol=1e-6)


def test_metrics_are_finite():
    expert_outputs = torch.randn(2, 3, 2, 4)
    gate_weights = torch.softmax(torch.randn(2, 3, 2), dim=-1)

    pcr = expert_parallel_component_ratio(expert_outputs)
    rre = removed_redundancy_energy(expert_outputs, gate_weights)

    assert torch.isfinite(pcr)
    assert torch.isfinite(rre)
    assert pcr >= 0
    assert rre >= 0
