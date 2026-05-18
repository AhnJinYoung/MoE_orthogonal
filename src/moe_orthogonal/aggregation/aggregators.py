from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .projection import (
    aggregate_gs_ortho,
    aggregate_res_ortho,
    aggregate_standard,
    aggregate_top1_ortho,
)

AggregationMode = Literal["standard", "top1_ortho", "gs_ortho", "res_ortho"]


@dataclass(frozen=True)
class AggregationConfig:
    mode: AggregationMode = "standard"
    eps: float = 1e-6


class ExpertAggregator(nn.Module):
    """Small nn.Module wrapper around aggregation functions.

    Keeping this as a module makes it easy for HF adapters to expose and swap
    aggregation behavior without changing router or expert weights.
    """

    def __init__(self, config: AggregationConfig | None = None):
        super().__init__()
        self.config = config or AggregationConfig()

    @property
    def mode(self) -> AggregationMode:
        return self.config.mode

    def forward(
        self,
        expert_outputs: torch.Tensor,
        gate_weights: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.mode == "standard":
            return aggregate_standard(expert_outputs, gate_weights)
        if self.config.mode == "top1_ortho":
            return aggregate_top1_ortho(expert_outputs, gate_weights, eps=self.config.eps)
        if self.config.mode == "gs_ortho":
            return aggregate_gs_ortho(expert_outputs, gate_weights, eps=self.config.eps)
        if self.config.mode == "res_ortho":
            if residual is None:
                raise ValueError("res_ortho aggregation requires residual")
            return aggregate_res_ortho(expert_outputs, gate_weights, residual, eps=self.config.eps)
        raise ValueError(f"Unknown aggregation mode: {self.config.mode}")
