from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import torch
from torch import nn

from moe_orthogonal.aggregation import AggregationConfig, ExpertAggregator


@dataclass(frozen=True)
class MoeCandidate:
    name: str
    class_name: str
    has_router: bool
    has_experts: bool
    num_experts: int | None


@dataclass(frozen=True)
class AdapterReport:
    model_class: str
    model_type: str | None
    candidates: list[MoeCandidate]


def _get_child(root: nn.Module, path: str) -> nn.Module:
    current = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _set_child(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = _get_child(root, parent_path) if parent_path else root
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


class PatchedMoeBlock(nn.Module):
    """Generic wrapper for common HF sparse MoE blocks.

    This wrapper intentionally targets the common HF pattern:
    hidden_states -> router Linear -> top-k -> per-expert MLPs -> weighted sum.
    If Gemma's remote code differs, the adapter will still inspect the model and
    fail explicitly instead of silently producing invalid results.
    """

    ROUTER_NAMES = ("gate", "router", "router_gate")
    EXPERT_NAMES = ("experts", "mlp_experts")

    def __init__(
        self,
        original: nn.Module,
        aggregation: AggregationConfig,
        *,
        norm_topk_prob: bool | None = None,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.aggregator = ExpertAggregator(aggregation)
        self.norm_topk_prob = norm_topk_prob
        self.router_jitter_noise = router_jitter_noise
        self.last_diagnostics: dict[str, torch.Tensor] = {}

        self.router = self._find_first(self.ROUTER_NAMES)
        self.experts = self._find_first(self.EXPERT_NAMES)
        if self.router is None or self.experts is None:
            raise TypeError(
                f"Cannot patch {type(original).__name__}: expected router and experts attributes"
            )

        self.num_experts = len(self.experts)
        self.top_k = self._infer_top_k(original)

    def _find_first(self, names: Iterable[str]) -> Any:
        for name in names:
            if hasattr(self.original, name):
                return getattr(self.original, name)
        return None

    @staticmethod
    def _infer_top_k(module: nn.Module) -> int:
        for name in ("top_k", "num_experts_per_tok", "num_selected_experts", "topk"):
            value = getattr(module, name, None)
            if value is not None:
                return int(value)
        config = getattr(module, "config", None)
        if config is not None:
            for name in ("num_experts_per_tok", "num_selected_experts", "top_k"):
                value = getattr(config, name, None)
                if value is not None:
                    return int(value)
        raise TypeError(f"Cannot infer top-k for {type(module).__name__}")

    def _expert_forward(self, expert: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return expert(hidden_states)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])

        router_input = hidden_states
        if self.training and self.router_jitter_noise > 0:
            router_input = router_input * torch.empty_like(router_input).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )

        router_logits = self.router(router_input).reshape(-1, self.num_experts)
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob is not False:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat_states.dtype)

        expert_outputs = flat_states.new_zeros(
            flat_states.shape[0],
            self.top_k,
            flat_states.shape[-1],
        )
        for expert_idx, expert in enumerate(self.experts):
            token_pos, rank_pos = torch.where(selected_experts == expert_idx)
            if token_pos.numel() == 0:
                continue
            expert_outputs[token_pos, rank_pos] = self._expert_forward(expert, flat_states[token_pos])

        aggregated = self.aggregator(expert_outputs, routing_weights, residual=flat_states)
        aggregated = aggregated.reshape(original_shape)

        self.last_diagnostics = {
            "expert_outputs": expert_outputs.detach(),
            "gate_weights": routing_weights.detach(),
            "topk_indices": selected_experts.detach(),
            "router_logits": router_logits.detach(),
        }

        return aggregated, router_logits


class HfMoeAdapter:
    """Adapter that inspects and patches HF MoE models."""

    MOE_NAME_HINTS = ("moe", "sparse", "expert")
    ROUTER_NAMES = PatchedMoeBlock.ROUTER_NAMES
    EXPERT_NAMES = PatchedMoeBlock.EXPERT_NAMES

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_modules: dict[str, nn.Module] = {}

    def inspect(self) -> AdapterReport:
        config = getattr(self.model, "config", None)
        candidates: list[MoeCandidate] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            class_name = type(module).__name__
            has_router = any(hasattr(module, attr) for attr in self.ROUTER_NAMES)
            has_experts = any(hasattr(module, attr) for attr in self.EXPERT_NAMES)
            hinted = any(hint in name.lower() or hint in class_name.lower() for hint in self.MOE_NAME_HINTS)
            if hinted or (has_router and has_experts):
                experts = next(
                    (getattr(module, attr) for attr in self.EXPERT_NAMES if hasattr(module, attr)),
                    None,
                )
                num_experts = len(experts) if hasattr(experts, "__len__") else None
                candidates.append(
                    MoeCandidate(
                        name=name,
                        class_name=class_name,
                        has_router=has_router,
                        has_experts=has_experts,
                        num_experts=num_experts,
                    )
                )
        return AdapterReport(
            model_class=type(self.model).__name__,
            model_type=getattr(config, "model_type", None),
            candidates=candidates,
        )

    def patch(self, aggregation: AggregationConfig) -> list[str]:
        patched: list[str] = []
        for candidate in self.inspect().candidates:
            if not (candidate.has_router and candidate.has_experts):
                continue
            module = _get_child(self.model, candidate.name)
            try:
                wrapped = PatchedMoeBlock(module, aggregation)
            except TypeError:
                continue
            self._original_modules[candidate.name] = module
            _set_child(self.model, candidate.name, wrapped)
            patched.append(candidate.name)

        if not patched:
            raise RuntimeError(
                "No compatible MoE block could be patched. Run `moe-ortho inspect` "
                "and add a model-specific adapter for the reported module class."
            )
        return patched

    def unpatch(self) -> None:
        for name, module in self._original_modules.items():
            _set_child(self.model, name, module)
        self._original_modules.clear()

    def set_mode(self, mode: str, eps: float = 1e-6) -> None:
        for module in self.model.modules():
            if isinstance(module, PatchedMoeBlock):
                module.aggregator.config = AggregationConfig(mode=mode, eps=eps)  # type: ignore[arg-type]

    def collect_diagnostics(self) -> list[dict[str, torch.Tensor]]:
        diagnostics = []
        for module in self.model.modules():
            if isinstance(module, PatchedMoeBlock) and module.last_diagnostics:
                diagnostics.append(module.last_diagnostics)
        return diagnostics

    def patch_forward_for_diagnostics(self) -> None:
        """Allow model outputs to expose patched block diagnostics without changing HF APIs."""

        original_forward = self.model.forward
        adapter = self

        def forward_with_diagnostics(model_self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            output = original_forward(*args, **kwargs)
            try:
                setattr(output, "moe_diagnostics", adapter.collect_diagnostics())
            except Exception:
                pass
            return output

        self.model.forward = MethodType(forward_with_diagnostics, self.model)
