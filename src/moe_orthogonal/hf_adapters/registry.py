from __future__ import annotations

from torch import nn

from .base import HfMoeAdapter


def get_adapter(model: nn.Module, model_id: str | None = None) -> HfMoeAdapter:
    """Return an HF MoE adapter.

    Gemma 4 A4B is handled by the generic sparse-MoE adapter first. A dedicated
    adapter can be added here if the remote modeling code uses a non-standard
    forward shape.
    """
    _ = model_id
    return HfMoeAdapter(model)
