import torch
from torch import nn

from moe_orthogonal.aggregation import AggregationConfig
from moe_orthogonal.hf_adapters import HfMoeAdapter


class DummyExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states):
        return hidden_states * self.scale


class DummyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 2
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = nn.ModuleList([DummyExpert(1.0), DummyExpert(2.0), DummyExpert(3.0)])
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = self.gate(flat)
        weights = torch.softmax(router_logits, dim=-1)
        weights, selected = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_pos, rank_pos = torch.where(selected == expert_idx)
            if token_pos.numel() == 0:
                continue
            out[token_pos] += weights[token_pos, rank_pos].unsqueeze(-1) * expert(flat[token_pos])
        return out.reshape_as(hidden_states), router_logits


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = DummyMoeBlock()


def test_adapter_inspects_and_patches_dummy_moe():
    model = DummyModel()
    adapter = HfMoeAdapter(model)

    report = adapter.inspect()
    assert report.candidates[0].name == "moe"

    hidden_states = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    original, _ = model.moe(hidden_states)

    patched = adapter.patch(AggregationConfig(mode="standard"))
    standard, _ = model.moe(hidden_states)

    assert patched == ["moe"]
    assert torch.allclose(original, standard, atol=1e-6)
    assert len(adapter.collect_diagnostics()) == 1

    adapter.unpatch()
    restored, _ = model.moe(hidden_states)
    assert torch.allclose(original, restored, atol=1e-6)
