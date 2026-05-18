from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from moe_orthogonal.aggregation import AggregationConfig
from moe_orthogonal.config import deep_get, load_yaml
from moe_orthogonal.hf_adapters import get_adapter
from moe_orthogonal.metrics import (
    cosine_similarity,
    expert_parallel_component_ratio,
    load_balance_stats,
    removed_redundancy_energy,
)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _load_hf_model(config: dict[str, Any]) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Install requirements.txt in the target environment."
        ) from exc

    model_id = deep_get(config, "model.model_id", "google/gemma-4-26B-A4B-it")
    trust_remote_code = bool(deep_get(config, "model.trust_remote_code", True))
    dtype_name = str(deep_get(config, "model.dtype", "bfloat16"))
    device_map = deep_get(config, "model.device_map", "auto")

    torch_dtype = "auto" if dtype_name == "auto" else _dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return model, tokenizer


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "model_class": report.model_class,
        "model_type": report.model_type,
        "candidates": [
            {
                "name": candidate.name,
                "class_name": candidate.class_name,
                "has_router": candidate.has_router,
                "has_experts": candidate.has_experts,
                "num_experts": candidate.num_experts,
            }
            for candidate in report.candidates
        ],
    }


def command_inspect(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    model, _ = _load_hf_model(config)
    adapter = get_adapter(model, deep_get(config, "model.model_id"))
    _print_json(_report_to_dict(adapter.inspect()))


def _patch_model(config: dict[str, Any], model: Any):
    aggregation = AggregationConfig(
        mode=deep_get(config, "aggregation.mode", "standard"),
        eps=float(deep_get(config, "aggregation.eps", 1e-6)),
    )
    adapter = get_adapter(model, deep_get(config, "model.model_id"))
    patched = adapter.patch(aggregation)
    adapter.patch_forward_for_diagnostics()
    return adapter, patched


def _diagnostic_summary(adapter: Any) -> dict[str, float]:
    diagnostics = adapter.collect_diagnostics()
    if not diagnostics:
        return {}

    pcr_values = []
    rre_values = []
    cos_values = []
    load_vars = []
    load_ratios = []
    for diag in diagnostics:
        expert_outputs = diag["expert_outputs"]
        gate_weights = diag["gate_weights"]
        topk_indices = diag["topk_indices"]
        num_experts = int(diag["router_logits"].shape[-1])
        pcr_values.append(expert_parallel_component_ratio(expert_outputs).detach().float().cpu())
        rre_values.append(removed_redundancy_energy(expert_outputs, gate_weights).detach().float().cpu())
        cos_values.append(cosine_similarity(expert_outputs).detach().float().cpu())
        load = load_balance_stats(topk_indices.cpu(), num_experts)
        load_vars.append(load.variance)
        load_ratios.append(load.max_min_ratio)

    return {
        "pcr": float(torch.stack(pcr_values).mean()),
        "rre": float(torch.stack(rre_values).mean()),
        "cosine_similarity": float(torch.stack(cos_values).mean()),
        "load_variance": float(torch.stack(load_vars).mean()),
        "load_max_min_ratio": float(torch.stack(load_ratios).mean()),
        "num_diagnostic_layers": len(diagnostics),
    }


def command_smoke(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    model, tokenizer = _load_hf_model(config)
    adapter, patched = _patch_model(config, model)
    prompt = args.prompt or deep_get(config, "eval.prompt", "The future of sparse MoE models is")
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = {
        "patched_layers": patched,
        "logits_shape": list(outputs.logits.shape),
        "diagnostics": _diagnostic_summary(adapter),
    }
    _print_json(result)


def command_compare(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    model, tokenizer = _load_hf_model(config)
    prompt = args.prompt or deep_get(config, "eval.prompt", "The future of sparse MoE models is")
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        original = model(**inputs).logits.detach().float().cpu()

    config.setdefault("aggregation", {})["mode"] = "standard"
    adapter, patched = _patch_model(config, model)
    with torch.no_grad():
        standard = model(**inputs).logits.detach().float().cpu()
    standard_diag = _diagnostic_summary(adapter)

    for module in model.modules():
        if hasattr(module, "aggregator"):
            module.aggregator.config = AggregationConfig(
                mode="top1_ortho",
                eps=float(deep_get(config, "aggregation.eps", 1e-6)),
            )

    with torch.no_grad():
        top1_ortho = model(**inputs).logits.detach().float().cpu()
    ortho_diag = _diagnostic_summary(adapter)

    result = {
        "patched_layers": patched,
        "standard_reproduction": {
            "max_abs_error": float((original - standard).abs().max()),
            "mean_abs_error": float((original - standard).abs().mean()),
            "next_token_top1_match": bool(
                original[:, -1].argmax(dim=-1).eq(standard[:, -1].argmax(dim=-1)).all()
            ),
        },
        "top1_ortho_delta": {
            "max_abs_delta_vs_standard": float((standard - top1_ortho).abs().max()),
            "mean_abs_delta_vs_standard": float((standard - top1_ortho).abs().mean()),
        },
        "standard_diagnostics": standard_diag,
        "top1_ortho_diagnostics": ortho_diag,
    }
    _print_json(result)


def command_eval_ppl(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is not installed. Install requirements.txt in the target environment."
        ) from exc

    from moe_orthogonal.eval import evaluate_ppl

    config = load_yaml(args.config)
    model, tokenizer = _load_hf_model(config)
    adapter, patched = _patch_model(config, model)

    dataset_name = deep_get(config, "data.dataset_name", "Salesforce/wikitext")
    dataset_config = deep_get(config, "data.dataset_config", "wikitext-103-raw-v1")
    split = deep_get(config, "data.split", "validation")
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    result = evaluate_ppl(
        model,
        tokenizer,
        dataset,
        text_column=deep_get(config, "data.text_column", "text"),
        seq_len=int(deep_get(config, "eval.seq_len", 1024)),
        max_eval_tokens=int(deep_get(config, "eval.max_eval_tokens", 32768)),
    )
    payload = {
        "patched_layers": patched,
        "loss": result.loss,
        "ppl": result.ppl,
        "tokens": result.tokens,
        "diagnostics": _diagnostic_summary(adapter),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_json(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moe-ortho")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.set_defaults(func=command_inspect)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--prompt")
    smoke_parser.set_defaults(func=command_smoke)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--config", required=True)
    compare_parser.add_argument("--prompt")
    compare_parser.set_defaults(func=command_compare)

    eval_parser = subparsers.add_parser("eval-ppl")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--output")
    eval_parser.set_defaults(func=command_eval_ppl)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
