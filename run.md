# Run Guide

## Gemma 4 26B A4B Inference-Only Experiments

Inspect model structure:

```bash
python3 -m moe_orthogonal.cli inspect --config configs/gemma4_26b_a4b_inference.yaml
```

Check that patched standard aggregation reproduces the original forward:

```bash
python3 -m moe_orthogonal.cli compare --config configs/gemma4_26b_a4b_inference.yaml
```

Run a small PPL evaluation:

```bash
python3 -m moe_orthogonal.cli eval-ppl \
  --config configs/gemma4_26b_a4b_inference.yaml \
  --output outputs/logs/gemma4_26b_a4b_standard_eval.json
```

Change `aggregation.mode` in the config to `top1_ortho` to run inference-only
Top1-Ortho. The first acceptance check is that `compare` reports a small
standard reproduction error before trusting Top1-Ortho metrics.
