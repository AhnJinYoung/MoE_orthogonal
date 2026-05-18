from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm


@dataclass(frozen=True)
class PplResult:
    loss: float
    ppl: float
    tokens: int


def evaluate_ppl(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    text_column: str = "text",
    seq_len: int = 1024,
    max_eval_tokens: int = 32768,
    device: str | torch.device | None = None,
) -> PplResult:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    buffer: list[int] = []

    with torch.no_grad():
        for row in tqdm(dataset, desc="eval-ppl"):
            text = row[text_column]
            buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            while len(buffer) >= seq_len + 1:
                chunk = buffer[: seq_len + 1]
                del buffer[:seq_len]
                input_ids = torch.tensor([chunk[:-1]], device=device)
                labels = torch.tensor([chunk[1:]], device=device)
                outputs = model(input_ids=input_ids, labels=labels)
                tokens = labels.numel()
                total_loss += float(outputs.loss.detach().cpu()) * tokens
                total_tokens += tokens
                if total_tokens >= max_eval_tokens:
                    loss = total_loss / total_tokens
                    return PplResult(loss=loss, ppl=math.exp(loss), tokens=total_tokens)

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated")
    loss = total_loss / total_tokens
    return PplResult(loss=loss, ppl=math.exp(loss), tokens=total_tokens)
