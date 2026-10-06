"""Round the physical coefficients through the storage precision."""

from __future__ import annotations

import torch


def quantize_roundtrip_(model, codec: dict) -> None:
    if str(codec.get("quantization", "float16")).lower() == "float32":
        return
    with torch.no_grad():
        model.coefficient_logits.copy_(model._coefficients_to_logits(model.coefficients().half().float()))
