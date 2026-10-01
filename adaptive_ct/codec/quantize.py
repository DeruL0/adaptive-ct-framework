"""Round deployed parameters through the codec's storage precision."""

from __future__ import annotations

import torch


def quantize_roundtrip_(model, codec: dict) -> None:
    """Round the deployed parameters through the codec's storage precision."""
    dtype = torch.float32 if str(codec.get("quantization", "float16")).lower() == "float32" else torch.float16
    if dtype == torch.float32:
        return
    with torch.no_grad():
        if hasattr(model, "leaf_latents"):
            model.leaf_latents.copy_(model.leaf_latents.to(dtype).float())
            model.decoder.weight.copy_(model.decoder.weight.to(dtype).float())
            model.decoder.bias.copy_(model.decoder.bias.to(dtype).float())
        else:
            physical = model.coefficients().to(dtype).float()
            model.coefficient_logits.copy_(model._coefficients_to_logits(physical))
