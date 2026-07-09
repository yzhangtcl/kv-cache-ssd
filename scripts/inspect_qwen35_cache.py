#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoConfig

from model_loader import load_tokenizer_and_model


def _shape(value):
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value[:3]:
            parts.append(_shape(item))
        suffix = "" if len(value) <= 3 else f", ... len={len(value)}"
        return f"{type(value).__name__}({', '.join(parts)}{suffix})"
    return type(value).__name__


def _public_tensor_attrs(obj):
    fields = {}
    for name, value in getattr(obj, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        if isinstance(value, torch.Tensor):
            fields[name] = _shape(value)
        elif isinstance(value, (list, tuple)) and any(isinstance(item, torch.Tensor) for item in value):
            fields[name] = _shape(value)
    return fields


def main() -> None:
    model_path = os.environ.get("MODEL_DIR", "/root/blockdata/data/models/qwen3.5-9b")
    prompt = os.environ.get("INSPECT_PROMPT", "请用一句话解释 KV cache。")
    max_layers = int(os.environ.get("INSPECT_MAX_LAYERS", "80"))

    print("model_path:", model_path)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("model_type:", getattr(cfg, "model_type", None))
    print("architectures:", getattr(cfg, "architectures", None))
    quantization_config = getattr(cfg, "quantization_config", None)
    if quantization_config is not None:
        print("quantization_config:", quantization_config)
    cfg_dict = cfg.to_dict()
    if (
        "fp8" in model_path.lower()
        or "finegrained_fp8" in str(quantization_config).lower()
        or "fp8" in str(quantization_config).lower()
    ) and os.environ.get("INSPECT_ALLOW_FP8") != "1":
        print(
            "Refusing to load an FP8 checkpoint for cache inspection. "
            "Set MODEL_DIR to the Qwen3.5-9B directory, or set INSPECT_ALLOW_FP8=1 if this is intentional.",
            file=sys.stderr,
        )
        sys.exit(2)
    if int(getattr(cfg, "num_hidden_layers", 0) or cfg_dict.get("num_hidden_layers", 0) or 0) > 48:
        print(
            "Warning: this config has more than 48 layers; check that MODEL_DIR is not still pointing at the 27B model.",
            file=sys.stderr,
        )

    tokenizer, model = load_tokenizer_and_model(model_path)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)

    cache = out.past_key_values
    print("model_type:", getattr(model.config, "model_type", None))
    print("cache_type:", type(cache))
    if hasattr(cache, "get_seq_length"):
        print("cache_seq_length:", cache.get_seq_length())
    if hasattr(cache, "has_previous_state"):
        try:
            print("cache_has_previous_state:", cache.has_previous_state())
        except Exception as exc:
            print("cache_has_previous_state_error:", type(exc).__name__, exc)

    model_layers = list(getattr(getattr(model, "model", None), "layers", []))
    cache_layers = list(getattr(cache, "layers", []))
    print("model_layers:", len(model_layers))
    print("cache_layers:", len(cache_layers))

    for idx, layer in enumerate(model_layers[:max_layers]):
        cache_layer = cache_layers[idx] if idx < len(cache_layers) else None
        layer_kind = []
        if hasattr(layer, "self_attn"):
            layer_kind.append("self_attn")
        if hasattr(layer, "linear_attn"):
            layer_kind.append("linear_attn")
        print(f"\n[{idx}] model_layer={type(layer).__name__} kind={'+'.join(layer_kind) or 'unknown'}")
        if cache_layer is None:
            print("  cache_layer: <missing>")
            continue
        print("  cache_layer:", type(cache_layer).__name__)
        fields = _public_tensor_attrs(cache_layer)
        if not fields:
            print("  tensor_fields: <none>")
        for name, value in fields.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
