#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


def load_tokenizer_and_model(
    model_path: str,
    *,
    device_map: str = "cuda:0",
    attn_implementation: str = "sdpa",
):
    """Load text tokenizer plus a Qwen-compatible model.

    Qwen3.6 checkpoints may require AutoModelForMultimodalLM while older text
    checkpoints work with AutoModelForCausalLM. Keep the public interface the
    same for the KV-cache scripts.
    """

    import transformers

    tokenizer = _load_tokenizer(model_path, transformers)
    kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    errors: list[str] = []
    for class_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM"):
        cls = getattr(transformers, class_name, None)
        if cls is None:
            errors.append(f"{class_name}: unavailable in transformers {transformers.__version__}")
            continue
        try:
            model = cls.from_pretrained(model_path, **kwargs)
            model.eval()
            return tokenizer, model
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        f"failed to load model from {model_path}. Try upgrading transformers/accelerate.\n{joined}"
    )


def _load_tokenizer(model_path: str, transformers):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise
        processor = processor_cls.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        return tokenizer if tokenizer is not None else processor
