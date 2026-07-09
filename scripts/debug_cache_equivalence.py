#!/usr/bin/env python3

from __future__ import annotations

import os

import torch

from model_loader import load_tokenizer_and_model
from ssd_block_kvcache import (
    RotaryEmbeddingAdapter,
    SSDBlockKVConfig,
    SSDBlockKVStore,
    _attention_layer_indices,
    _cache_entries,
    _clone_linear_attention_cache,
    _forward_with_cache,
    _mean_last_key_query,
)


def _render_prompt(tokenizer, prompt: str) -> str:
    if os.environ.get("USE_CHAT_TEMPLATE", "1") != "1":
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if os.environ.get("ENABLE_THINKING") is not None:
        kwargs["enable_thinking"] = os.environ.get("ENABLE_THINKING") == "1"
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def _top_tokens(tokenizer, logits: torch.Tensor, k: int = 5) -> list[tuple[int, str, float]]:
    values, indices = torch.topk(logits[0, -1].float(), k=k)
    out = []
    for token_id, value in zip(indices.tolist(), values.tolist()):
        out.append((int(token_id), tokenizer.decode([int(token_id)]), float(value)))
    return out


def _cache_attention_layer_indices(past_key_values) -> list[int]:
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return []
    out = []
    for idx, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
            out.append(idx)
    return out


def _has_previous_state(past_key_values) -> str:
    if not hasattr(past_key_values, "has_previous_state"):
        return "<missing>"
    try:
        return str(bool(past_key_values.has_previous_state()))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    model_path = os.environ.get("MODEL_DIR", "/root/blockdata/data/models/qwen3.5-9b")
    ssd_dir = os.environ.get("SSD_DIR", "/root/blockdata/data/kvssd-debug")
    prompt = os.environ.get("PROMPT", "请用三句话解释 KV cache 为什么影响长上下文推理速度。")

    tokenizer, model = load_tokenizer_and_model(model_path)

    rendered = _render_prompt(tokenizer, prompt)
    input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    prompt_len = int(input_ids.shape[-1])
    print("model_path:", model_path)
    print("prompt_tokens:", prompt_len)

    with torch.inference_mode():
        prefill = _forward_with_cache(model, input_ids, past_key_values=None, position_start=0)

    prefill_entries = _cache_entries(prefill.past_key_values)
    print("attention_kv_layers:", len(prefill_entries))
    print("inferred_attention_layer_indices:", _attention_layer_indices(model, len(prefill_entries)))
    print("cache_attention_layer_indices:", _cache_attention_layer_indices(prefill.past_key_values))
    print("prefill_has_previous_state:", _has_previous_state(prefill.past_key_values))
    print("prefill_top5:", _top_tokens(tokenizer, prefill.logits))

    next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
    print("chosen_next:", int(next_token.item()), tokenizer.decode([int(next_token.item())]))

    head_dim = int(prefill_entries[0][0].shape[-1])
    model_dtype = prefill_entries[0][0].dtype
    rotary = RotaryEmbeddingAdapter.from_model(model, head_dim=head_dim)
    store = SSDBlockKVStore(
        ssd_dir,
        SSDBlockKVConfig(
            block_size=256,
            top_k_blocks=1000,
            summary_centroids_per_block=4,
            preserve_original_positions=True,
        ),
        rotary,
        reset=True,
    )
    store.add_past_key_values(prefill.past_key_values, token_start=0)
    query = _mean_last_key_query(prefill.past_key_values, rotary, position=prompt_len - 1)
    selection = store.select_blocks(query)
    print("selected_blocks:", selection.block_ids)
    print("selected_scores:", selection.scores)

    replay_cache = store.build_past_key_values(
        selection.block_ids,
        device="cuda",
        dtype=model_dtype,
        preserve_original_positions=True,
    )
    replay_entries = _cache_entries(replay_cache)
    print("replay_kv_layers:", len(replay_entries))
    if prefill_entries and replay_entries:
        key_diff = (prefill_entries[0][0] - replay_entries[0][0]).float().abs().max().item()
        value_diff = (prefill_entries[0][1] - replay_entries[0][1]).float().abs().max().item()
        print("layer0_key_max_abs_diff:", key_diff)
        print("layer0_value_max_abs_diff:", value_diff)
        all_key_diff = max(
            (ref_key - replay_key).float().abs().max().item()
            for (ref_key, _), (replay_key, _) in zip(prefill_entries, replay_entries)
        )
        all_value_diff = max(
            (ref_value - replay_value).float().abs().max().item()
            for (_, ref_value), (_, replay_value) in zip(prefill_entries, replay_entries)
        )
        print("all_layers_key_max_abs_diff:", all_key_diff)
        print("all_layers_value_max_abs_diff:", all_value_diff)

    exact_attention_tuple = tuple(
        (key.detach().clone(), value.detach().clone()) for key, value in prefill_entries
    )

    linear_state_cache = _clone_linear_attention_cache(prefill.past_key_values, model=model)
    print("linear_state_cache_has_previous_state:", _has_previous_state(linear_state_cache))
    with torch.inference_mode():
        ref = _forward_with_cache(
            model,
            next_token,
            past_key_values=prefill.past_key_values,
            position_start=prompt_len,
        )
        exact_tuple = _forward_with_cache(
            model,
            next_token,
            past_key_values=exact_attention_tuple,
            position_start=prompt_len,
            linear_state_cache=linear_state_cache,
        )
        ssd = _forward_with_cache(
            model,
            next_token,
            past_key_values=replay_cache,
            position_start=prompt_len,
            linear_state_cache=linear_state_cache,
        )

    diff = (ref.logits[:, -1, :].float() - ssd.logits[:, -1, :].float()).abs()
    exact_diff = (ref.logits[:, -1, :].float() - exact_tuple.logits[:, -1, :].float()).abs()
    print("exact_tuple_logits_max_abs_diff:", exact_diff.max().item())
    print("exact_tuple_logits_mean_abs_diff:", exact_diff.mean().item())
    print("next_step_logits_max_abs_diff:", diff.max().item())
    print("next_step_logits_mean_abs_diff:", diff.mean().item())
    print("ref_top5:", _top_tokens(tokenizer, ref.logits))
    print("exact_tuple_top5:", _top_tokens(tokenizer, exact_tuple.logits))
    print("ssd_top5:", _top_tokens(tokenizer, ssd.logits))


if __name__ == "__main__":
    main()
