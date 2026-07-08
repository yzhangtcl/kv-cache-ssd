#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable, Sequence

import torch


PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class SSDBlockKVConfig:
    """Configuration for the baseline SSD block KV cache.

    The baseline stores de-rotated keys and original values on SSD. Every
    `block_size` tokens form one searchable block. By default a block is
    represented by one mean de-rotated key vector from `summary_layer`. Setting
    `summary_centroids_per_block` above 1 splits tokens inside a block with a
    small KD-tree-style partition and stores one mean per partition.
    """

    block_size: int = 256
    top_k_blocks: int = 8
    summary_layer: int = 0
    replay_position_base: int = 1
    dtype_on_ssd: torch.dtype = torch.float16
    metadata_name: str = "metadata.pt"
    autosave_metadata: bool = False
    summary_centroids_per_block: int = 1


@dataclass
class SSDBlockKVStats:
    blocks_written: int = 0
    tokens_written: int = 0
    bytes_written: int = 0
    block_searches: int = 0
    blocks_loaded: int = 0
    tokens_loaded: int = 0
    build_cache_calls: int = 0


@dataclass(frozen=True)
class KVBlockMeta:
    block_id: int
    token_start: int
    token_count: int
    path: str


@dataclass(frozen=True)
class BlockSelection:
    block_ids: list[int]
    scores: list[float]


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_sec: float
    peak_memory_gb: float
    ssd_cache: SSDBlockKVStats


def _cache_entries(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past_key_values is None:
        return []
    if isinstance(past_key_values, tuple):
        return [(key, value) for key, value in past_key_values]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return [
            (key, value)
            for key, value in zip(past_key_values.key_cache, past_key_values.value_cache)
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor)
        ]
    if hasattr(past_key_values, "layers"):
        entries = []
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                entries.append((key, value))
        return entries
    try:
        entries = []
        for item in past_key_values:
            key, value = item[0], item[1]
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                entries.append((key, value))
        return entries
    except TypeError:
        return []


def _cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    entries = _cache_entries(past_key_values)
    if not entries:
        return 0
    return int(entries[0][0].shape[-2])


def _dynamic_cache_cls():
    for module_name in ("transformers.cache_utils", "transformers"):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, "DynamicCache", None)
        if cls is not None:
            return cls
    return None


def _model_layers(model) -> list:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        layers = _find_attr(model, path)
        if layers is not None:
            try:
                return list(layers)
            except TypeError:
                return []
    return []


def _attention_layer_indices(model, attention_count: int) -> list[int]:
    layers = _model_layers(model)
    if layers:
        indices = [
            idx
            for idx, layer in enumerate(layers)
            if hasattr(layer, "self_attn") or not hasattr(layer, "linear_attn")
        ]
        if len(indices) == int(attention_count):
            return indices

    config = getattr(model, "config", None)
    for attr in ("layer_types", "layers_block_type", "attention_layer_types"):
        values = getattr(config, attr, None)
        if values is None:
            continue
        indices = [
            idx
            for idx, value in enumerate(values)
            if "linear" not in str(value).lower()
        ]
        if len(indices) == int(attention_count):
            return indices

    return list(range(int(attention_count)))


def _new_dynamic_cache(cls, model=None):
    config = getattr(model, "config", None)
    if config is not None:
        for kwargs in ({"config": config}, {"model_config": config}):
            try:
                return cls(**kwargs)
            except TypeError:
                pass
            except Exception:
                pass
    try:
        return cls()
    except TypeError:
        try:
            return cls(config=None)
        except Exception:
            return None


def _cache_layers(past_key_values) -> list:
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return []
    try:
        return list(layers)
    except TypeError:
        return []


def _copy_linear_attention_state(target_cache, source_cache) -> None:
    target_layers = _cache_layers(target_cache)
    source_layers = _cache_layers(source_cache)
    if not target_layers or not source_layers:
        return

    for target_layer, source_layer in zip(target_layers, source_layers):
        for attr in ("conv_states", "recurrent_states"):
            value = getattr(source_layer, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(target_layer, attr, value.detach().clone())


def _clone_linear_attention_cache(past_key_values, model=None):
    if past_key_values is None:
        return None
    cls = _dynamic_cache_cls()
    if cls is None:
        return None
    cache = _new_dynamic_cache(cls, model=model)
    if cache is None:
        return None
    _copy_linear_attention_state(cache, past_key_values)
    return cache


def _legacy_tuple_to_dynamic_cache(past_key_values: PastKeyValues, model=None):
    cls = _dynamic_cache_cls()
    if cls is None:
        return past_key_values

    if model is None:
        from_legacy = getattr(cls, "from_legacy_cache", None)
        if from_legacy is not None:
            try:
                return from_legacy(past_key_values)
            except Exception:
                pass

    cache = _new_dynamic_cache(cls, model=model)
    if cache is None:
        return past_key_values

    update = getattr(cache, "update", None)
    if update is None:
        return past_key_values

    try:
        layer_indices = _attention_layer_indices(model, len(past_key_values)) if model is not None else range(len(past_key_values))
        for layer_idx, (key, value) in zip(layer_indices, past_key_values):
            update(key, value, layer_idx)
        return cache
    except Exception:
        return past_key_values


def _to_model_cache(past_key_values, model=None, linear_state_cache=None):
    if past_key_values is None or hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    if isinstance(past_key_values, tuple):
        cache = _legacy_tuple_to_dynamic_cache(past_key_values, model=model)
        if cache is not past_key_values and linear_state_cache is not None:
            _copy_linear_attention_state(cache, linear_state_cache)
        return cache
    return past_key_values


def _position_ids(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


def _cache_position(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long)


def _model_input_device(model) -> torch.device:
    try:
        return next(model.get_input_embeddings().parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu"))


def _cuda_synchronize_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _find_attr(root, dotted_path: str):
    obj = root
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


class RotaryEmbeddingAdapter:
    """Small adapter around Qwen/HF rotary embeddings.

    Qwen-style RoPE rotates the key before it enters the KV cache. The SSD
    store uses inverse rotation before writing and normal rotation when a block
    is replayed back into GPU memory.
    """

    def __init__(
        self,
        *,
        rotary_module=None,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        device: torch.device | str | None = None,
    ) -> None:
        self.rotary_module = rotary_module
        self.head_dim = int(head_dim)
        self.rope_theta = float(rope_theta)
        self.device = torch.device(device) if device is not None else None

    @classmethod
    def from_model(cls, model, head_dim: int) -> "RotaryEmbeddingAdapter":
        rotary_module = None
        for path in (
            "model.rotary_emb",
            "transformer.rotary_emb",
            "base_model.model.rotary_emb",
        ):
            rotary_module = _find_attr(model, path)
            if rotary_module is not None:
                break

        config = getattr(model, "config", None)
        rope_theta = getattr(config, "rope_theta", 1_000_000.0)
        return cls(
            rotary_module=rotary_module,
            head_dim=head_dim,
            rope_theta=rope_theta,
            device=_model_input_device(model),
        )

    def apply(self, key: torch.Tensor, positions: torch.Tensor | Sequence[int]) -> torch.Tensor:
        cos, sin = self.cos_sin(positions, like=key)
        return self._apply_with_cos_sin(key, cos, sin, inverse=False)

    def inverse(self, key: torch.Tensor, positions: torch.Tensor | Sequence[int]) -> torch.Tensor:
        cos, sin = self.cos_sin(positions, like=key)
        return self._apply_with_cos_sin(key, cos, sin, inverse=True)

    def cos_sin(
        self,
        positions: torch.Tensor | Sequence[int],
        *,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(list(positions), device=like.device, dtype=torch.long)
        positions = positions.to(device=like.device, dtype=torch.long).reshape(1, -1)

        if self.rotary_module is not None:
            with torch.inference_mode():
                try:
                    cos, sin = self.rotary_module(like, positions)
                    return self._shape_cos_sin(cos, sin, like)
                except TypeError:
                    pass

                seq_len = int(positions.max().item()) + 1 if positions.numel() else 0
                try:
                    cos, sin = self.rotary_module(like, seq_len=seq_len)
                    cos = cos.index_select(0, positions.reshape(-1))
                    sin = sin.index_select(0, positions.reshape(-1))
                    return self._shape_cos_sin(cos, sin, like)
                except TypeError:
                    pass

        return self._fallback_cos_sin(positions.reshape(-1), like)

    def _shape_cos_sin(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.to(device=like.device, dtype=torch.float32)
        sin = sin.to(device=like.device, dtype=torch.float32)
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif cos.dim() != 4:
            raise ValueError(f"unsupported rotary cos/sin rank: {cos.dim()}")
        return cos, sin

    def _fallback_cos_sin(
        self,
        positions: torch.Tensor,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for fallback RoPE")
        device = like.device
        dtype = torch.float32
        idx = torch.arange(0, self.head_dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.rope_theta ** (idx / self.head_dim))
        freqs = torch.outer(positions.to(device=device, dtype=dtype), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    def _apply_with_cos_sin(
        self,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        original_dtype = key.dtype
        rot_dim = int(cos.shape[-1])
        if rot_dim > int(key.shape[-1]):
            raise ValueError(
                f"rotary dimension {rot_dim} is larger than key head dimension {key.shape[-1]}"
            )
        rotated_part = key[..., :rot_dim].float()
        pass_part = key[..., rot_dim:]
        if inverse:
            rotated_part = rotated_part * cos - _rotate_half(rotated_part) * sin
        else:
            rotated_part = rotated_part * cos + _rotate_half(rotated_part) * sin
        if pass_part.numel():
            rotated = torch.cat((rotated_part.to(original_dtype), pass_part), dim=-1)
        else:
            rotated = rotated_part.to(original_dtype)
        return rotated.contiguous()


class SSDBlockKVStore:
    """SSD-backed block store for de-rotated KV cache tensors."""

    def __init__(
        self,
        root: str | Path,
        config: SSDBlockKVConfig,
        rotary: RotaryEmbeddingAdapter,
        *,
        reset: bool = False,
    ) -> None:
        if config.block_size <= 0:
            raise ValueError("block_size must be positive")
        if config.top_k_blocks <= 0:
            raise ValueError("top_k_blocks must be positive")
        if config.summary_centroids_per_block <= 0:
            raise ValueError("summary_centroids_per_block must be positive")
        self.root = Path(root)
        self.config = config
        self.rotary = rotary
        self.stats = SSDBlockKVStats()
        self._metas: list[KVBlockMeta] = []
        self._summaries: list[torch.Tensor] = []
        self._next_block_id = 0

        if reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not reset:
            self._load_metadata_if_present()

    @property
    def block_count(self) -> int:
        return len(self._metas)

    @property
    def token_count(self) -> int:
        return sum(meta.token_count for meta in self._metas)

    def add_past_key_values(
        self,
        past_key_values,
        *,
        token_start: int = 0,
        token_offset: int = 0,
        token_count: int | None = None,
    ) -> None:
        """Write a slice of a Transformers past_key_values object to SSD.

        `token_start` is the original absolute token index for the first token
        in the slice. The stored keys are inverse-RoPE'd with these positions.
        """

        entries = _cache_entries(past_key_values)
        if not entries:
            return
        seq_len = int(entries[0][0].shape[-2])
        if token_count is None:
            token_count = seq_len - token_offset
        token_count = int(token_count)
        if token_offset < 0 or token_count < 0 or token_offset + token_count > seq_len:
            raise ValueError("invalid token slice for past_key_values")

        cursor = 0
        while cursor < token_count:
            take = min(self.config.block_size, token_count - cursor)
            start = token_offset + cursor
            positions = torch.arange(
                token_start + cursor,
                token_start + cursor + take,
                device=entries[0][0].device,
                dtype=torch.long,
            )
            self._write_block(entries, start=start, length=take, positions=positions)
            cursor += take

    def add_unrotated_layers(
        self,
        layers: PastKeyValues,
        *,
        token_start: int,
    ) -> None:
        """Write already de-rotated keys and values to SSD."""

        entries = _cache_entries(layers)
        if not entries:
            return
        seq_len = int(entries[0][0].shape[-2])
        cursor = 0
        while cursor < seq_len:
            take = min(self.config.block_size, seq_len - cursor)
            block_layers = []
            for key, value in entries:
                block_layers.append(
                    (
                        key[:, :, cursor : cursor + take].detach().cpu().to(self.config.dtype_on_ssd),
                        value[:, :, cursor : cursor + take].detach().cpu().to(self.config.dtype_on_ssd),
                    )
                )
            self._write_unrotated_block(tuple(block_layers), token_start + cursor)
            cursor += take

    def select_blocks(
        self,
        query: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> BlockSelection:
        """Return IDs of the highest-scoring blocks for a query vector."""

        self.stats.block_searches += 1
        if not self._summaries:
            return BlockSelection(block_ids=[], scores=[])
        top_k = self.config.top_k_blocks if top_k is None else int(top_k)
        top_k = max(0, min(top_k, len(self._summaries)))
        if top_k == 0:
            return BlockSelection(block_ids=[], scores=[])

        summaries = torch.stack(self._summaries, dim=0).to(device=query.device, dtype=torch.float32)
        query_for_score = _normalize_query_for_summaries(query, summaries)
        if query_for_score.dim() == 1:
            centroid_vectors = torch.nn.functional.normalize(summaries.mean(dim=2), dim=-1)
            centroid_scores = centroid_vectors @ query_for_score
        else:
            centroid_scores = (summaries * query_for_score.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            centroid_scores = centroid_scores.mean(dim=-1)
        scores = centroid_scores.max(dim=1).values
        values, indices = torch.topk(scores, k=top_k, largest=True)
        block_ids = [self._metas[int(idx)].block_id for idx in indices.detach().cpu().tolist()]
        return BlockSelection(
            block_ids=block_ids,
            scores=[float(score) for score in values.detach().cpu().tolist()],
        )

    def build_past_key_values(
        self,
        block_ids: Sequence[int],
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        include_unrotated_tail: PastKeyValues | None = None,
    ) -> PastKeyValues:
        """Load selected blocks and replay them as a contiguous GPU KV cache.

        The selected SSD keys and optional tail keys are de-rotated internally.
        This method assigns them new contiguous RoPE positions beginning at
        `config.replay_position_base`, matching the "1..k" replay convention.
        """

        self.stats.build_cache_calls += 1
        target_device = torch.device(device)
        selected = self._load_blocks_in_sequence_order(block_ids)
        tail_entries = _cache_entries(include_unrotated_tail)
        if not selected and not tail_entries:
            return tuple()

        layer_count = len(selected[0][0]) if selected else len(tail_entries)
        merged_layers = []
        replay_cursor = int(self.config.replay_position_base)

        for layer_idx in range(layer_count):
            key_parts = []
            value_parts = []
            for block_layers, _meta in selected:
                key_cpu, value_cpu = block_layers[layer_idx]
                key_parts.append(key_cpu)
                value_parts.append(value_cpu)
            if tail_entries:
                tail_key, tail_value = tail_entries[layer_idx]
                key_parts.append(tail_key.detach().cpu())
                value_parts.append(tail_value.detach().cpu())

            key_unrotated = torch.cat(key_parts, dim=2).to(device=target_device)
            value = torch.cat(value_parts, dim=2).to(device=target_device)
            if dtype is not None:
                key_unrotated = key_unrotated.to(dtype)
                value = value.to(dtype)

            length = int(key_unrotated.shape[-2])
            positions = torch.arange(
                replay_cursor,
                replay_cursor + length,
                device=target_device,
                dtype=torch.long,
            )
            key = self.rotary.apply(key_unrotated, positions)
            merged_layers.append((key.contiguous(), value.contiguous()))

        return tuple(merged_layers)

    def _write_block(
        self,
        entries: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        start: int,
        length: int,
        positions: torch.Tensor,
    ) -> None:
        block_layers = []
        for key, value in entries:
            key_slice = key[:, :, start : start + length]
            value_slice = value[:, :, start : start + length]
            key_unrotated = self.rotary.inverse(key_slice, positions)
            block_layers.append(
                (
                    key_unrotated.detach().cpu().to(self.config.dtype_on_ssd),
                    value_slice.detach().cpu().to(self.config.dtype_on_ssd),
                )
            )
        self._write_unrotated_block(tuple(block_layers), int(positions[0].item()))

    def _write_unrotated_block(self, block_layers: PastKeyValues, token_start: int) -> None:
        block_id = self._next_block_id
        self._next_block_id += 1
        path = self.root / f"block_{block_id:08d}.pt"

        torch.save({"layers": block_layers}, path)
        token_count = int(block_layers[0][0].shape[-2])
        summary = self._make_block_summary(block_layers)
        meta = KVBlockMeta(
            block_id=block_id,
            token_start=int(token_start),
            token_count=token_count,
            path=path.name,
        )
        self._metas.append(meta)
        self._summaries.append(summary.cpu())

        self.stats.blocks_written += 1
        self.stats.tokens_written += token_count
        self.stats.bytes_written += path.stat().st_size
        if self.config.autosave_metadata:
            self.save_metadata()

    def _make_block_summary(self, block_layers: PastKeyValues) -> torch.Tensor:
        layer_idx = min(max(0, self.config.summary_layer), len(block_layers) - 1)
        key = block_layers[layer_idx][0].float()
        centroid_count = max(1, int(self.config.summary_centroids_per_block))
        # [batch, kv_heads, tokens, dim] -> [tokens, kv_heads, dim]
        token_vectors = key.mean(dim=0).permute(1, 0, 2).contiguous()
        centroids = _block_centroids_by_kdtree(
            token_vectors,
            max_centroids=centroid_count,
        )
        if int(centroids.shape[0]) < centroid_count:
            pad = centroids[-1:].expand(centroid_count - int(centroids.shape[0]), -1, -1)
            centroids = torch.cat((centroids, pad), dim=0)
        return torch.nn.functional.normalize(centroids, dim=-1)

    def _load_blocks_in_sequence_order(
        self,
        block_ids: Sequence[int],
    ) -> list[tuple[PastKeyValues, KVBlockMeta]]:
        if not block_ids:
            return []
        wanted = set(int(block_id) for block_id in block_ids)
        by_id = {meta.block_id: meta for meta in self._metas}
        metas = [by_id[block_id] for block_id in wanted if block_id in by_id]
        metas.sort(key=lambda meta: meta.token_start)

        loaded = []
        for meta in metas:
            payload = torch.load(self.root / meta.path, map_location="cpu")
            layers = tuple((key, value) for key, value in payload["layers"])
            loaded.append((layers, meta))
            self.stats.blocks_loaded += 1
            self.stats.tokens_loaded += meta.token_count
        return loaded

    def save_metadata(self) -> None:
        payload = {
            "config": {
                **asdict(self.config),
                "dtype_on_ssd": str(self.config.dtype_on_ssd).replace("torch.", ""),
            },
            "metas": [asdict(meta) for meta in self._metas],
            "summaries": self._summaries,
            "stats": asdict(self.stats),
            "next_block_id": self._next_block_id,
        }
        torch.save(payload, self.root / self.config.metadata_name)
        json_payload = {
            "config": payload["config"],
            "metas": payload["metas"],
            "stats": payload["stats"],
            "next_block_id": self._next_block_id,
        }
        (self.root / "metadata.json").write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_metadata_if_present(self) -> None:
        path = self.root / self.config.metadata_name
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu")
        self._metas = [KVBlockMeta(**item) for item in payload.get("metas", [])]
        self._summaries = [
            item.float().cpu() if item.dim() == 3 else item.float().cpu().unsqueeze(0)
            for item in payload.get("summaries", [])
        ]
        self._next_block_id = int(payload.get("next_block_id", len(self._metas)))


def _block_centroids_by_kdtree(
    token_vectors: torch.Tensor,
    *,
    max_centroids: int,
) -> torch.Tensor:
    """Split tokens by high-variance dimensions and return per-leaf means.

    `token_vectors` has shape [tokens, kv_heads, dim]. We build a tiny
    KD-tree-like partition on flattened token vectors, then store a per-head
    mean for every leaf. This gives a block several local representatives
    without introducing sklearn/faiss as a dependency.
    """

    token_count = int(token_vectors.shape[0])
    if token_count == 0:
        raise ValueError("cannot summarize an empty KV block")
    max_centroids = max(1, min(int(max_centroids), token_count))
    if max_centroids == 1:
        return token_vectors.mean(dim=0, keepdim=True)

    flat = token_vectors.reshape(token_count, -1).float()
    leaves = [torch.arange(token_count, device=token_vectors.device, dtype=torch.long)]
    while len(leaves) < max_centroids:
        split_at = -1
        split_var = -1.0
        for idx, leaf in enumerate(leaves):
            if int(leaf.numel()) <= 1:
                continue
            variance = flat.index_select(0, leaf).var(dim=0, unbiased=False).max().item()
            if variance > split_var:
                split_at = idx
                split_var = variance
        if split_at < 0:
            break

        leaf = leaves.pop(split_at)
        values = flat.index_select(0, leaf)
        split_dim = int(values.var(dim=0, unbiased=False).argmax().item())
        order = torch.argsort(values[:, split_dim], stable=True)
        midpoint = max(1, min(int(order.numel()) - 1, int(order.numel()) // 2))
        left = leaf.index_select(0, order[:midpoint])
        right = leaf.index_select(0, order[midpoint:])
        leaves.extend([left, right])

    centroids = []
    for leaf in leaves[:max_centroids]:
        centroids.append(token_vectors.index_select(0, leaf).mean(dim=0))
    return torch.stack(centroids, dim=0)


def _normalize_query_for_summaries(query: torch.Tensor, summaries: torch.Tensor) -> torch.Tensor:
    dim = int(summaries.shape[-1])
    heads = int(summaries.shape[-2])
    query = query.detach().to(device=summaries.device, dtype=torch.float32)
    if query.shape[-1] != dim:
        raise ValueError(f"query head dimension {query.shape[-1]} does not match summary dim {dim}")
    if query.dim() == 1:
        return torch.nn.functional.normalize(query, dim=0)
    if query.shape[-2] == heads:
        query = query.reshape(-1, heads, dim).mean(dim=0)
        return torch.nn.functional.normalize(query, dim=-1)
    query = query.reshape(-1, dim).mean(dim=0)
    return torch.nn.functional.normalize(query, dim=0)


def _mean_last_key_query(
    past_key_values,
    rotary: RotaryEmbeddingAdapter,
    *,
    position: int | None = None,
) -> torch.Tensor:
    entries = _cache_entries(past_key_values)
    if not entries:
        return torch.empty(0)
    key = entries[0][0][:, :, -1:, :]
    if position is not None:
        key = rotary.inverse(key, torch.tensor([position], device=key.device, dtype=torch.long))
    return key[0, :, 0, :].float()


def _forward_with_cache(
    model,
    input_ids: torch.Tensor,
    *,
    past_key_values,
    position_start: int,
    linear_state_cache=None,
):
    past_len = _cache_seq_len(past_key_values)
    model_cache = _to_model_cache(past_key_values, model=model, linear_state_cache=linear_state_cache)
    if isinstance(model_cache, tuple):
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if "qwen3_5" in str(model_type).lower():
            raise RuntimeError(
                "failed to convert legacy tuple past_key_values to a Transformers Cache object; "
                "please check that transformers exposes DynamicCache in this environment"
            )
    attention_mask = torch.ones(
        (input_ids.shape[0], past_len + int(input_ids.shape[1])),
        device=input_ids.device,
        dtype=torch.long,
    )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": _position_ids(position_start, int(input_ids.shape[1]), input_ids.device),
        "cache_position": _cache_position(past_len, int(input_ids.shape[1]), input_ids.device),
        "past_key_values": model_cache,
        "use_cache": True,
    }
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" not in str(exc):
            raise
        kwargs.pop("cache_position")
        return model(**kwargs)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    greedy: bool,
    repetition_penalty: float = 1.0,
    penalty_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    logits = logits[:, -1, :].clone()
    if repetition_penalty != 1.0 and penalty_token_ids:
        token_ids = torch.tensor(list(penalty_token_ids), device=logits.device, dtype=torch.long)
        token_logits = logits.index_select(dim=1, index=token_ids)
        penalized = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )
        logits.scatter_(dim=1, index=token_ids.unsqueeze(0).expand_as(penalized), src=penalized)

    if greedy or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _eos_token_ids(tokenizer) -> set[int]:
    values: Iterable[int | list[int] | tuple[int, ...] | None] = [
        getattr(tokenizer, "eos_token_id", None)
    ]
    ids = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(value)
        else:
            ids.update(int(item) for item in value)
    return ids


def _unrotated_tail_from_last_token(
    past_key_values,
    rotary: RotaryEmbeddingAdapter,
    *,
    replay_position: int,
) -> PastKeyValues:
    layers = []
    for key, value in _cache_entries(past_key_values):
        key_last = key[:, :, -1:, :]
        value_last = value[:, :, -1:, :]
        key_unrotated = rotary.inverse(
            key_last,
            torch.tensor([replay_position], device=key_last.device, dtype=torch.long),
        )
        layers.append((key_unrotated.detach().cpu(), value_last.detach().cpu()))
    return tuple(layers)


def _append_tail(left: PastKeyValues | None, right: PastKeyValues) -> PastKeyValues:
    if not left:
        return tuple((key.clone(), value.clone()) for key, value in right)
    out = []
    for (left_key, left_value), (right_key, right_value) in zip(left, right):
        out.append(
            (
                torch.cat((left_key, right_key), dim=2).contiguous(),
                torch.cat((left_value, right_value), dim=2).contiguous(),
            )
        )
    return tuple(out)


def _tail_length(tail: PastKeyValues | None) -> int:
    if not tail:
        return 0
    return int(tail[0][0].shape[-2])


def generate_with_ssd_block_kv(
    model,
    tokenizer,
    prompt: str,
    *,
    ssd_dir: str | Path,
    max_new_tokens: int,
    prefill_chunk_tokens: int = 1024,
    config: SSDBlockKVConfig | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    greedy: bool = True,
    repetition_penalty: float = 1.0,
    use_chat_template: bool = False,
    chat_template_enable_thinking: bool | None = None,
    stream_callback=None,
) -> GenerationResult:
    """Baseline generation path using SSD block retrieval.

    This intentionally keeps the prefill implementation simple: it computes the
    full prompt cache once, writes de-rotated prompt KV blocks to SSD, then uses
    block retrieval for decode. The next optimization point is replacing this
    full-prefill stage with chunked prefill plus online block retrieval.
    """

    config = config or SSDBlockKVConfig()
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if chat_template_enable_thinking is not None:
            chat_kwargs["enable_thinking"] = chat_template_enable_thinking
        try:
            rendered_prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            rendered_prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    device = _model_input_device(model)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if prefill_chunk_tokens <= 0:
        raise ValueError("prefill_chunk_tokens must be positive")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _cuda_synchronize_if_available()
    started = time.perf_counter()

    prompt_len = int(input_ids.shape[1])
    past_key_values = None
    logits = None
    cursor = 0
    while cursor < prompt_len:
        chunk = input_ids[:, cursor : cursor + prefill_chunk_tokens]
        with torch.inference_mode():
            out = _forward_with_cache(
                model,
                chunk,
                past_key_values=past_key_values,
                position_start=cursor,
            )
        past_key_values = out.past_key_values
        logits = out.logits
        cursor += int(chunk.shape[1])

    entries = _cache_entries(past_key_values)
    if not entries:
        raise RuntimeError("model did not return usable past_key_values")
    head_dim = int(entries[0][0].shape[-1])
    model_dtype = entries[0][0].dtype
    rotary = RotaryEmbeddingAdapter.from_model(model, head_dim=head_dim)
    store = SSDBlockKVStore(ssd_dir, config, rotary, reset=True)
    store.add_past_key_values(past_key_values, token_start=0)
    linear_state_cache = _clone_linear_attention_cache(past_key_values, model=model)

    initial_query = _mean_last_key_query(
        past_key_values,
        rotary,
        position=max(0, prompt_len - 1),
    )
    del past_key_values
    del entries
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    penalty_token_ids = set(int(token_id) for token_id in input_ids[0].tolist())
    next_token = _sample_next_token(
        logits,
        temperature=temperature,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        penalty_token_ids=penalty_token_ids,
    )
    eos_ids = _eos_token_ids(tokenizer)
    generated: list[int] = []
    streamed_text = ""
    decode_tail: PastKeyValues | None = None
    query = initial_query

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        penalty_token_ids.add(token_id)
        if stream_callback is not None and token_id not in eos_ids:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            delta = text[len(streamed_text) :] if text.startswith(streamed_text) else text
            if delta:
                stream_callback(delta)
            streamed_text = text
        if token_id in eos_ids:
            break

        selection = store.select_blocks(query)
        gpu_cache = store.build_past_key_values(
            selection.block_ids,
            device=device,
            dtype=model_dtype,
            include_unrotated_tail=decode_tail,
        )
        replay_position = config.replay_position_base + _cache_seq_len(gpu_cache)
        with torch.inference_mode():
            out = _forward_with_cache(
                model,
                next_token,
                past_key_values=gpu_cache,
                position_start=replay_position,
                linear_state_cache=linear_state_cache,
            )
        linear_state_cache = _clone_linear_attention_cache(out.past_key_values, model=model)

        new_tail = _unrotated_tail_from_last_token(
            out.past_key_values,
            rotary,
            replay_position=replay_position,
        )
        decode_tail = _append_tail(decode_tail, new_tail)
        if _tail_length(decode_tail) >= config.block_size:
            store.add_unrotated_layers(decode_tail, token_start=prompt_len + len(generated) - _tail_length(decode_tail))
            decode_tail = None

        query = _mean_last_key_query(
            out.past_key_values,
            rotary,
            position=replay_position,
        )
        next_token = _sample_next_token(
            out.logits,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
            penalty_token_ids=penalty_token_ids,
        )

    if decode_tail:
        store.add_unrotated_layers(
            decode_tail,
            token_start=prompt_len + len(generated) - _tail_length(decode_tail),
        )
    store.save_metadata()

    _cuda_synchronize_if_available()
    elapsed = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    return GenerationResult(
        text=tokenizer.decode(generated, skip_special_tokens=True),
        prompt_tokens=prompt_len,
        generated_tokens=len(generated),
        elapsed_sec=elapsed,
        peak_memory_gb=peak_gb,
        ssd_cache=store.stats,
    )


__all__ = [
    "BlockSelection",
    "GenerationResult",
    "KVBlockMeta",
    "PastKeyValues",
    "RotaryEmbeddingAdapter",
    "SSDBlockKVConfig",
    "SSDBlockKVStats",
    "SSDBlockKVStore",
    "generate_with_ssd_block_kv",
]
