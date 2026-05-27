from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .deployment import Deployment
from .models import ModelShape
from .params import ParamBreakdown, estimate_params
from .utils import as_gib, ceil_div, parse_dtype_bytes


def _has_skip_module(shape: Optional[ModelShape], patterns: tuple[str, ...]) -> bool:
    if not shape:
        return False
    return any(any(pattern in module for pattern in patterns) for module in shape.quant_modules_to_not_convert)


def quantized_weight_bytes(deploy: Deployment, shape: Optional[ModelShape] = None) -> Tuple[float, float, float, str, str]:
    config_quant = shape.config_quant_method if shape else None
    non_expert_dtype = deploy.quantization or config_quant or deploy.dtype
    expert_dtype = deploy.quantization or (shape.expert_dtype if shape and shape.expert_dtype else non_expert_dtype)
    non_expert_bytes = parse_dtype_bytes(non_expert_dtype)
    expert_bytes = parse_dtype_bytes(expert_dtype)
    overhead = deploy.quant_overhead_ratio if min(non_expert_bytes, expert_bytes) < 2.0 else 0.0
    return non_expert_bytes, expert_bytes, overhead, str(non_expert_dtype), str(expert_dtype)


def estimate_weight_bytes(
    params: ParamBreakdown, deploy: Deployment, shape: Optional[ModelShape] = None
) -> Tuple[float, Dict[str, Any]]:
    non_expert_bytes_per_param, expert_bytes_per_param, overhead_ratio, non_expert_dtype, expert_dtype = (
        quantized_weight_bytes(deploy, shape)
    )
    tp = max(1, deploy.tp)
    dp = max(1, deploy.dp)
    pp = max(1, deploy.pp)

    non_expert_shard = params.non_expert / tp / pp
    expert_shard = params.expert / (tp * dp) / pp if deploy.ep else params.expert / tp / pp

    base_params = non_expert_shard + expert_shard
    non_expert_bytes = non_expert_shard * non_expert_bytes_per_param
    expert_bytes = expert_shard * expert_bytes_per_param
    preserved_non_expert_bytes = 0.0
    preserved_params = 0.0
    if shape and non_expert_bytes_per_param < 2.0:
        tp = max(1, deploy.tp)
        pp = max(1, deploy.pp)
        if _has_skip_module(shape, ("embed_tokens",)):
            preserved_params += params.embedding / tp / pp
        if _has_skip_module(shape, ("lm_head",)):
            preserved_params += params.lm_head / tp / pp
        preserved_non_expert_bytes = preserved_params * (parse_dtype_bytes(deploy.dtype) - non_expert_bytes_per_param)
        non_expert_bytes += preserved_non_expert_bytes
    base_bytes = non_expert_bytes + expert_bytes
    overhead_bytes = base_bytes * overhead_ratio
    return base_bytes + overhead_bytes, {
        "weight_params_per_gpu": base_params,
        "non_expert_params_per_gpu": non_expert_shard,
        "expert_params_per_gpu": expert_shard,
        "non_expert_weight_gib": as_gib(non_expert_bytes),
        "expert_weight_gib": as_gib(expert_bytes),
        "bytes_per_weight_param": non_expert_bytes_per_param,
        "bytes_per_non_expert_weight_param": non_expert_bytes_per_param,
        "bytes_per_expert_weight_param": expert_bytes_per_param,
        "non_expert_weight_dtype": non_expert_dtype,
        "expert_weight_dtype": expert_dtype,
        "quant_overhead_gib": as_gib(overhead_bytes),
        "preserved_non_quantized_params_per_gpu": preserved_params,
        "preserved_non_quantized_gib": as_gib(preserved_non_expert_bytes),
    }


def estimate_kv_cache_bytes(shape: ModelShape, deploy: Deployment) -> Tuple[float, Dict[str, Any]]:
    dtype = deploy.dtype if deploy.kv_cache_dtype == "auto" else deploy.kv_cache_dtype
    bytes_per_kv = parse_dtype_bytes(dtype)
    kv_layers_total = shape.full_attention_layers if shape.layer_types else shape.num_layers
    local_layers = ceil_div(kv_layers_total, max(1, deploy.pp))
    local_kv_heads = ceil_div(shape.num_key_value_heads, max(1, deploy.tp))
    tokens_by_seqs = deploy.max_model_len * deploy.max_num_seqs
    token_budget = deploy.max_num_batched_tokens or tokens_by_seqs
    rounded_tokens = ceil_div(token_budget, deploy.block_size) * deploy.block_size

    windowed_layers_local: Dict[int, int] = {}
    if shape.compress_ratios:
        ratios = shape.compress_ratios[:kv_layers_total]
        ratio_layers = ratios[:local_layers]
        effective_tokens = sum(
            rounded_tokens if ratio <= 0 else ceil_div(rounded_tokens, max(1, ratio)) for ratio in ratio_layers
        )
        compression_note = "DeepSeek-V4 token-wise compression ratios applied per local layer."
    elif shape.sliding_window_list and shape.swa_layers:
        if len(shape.sliding_window_list) == len(shape.swa_layers):
            window_by_layer = {
                int(layer): int(window)
                for layer, window in zip(shape.swa_layers, shape.sliding_window_list)
                if int(layer) < kv_layers_total
            }
        elif len(shape.sliding_window_list) >= kv_layers_total:
            window_by_layer = {
                layer: int(shape.sliding_window_list[layer])
                for layer in range(kv_layers_total)
                if int(shape.sliding_window_list[layer]) > 0
            }
        else:
            window_by_layer = {}
        local_layer_indices = list(range(local_layers))
        effective_tokens = 0
        for layer_idx in local_layer_indices:
            window = window_by_layer.get(layer_idx)
            if window:
                layer_tokens = min(rounded_tokens, ceil_div(window, deploy.block_size) * deploy.block_size)
                windowed_layers_local[layer_idx] = layer_tokens
            else:
                layer_tokens = rounded_tokens
            effective_tokens += layer_tokens
        compression_note = "Sliding-window attention layers cap KV tokens per configured window; other layers use full token budget."
    else:
        effective_tokens = rounded_tokens * local_layers
        compression_note = "No KV token compression configured."

    if shape.kv_lora_rank:
        cache_dim = shape.kv_lora_rank + (shape.qk_rope_head_dim or 0)
        bytes_per_layer_token = cache_dim * bytes_per_kv
        cache_layout = "compressed_kv_lora"
    else:
        cache_dim = 2 * local_kv_heads * shape.head_dim
        bytes_per_layer_token = cache_dim * bytes_per_kv
        cache_layout = "standard_kv"

    bytes_total = effective_tokens * bytes_per_layer_token
    bytes_per_token = bytes_total / rounded_tokens if rounded_tokens else 0
    return bytes_total, {
        "kv_tokens_reserved_per_gpu": rounded_tokens,
        "kv_effective_layer_tokens_per_gpu": effective_tokens,
        "kv_token_budget_raw": token_budget,
        "kv_block_size": deploy.block_size,
        "kv_bytes_per_token_per_gpu": bytes_per_token,
        "kv_gib_per_1k_tokens_per_gpu": as_gib(bytes_per_token * 1000),
        "kv_compression_note": compression_note,
        "kv_compress_ratios_local": shape.compress_ratios[:local_layers] if shape.compress_ratios else [],
        "kv_windowed_layers_local": windowed_layers_local,
        "local_kv_heads": local_kv_heads,
        "local_layers": local_layers,
        "kv_layers_total": kv_layers_total,
        "kv_cache_layout": cache_layout,
        "kv_cache_dim_per_token_layer": cache_dim,
        "bytes_per_kv_element": bytes_per_kv,
    }


def estimate_lora_bytes(shape: ModelShape, deploy: Deployment) -> float:
    if deploy.lora_rank <= 0 or deploy.lora_count <= 0:
        return 0.0
    bytes_per = parse_dtype_bytes(deploy.dtype)
    adapted_linears = 7
    params_per_adapter = shape.num_layers * adapted_linears * deploy.lora_rank * (
        shape.hidden_size + shape.hidden_size
    )
    return params_per_adapter * deploy.lora_count * bytes_per / max(1, deploy.tp)


def estimate_speculative_bytes(shape: ModelShape, deploy: Deployment, config_dir: Path) -> Tuple[float, str]:
    if not deploy.speculative:
        return 0.0, "disabled"

    spec = deploy.speculative
    mode = str(spec.get("mode", "ngram")).lower()
    if mode in {"ngram", "suffix"}:
        ratio = float(spec.get("overhead_ratio", 0.02))
        return 0.0, f"{mode}: scheduler overhead handled in runtime reserve; suggested reserve +{ratio:.0%}"

    if mode == "draft_model":
        draft_config = spec.get("draft_config")
        if not draft_config:
            ratio = float(spec.get("weight_ratio", 0.15))
            return ratio, f"draft_model: no draft_config supplied, using {ratio:.0%} of target weight+KV"
        draft_path = Path(draft_config)
        if not draft_path.is_absolute():
            draft_path = config_dir / draft_path
        draft_shape = ModelShape.from_config(json.loads(draft_path.read_text(encoding="utf-8")))
        draft_params = estimate_params(draft_shape)
        weight_bytes, _ = estimate_weight_bytes(draft_params, deploy, draft_shape)
        kv_bytes, _ = estimate_kv_cache_bytes(draft_shape, deploy)
        return weight_bytes + kv_bytes, f"draft_model: estimated from {draft_path}"

    hidden_layers = int(spec.get("num_layers", 1))
    vocab_head = bool(spec.get("vocab_head", False))
    params = hidden_layers * 3 * shape.hidden_size * shape.hidden_size
    if vocab_head:
        params += shape.hidden_size * shape.vocab_size
    bytes_total = params * parse_dtype_bytes(deploy.dtype) / max(1, deploy.tp)
    return bytes_total, f"{mode}: approximate auxiliary module"
