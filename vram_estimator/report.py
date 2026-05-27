from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .deployment import Deployment
from .memory import estimate_kv_cache_bytes, estimate_lora_bytes, estimate_speculative_bytes, estimate_weight_bytes
from .models import ModelShape
from .params import deepseek_v4_reported_params, estimate_active_params, estimate_params
from .utils import GIB, as_gib, pct


def _per_gpu_breakdown(
    *,
    shape: ModelShape,
    weight_bytes: float,
    weight_meta: Dict[str, Any],
    kv_bytes: float,
    kv_meta: Dict[str, Any],
    activation_bytes: float,
    runtime_bytes: float,
    lora_bytes: float,
    spec_bytes: float,
    spec_note: str,
    total_bytes: float,
) -> Dict[str, Any]:
    return {
        "weights": {
            "gib": as_gib(weight_bytes),
            "percent_of_total": pct(weight_bytes, total_bytes),
            "description": "Resident model weights held by this GPU after TP/PP and optional EP sharding.",
            "children": {
                "non_expert": {
                    "gib": weight_meta["non_expert_weight_gib"],
                    "params_b": weight_meta["non_expert_params_per_gpu"] / 1e9,
                },
                "expert": {
                    "gib": weight_meta["expert_weight_gib"],
                    "params_b": weight_meta["expert_params_per_gpu"] / 1e9,
                },
                "quant_overhead": {
                    "gib": weight_meta["quant_overhead_gib"],
                    "description": "Scale/zero/group metadata reserve for low-bit weights.",
                },
            },
        },
        "kv_cache": {
            "gib": as_gib(kv_bytes),
            "percent_of_total": pct(kv_bytes, total_bytes),
            "description": "Reserved KV cache capacity for this GPU.",
            "children": {
                "reserved_tokens": kv_meta["kv_tokens_reserved_per_gpu"],
                "raw_token_budget": kv_meta["kv_token_budget_raw"],
                "block_size": kv_meta["kv_block_size"],
                "local_layers": kv_meta["local_layers"],
                "kv_layers_total": kv_meta["kv_layers_total"],
                "local_kv_heads": kv_meta["local_kv_heads"],
                "head_dim": shape.head_dim,
                "cache_layout": kv_meta["kv_cache_layout"],
                "cache_dim_per_token_layer": kv_meta["kv_cache_dim_per_token_layer"],
                "bytes_per_kv_element": kv_meta["bytes_per_kv_element"],
                "gib_per_1k_tokens": kv_meta["kv_gib_per_1k_tokens_per_gpu"],
                "effective_layer_tokens": kv_meta["kv_effective_layer_tokens_per_gpu"],
                "compression_note": kv_meta["kv_compression_note"],
                "compress_ratios_local": kv_meta["kv_compress_ratios_local"],
                "windowed_layers_local": kv_meta["kv_windowed_layers_local"],
            },
        },
        "activation_peak": {
            "gib": as_gib(activation_bytes),
            "percent_of_total": pct(activation_bytes, total_bytes),
            "description": "Estimated temporary activation peak. Override with activation_peak_gib for profiling-calibrated values.",
        },
        "runtime_overhead": {
            "gib": as_gib(runtime_bytes),
            "percent_of_total": pct(runtime_bytes, total_bytes),
            "description": "CUDA/vLLM runtime reserve, allocator fragmentation, kernels, graphs and workspaces.",
        },
        "lora": {
            "gib": as_gib(lora_bytes),
            "percent_of_total": pct(lora_bytes, total_bytes),
            "description": "Approximate resident LoRA adapter memory.",
        },
        "speculative": {
            "gib": as_gib(spec_bytes),
            "percent_of_total": pct(spec_bytes, total_bytes),
            "description": spec_note,
        },
    }


def build_report(config: Dict[str, Any], deploy: Deployment, config_path: Path) -> Dict[str, Any]:
    shape = ModelShape.from_config(config)
    params = estimate_params(shape)
    weight_bytes, weight_meta = estimate_weight_bytes(params, deploy, shape)
    kv_bytes, kv_meta = estimate_kv_cache_bytes(shape, deploy)

    activation_bytes = (
        deploy.activation_peak_gib * GIB
        if deploy.activation_peak_gib is not None
        else weight_bytes * deploy.activation_factor
    )
    runtime_bytes = deploy.runtime_overhead_gib * GIB
    lora_bytes = estimate_lora_bytes(shape, deploy)
    spec_value, spec_note = estimate_speculative_bytes(shape, deploy, config_path.parent)
    spec_bytes = (weight_bytes + kv_bytes) * spec_value if isinstance(spec_value, float) and spec_value < 1.0 else float(spec_value)

    total_bytes = weight_bytes + kv_bytes + activation_bytes + runtime_bytes + lora_bytes + spec_bytes
    usable_bytes = None
    fit = None
    headroom_gib = None
    if deploy.gpu_memory_gib:
        usable_bytes = deploy.gpu_memory_gib * GIB * deploy.gpu_memory_utilization
        fit = total_bytes <= usable_bytes
        headroom_gib = as_gib(usable_bytes - total_bytes)

    reported = deepseek_v4_reported_params(shape)
    return {
        "model": {
            "model_type": shape.model_type,
            "wrapper_model_type": shape.wrapper_model_type,
            "language_model_only": shape.language_model_only,
            "architectures": shape.raw_architectures,
            "hidden_size": shape.hidden_size,
            "num_layers": shape.num_layers,
            "num_attention_heads": shape.num_attention_heads,
            "num_key_value_heads": shape.num_key_value_heads,
            "head_dim": shape.head_dim,
            "vocab_size": shape.vocab_size,
            "max_position_embeddings": shape.max_position_embeddings,
            "moe_layers": shape.moe_layers,
            "dense_mlp_layers": shape.dense_mlp_layers,
            "num_experts": shape.num_experts,
            "num_shared_experts": shape.num_shared_experts,
            "num_experts_per_tok": shape.num_experts_per_tok,
            "full_attention_layers": shape.full_attention_layers,
            "linear_attention_layers": shape.linear_attention_layers,
            "dsa_layers": shape.dsa_layers or [],
            "swa_layers": shape.swa_layers or [],
            "sliding_window_list": shape.sliding_window_list or [],
            "decoder_sparse_step": shape.decoder_sparse_step,
            "mlp_only_layers": shape.mlp_only_layers or [],
            "num_nextn_predict_layers": shape.num_nextn_predict_layers,
            "config_quant_method": shape.config_quant_method,
            "expert_dtype": shape.expert_dtype,
            "reported_params": (
                {"total": reported[0], "active": reported[1], "source": "DeepSeek-V4 official model card rounded values"}
                if reported
                else None
            ),
        },
        "deployment": deploy.__dict__,
        "params": params.as_dict(),
        "active_params": {
            "per_token": estimate_active_params(params, shape),
            "note": "For MoE, active params count top-k routed experts plus shared/non-expert params. VRAM still stores resident experts.",
        },
        "per_gpu_gib": {
            "weights": as_gib(weight_bytes),
            "kv_cache": as_gib(kv_bytes),
            "activation_peak": as_gib(activation_bytes),
            "runtime_overhead": as_gib(runtime_bytes),
            "lora": as_gib(lora_bytes),
            "speculative": as_gib(spec_bytes),
            "total": as_gib(total_bytes),
            "usable_budget": as_gib(usable_bytes) if usable_bytes else None,
            "headroom": headroom_gib,
        },
        "per_gpu_breakdown": _per_gpu_breakdown(
            shape=shape,
            weight_bytes=weight_bytes,
            weight_meta=weight_meta,
            kv_bytes=kv_bytes,
            kv_meta=kv_meta,
            activation_bytes=activation_bytes,
            runtime_bytes=runtime_bytes,
            lora_bytes=lora_bytes,
            spec_bytes=spec_bytes,
            spec_note=spec_note,
            total_bytes=total_bytes,
        ),
        "fit": fit,
        "notes": {
            **weight_meta,
            **kv_meta,
            "speculative": spec_note,
            "warning": "This is an estimator. Validate with vLLM startup logs/profile on the target hardware.",
        },
    }
