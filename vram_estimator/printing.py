from __future__ import annotations

from typing import Any, Dict


def print_human(report: Dict[str, Any]) -> None:
    model = report["model"]
    params = report["params"]
    active_params = report["active_params"]
    vram = report["per_gpu_gib"]
    breakdown = report["per_gpu_breakdown"]
    notes = report["notes"]

    print(f"Model: {model['model_type']}  layers={model['num_layers']} hidden={model['hidden_size']}")
    if model["num_experts"]:
        print(
            "MoE: "
            f"moe_layers={model['moe_layers']} experts={model['num_experts']} "
            f"shared={model['num_shared_experts']} top_k={model['num_experts_per_tok']}"
        )
    print()
    print("Parameter breakdown:")
    for key in [
        "embedding",
        "lm_head",
        "vision",
        "attention",
        "dense_mlp",
        "router",
        "routed_experts",
        "shared_experts",
        "norms",
        "bias",
        "non_expert",
        "expert",
        "total",
    ]:
        print(f"  {key:16s} {params[key] / 1e9:10.3f} B")
    print(f"  {'active/token':16s} {active_params['per_token'] / 1e9:10.3f} B")
    print()
    print("Per-GPU VRAM estimate:")
    print(f"  {'component':16s} {'GiB':>10s} {'% total':>9s}  detail")
    component_keys = ["weights", "kv_cache", "kv_transfer", "comm_buffer", "activation_peak", "runtime_overhead", "lora", "speculative"]
    for key in [item for item in component_keys if item in breakdown]:
        item = breakdown[key]
        print(f"  {key:16s} {item['gib']:10.2f} {item['percent_of_total']:8.1f}%  {item['description']}")
        if key == "weights":
            children = item["children"]
            print(
                f"    {'non_expert':14s} {children['non_expert']['gib']:10.2f} GiB"
                f"  params={children['non_expert']['params_b']:.3f}B"
            )
            print(
                f"    {'expert':14s} {children['expert']['gib']:10.2f} GiB"
                f"  params={children['expert']['params_b']:.3f}B"
            )
            print(f"    {'quant_overhead':14s} {children['quant_overhead']['gib']:10.2f} GiB")
        elif key == "kv_cache":
            children = item["children"]
            print(
                f"    tokens={children['reserved_tokens']} raw_budget={children['raw_token_budget']} "
                f"block={children['block_size']} layers={children['local_layers']} "
                f"kv_layers_total={children['kv_layers_total']} kv_heads={children['local_kv_heads']} "
                f"head_dim={children['head_dim']} layout={children['cache_layout']} "
                f"cache_dim/layer={children['cache_dim_per_token_layer']} "
                f"bytes/elem={children['bytes_per_kv_element']}"
            )
            if children["effective_layer_tokens"] != children["reserved_tokens"] * children["local_layers"]:
                print(
                    f"    effective_layer_tokens={children['effective_layer_tokens']} "
                    f"({children['compression_note']})"
                )
            if children.get("windowed_layers_local"):
                print(f"    windowed_layers_local={len(children['windowed_layers_local'])}")
            print(f"    per_1k_tokens={children['gib_per_1k_tokens']:.4f} GiB/GPU")
        elif key == "kv_transfer":
            children = item["children"]
            print(
                f"    tokens={children['tokens']} block={children['block_size']} "
                f"buffer_factor={children['buffer_factor']} bytes/token={children['bytes_per_token_per_gpu']:.0f} "
                f"direction={children['direction']} mode={children['mode']} sizing={children['sizing_method']}"
            )
    print(f"  {'total':16s} {vram['total']:10.2f} {'100.0%':>9s}")
    if vram["usable_budget"] is not None:
        print(f"  {'usable_budget':16s} {vram['usable_budget']:10.2f} GiB")
        print(f"  {'headroom':16s} {vram['headroom']:10.2f} GiB")
        print(f"  {'fit':16s} {str(report['fit']):>10s}")
    print()
    print("Planner details:")
    detail_keys = [
        "weight_params_per_gpu",
        "non_expert_params_per_gpu",
        "expert_params_per_gpu",
        "bytes_per_weight_param",
        "bytes_per_non_expert_weight_param",
        "bytes_per_expert_weight_param",
        "non_expert_weight_dtype",
        "expert_weight_dtype",
        "quant_overhead_gib",
        "kv_tokens_reserved_per_gpu",
        "local_kv_heads",
        "local_layers",
        "kv_layers_total",
        "kv_cache_layout",
        "kv_cache_dim_per_token_layer",
        "bytes_per_kv_element",
        "speculative",
    ]
    for key in detail_keys:
        if key in notes:
            value = notes[key]
            if isinstance(value, float) and "params" in key:
                value = f"{value / 1e9:.3f} B"
            print(f"  {key:26s} {value}")


def print_pd_human(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print("Scenario: PD disaggregated serving")
    print()
    print("Summary:")
    print(f"  {'prefill total/GPU':22s} {summary['prefill_total_gib_per_gpu']:10.2f} GiB")
    print(f"  {'decode total/GPU':22s} {summary['decode_total_gib_per_gpu']:10.2f} GiB")
    print(f"  {'prefill KV/GPU':22s} {summary['prefill_kv_gib_per_gpu']:10.2f} GiB")
    print(f"  {'decode KV/GPU':22s} {summary['decode_kv_gib_per_gpu']:10.2f} GiB")
    if "prefill_kv_transfer_gib_per_gpu" in summary:
        print(f"  {'prefill transfer/GPU':22s} {summary['prefill_kv_transfer_gib_per_gpu']:10.2f} GiB")
        print(f"  {'decode transfer/GPU':22s} {summary['decode_kv_transfer_gib_per_gpu']:10.2f} GiB")
        print(f"  {'prefill comm/GPU':22s} {summary['prefill_comm_buffer_gib_per_gpu']:10.2f} GiB")
        print(f"  {'decode comm/GPU':22s} {summary['decode_comm_buffer_gib_per_gpu']:10.2f} GiB")
    print(f"  {'prefill weights/GPU':22s} {summary['prefill_weight_gib_per_gpu']:10.2f} GiB")
    print(f"  {'decode weights/GPU':22s} {summary['decode_weight_gib_per_gpu']:10.2f} GiB")
    print(f"  {'prefill fit':22s} {str(summary['prefill_fit']):>10s}")
    print(f"  {'decode fit':22s} {str(summary['decode_fit']):>10s}")
    print()

    print("== Prefill nodes ==")
    print_human(report["prefill"])
    print()
    print("== Decode nodes ==")
    print_human(report["decode"])
