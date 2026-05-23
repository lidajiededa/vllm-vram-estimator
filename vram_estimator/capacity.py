from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .deployment import Deployment
from .models import ModelShape
from .report import build_report
from .utils import as_gib


def _with_model_len(deploy: Deployment, max_model_len: int) -> Deployment:
    return replace(deploy, max_model_len=max_model_len)


def _fit_at(config: Dict[str, Any], deploy: Deployment, config_path: Path, max_model_len: int) -> Dict[str, Any]:
    return build_report(config, _with_model_len(deploy, max_model_len), config_path)


def find_max_context_len(
    config: Dict[str, Any],
    deploy: Deployment,
    config_path: Path,
    *,
    upper: int | None = None,
) -> Dict[str, Any]:
    if not deploy.gpu_memory_gib:
        raise ValueError("gpu_memory_gib is required for max context search")
    if deploy.max_num_batched_tokens is not None:
        report = build_report(config, deploy, config_path)
        return {
            "mode": "fixed_token_budget",
            "max_model_len": deploy.max_model_len,
            "fit": report["fit"],
            "report": report,
            "warning": "max_num_batched_tokens is set, so KV memory is controlled by that fixed token budget rather than max_model_len.",
        }

    shape = ModelShape.from_config(config)
    model_limit = shape.max_position_embeddings
    search_upper = upper or model_limit or deploy.max_model_len
    if search_upper <= 0:
        raise ValueError("capacity search upper bound must be positive")

    while _fit_at(config, deploy, config_path, search_upper)["fit"]:
        next_upper = search_upper * 2
        if upper is not None or model_limit is not None or next_upper > 16_777_216:
            break
        search_upper = next_upper

    low = 0
    high = search_upper
    best_len = 0
    best_report = None
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        report = _fit_at(config, deploy, config_path, mid)
        if report["fit"]:
            best_len = mid
            best_report = report
            low = mid + 1
        else:
            high = mid - 1

    if best_report is None:
        best_report = _fit_at(config, deploy, config_path, 1)

    return {
        "mode": "max_model_len",
        "max_model_len": best_len,
        "fit": bool(best_report["fit"]),
        "report": best_report,
        "budget_gib": deploy.gpu_memory_gib * deploy.gpu_memory_utilization,
        "total_gib": best_report["per_gpu_gib"]["total"],
        "headroom_gib": best_report["per_gpu_gib"]["headroom"],
        "max_num_seqs": deploy.max_num_seqs,
        "block_size": deploy.block_size,
        "model_max_position_embeddings": model_limit,
        "limited_by_model_context": upper is None and model_limit is not None and best_len >= model_limit,
    }


def summarize_capacity(result: Dict[str, Any]) -> Dict[str, Any]:
    report = result["report"]
    if result["mode"] == "fixed_token_budget":
        return {
            "mode": result["mode"],
            "max_model_len": result["max_model_len"],
            "fit": result["fit"],
            "total_gib": report["per_gpu_gib"]["total"],
            "budget_gib": report["per_gpu_gib"]["usable_budget"],
            "headroom_gib": report["per_gpu_gib"]["headroom"],
            "warning": result["warning"],
        }
    return {
        "mode": result["mode"],
        "max_model_len": result["max_model_len"],
        "max_num_seqs": result["max_num_seqs"],
        "block_size": result["block_size"],
        "model_max_position_embeddings": result.get("model_max_position_embeddings"),
        "limited_by_model_context": result.get("limited_by_model_context", False),
        "total_gib": result["total_gib"],
        "budget_gib": result["budget_gib"],
        "headroom_gib": result["headroom_gib"],
        "kv_cache_gib": report["per_gpu_gib"]["kv_cache"],
        "weights_gib": report["per_gpu_gib"]["weights"],
    }


def print_capacity_human(result: Dict[str, Any], label: str = "Deployment") -> None:
    summary = summarize_capacity(result)
    print(f"{label}: max context capacity")
    print(f"  {'mode':18s} {summary['mode']}")
    print(f"  {'max_model_len':18s} {summary['max_model_len']}")
    if "max_num_seqs" in summary:
        print(f"  {'max_num_seqs':18s} {summary['max_num_seqs']}")
        print(f"  {'block_size':18s} {summary['block_size']}")
    if summary.get("model_max_position_embeddings"):
        print(f"  {'model context cap':18s} {summary['model_max_position_embeddings']}")
    print(f"  {'total/GPU':18s} {summary['total_gib']:.2f} GiB")
    print(f"  {'budget/GPU':18s} {summary['budget_gib']:.2f} GiB")
    print(f"  {'headroom/GPU':18s} {summary['headroom_gib']:.2f} GiB")
    if "kv_cache_gib" in summary:
        print(f"  {'kv_cache/GPU':18s} {summary['kv_cache_gib']:.2f} GiB")
        print(f"  {'weights/GPU':18s} {summary['weights_gib']:.2f} GiB")
    if "warning" in summary:
        print(f"  warning: {summary['warning']}")
    elif summary.get("limited_by_model_context"):
        print("  note: capped by model config max_position_embeddings; use --capacity-upper to run a hardware-only extrapolation.")


def build_pd_capacity_report(
    config: Dict[str, Any],
    prefill_deploy: Deployment,
    decode_deploy: Deployment,
    config_path: Path,
    *,
    upper: int | None = None,
) -> Dict[str, Any]:
    prefill = find_max_context_len(config, prefill_deploy, config_path, upper=upper)
    decode = find_max_context_len(config, decode_deploy, config_path, upper=upper)
    return {
        "scenario": "pd_disaggregated_capacity",
        "prefill": prefill,
        "decode": decode,
        "summary": {
            "prefill": summarize_capacity(prefill),
            "decode": summarize_capacity(decode),
        },
    }
