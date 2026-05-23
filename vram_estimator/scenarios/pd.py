from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from ..deployment import Deployment
from ..report import build_report
from ..utils import GIB, as_gib, ceil_div, pct


PD_SECTION_KEYS = {"prefill", "decode", "pd", "scenario", "kind", "description", "kv_transfer"}


def is_pd_deployment(data: Dict[str, Any]) -> bool:
    return isinstance(data.get("prefill"), dict) and isinstance(data.get("decode"), dict)


def _merge_role_deployment(data: Dict[str, Any], role: str) -> Dict[str, Any]:
    common = {k: v for k, v in data.items() if k not in PD_SECTION_KEYS}
    role_data = dict(data.get(role, {}))
    merged = {**common, **role_data}
    merged["role"] = role
    return merged


def pd_deployments(deployment_data: Dict[str, Any]) -> tuple[Deployment, Deployment]:
    prefill_deploy = Deployment.from_mapping(_merge_role_deployment(deployment_data, "prefill"))
    decode_deploy = Deployment.from_mapping(_merge_role_deployment(deployment_data, "decode"))
    return prefill_deploy, decode_deploy


def _transfer_config(deployment_data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deployment_data.get("kv_transfer")
    if cfg is False:
        return {"enabled": False}
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("kv_transfer must be an object or false")
    merged = {
        "enabled": True,
        "mode": "vllm_buffer",
        "direction": "p2d",
        "kv_buffer_size_bytes": 1_000_000_000,
        "buffer_factor": 1.0,
        "comm_overhead_gib": 0.0,
    }
    merged.update(cfg)
    return merged


def _role_transfer_tokens(cfg: Dict[str, Any], role: str, prefill_deploy: Deployment) -> int:
    role_key = f"{role}_tokens"
    if cfg.get(role_key) is not None:
        return int(cfg[role_key])
    if cfg.get("tokens") is not None:
        return int(cfg["tokens"])
    return int(prefill_deploy.max_num_batched_tokens or (prefill_deploy.max_model_len * prefill_deploy.max_num_seqs))


def _role_buffer_factor(cfg: Dict[str, Any], role: str) -> float:
    return float(cfg.get(f"{role}_buffer_factor", cfg.get("buffer_factor", 2.0)))


def _role_comm_overhead_gib(cfg: Dict[str, Any], role: str) -> float:
    return float(cfg.get(f"{role}_comm_overhead_gib", cfg.get("comm_overhead_gib", 0.0)))


def _add_pd_extra_memory(
    report: Dict[str, Any],
    *,
    role: str,
    cfg: Dict[str, Any],
    prefill_deploy: Deployment,
) -> Dict[str, Any]:
    report = deepcopy(report)
    if not cfg.get("enabled", True):
        return report

    block_size = int(report["notes"]["kv_block_size"])
    tokens = ceil_div(_role_transfer_tokens(cfg, role, prefill_deploy), block_size) * block_size
    bytes_per_token = float(report["notes"]["kv_bytes_per_token_per_gpu"])
    factor = _role_buffer_factor(cfg, role)
    mode = str(cfg.get("mode", "vllm_buffer")).lower()
    sizing_method = "payload_tokens"
    if cfg.get(f"{role}_kv_buffer_size_gib") is not None:
        transfer_bytes = float(cfg[f"{role}_kv_buffer_size_gib"]) * GIB
        sizing_method = "fixed_role_kv_buffer_size_gib"
    elif cfg.get("kv_buffer_size_gib") is not None:
        transfer_bytes = float(cfg["kv_buffer_size_gib"]) * GIB
        sizing_method = "fixed_kv_buffer_size_gib"
    elif cfg.get(f"{role}_kv_buffer_size_bytes") is not None:
        transfer_bytes = float(cfg[f"{role}_kv_buffer_size_bytes"])
        sizing_method = "fixed_role_kv_buffer_size_bytes"
    elif cfg.get("kv_buffer_size_bytes") is not None and mode in {"vllm", "vllm_buffer", "fixed"}:
        transfer_bytes = float(cfg["kv_buffer_size_bytes"])
        sizing_method = "fixed_kv_buffer_size_bytes"
    else:
        transfer_bytes = tokens * bytes_per_token * factor
    comm_bytes = _role_comm_overhead_gib(cfg, role) * GIB
    extra_bytes = transfer_bytes + comm_bytes

    vram = report["per_gpu_gib"]
    total_bytes = (vram["total"] * GIB) + extra_bytes
    vram["kv_transfer"] = as_gib(transfer_bytes)
    vram["comm_buffer"] = as_gib(comm_bytes)
    vram["total"] = as_gib(total_bytes)
    if vram["usable_budget"] is not None:
        vram["headroom"] = vram["usable_budget"] - vram["total"]
        report["fit"] = vram["total"] <= vram["usable_budget"]

    breakdown = report["per_gpu_breakdown"]
    breakdown["kv_transfer"] = {
        "gib": as_gib(transfer_bytes),
        "percent_of_total": pct(transfer_bytes, total_bytes),
        "description": "PD KV transfer staging buffer for connector send/recv, estimated from local KV layout.",
        "children": {
            "tokens": tokens,
            "block_size": block_size,
            "buffer_factor": factor,
                "bytes_per_token_per_gpu": bytes_per_token,
                "direction": str(cfg.get("direction", "p2d")),
                "mode": mode,
                "sizing_method": sizing_method,
                "note": "Uses this role's per-GPU KV bytes/token, so asymmetric TP/DP is reflected.",
            },
    }
    breakdown["comm_buffer"] = {
        "gib": as_gib(comm_bytes),
        "percent_of_total": pct(comm_bytes, total_bytes),
        "description": "PD communication workspace reserve for connector/NIC/HCCL/NCCL staging.",
    }
    for item in breakdown.values():
        if isinstance(item, dict) and "gib" in item:
            item["percent_of_total"] = pct(item["gib"] * GIB, total_bytes)

    report["notes"].update(
        {
            "pd_kv_transfer_enabled": True,
            "pd_kv_transfer_tokens": tokens,
            "pd_kv_transfer_buffer_factor": factor,
            "pd_kv_transfer_mode": mode,
            "pd_kv_transfer_sizing_method": sizing_method,
            "pd_kv_transfer_gib": as_gib(transfer_bytes),
            "pd_comm_buffer_gib": as_gib(comm_bytes),
            "pd_kv_transfer_note": (
                "Estimated explicit PD connector buffer. Set kv_transfer=false to disable, or tune tokens/"
                "prefill_tokens/decode_tokens and buffer_factor per deployment."
            ),
        }
    )
    return report


def build_pd_report(config: Dict[str, Any], deployment_data: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    prefill_deploy, decode_deploy = pd_deployments(deployment_data)
    prefill = build_report(config, prefill_deploy, config_path)
    decode = build_report(config, decode_deploy, config_path)
    kv_transfer = _transfer_config(deployment_data)
    prefill = _add_pd_extra_memory(prefill, role="prefill", cfg=kv_transfer, prefill_deploy=prefill_deploy)
    decode = _add_pd_extra_memory(decode, role="decode", cfg=kv_transfer, prefill_deploy=prefill_deploy)

    return {
        "scenario": "pd_disaggregated",
        "description": "Prefill and decode are estimated as separate vLLM deployments with independent parallelism and budgets.",
        "prefill": prefill,
        "decode": decode,
        "summary": {
            "prefill_total_gib_per_gpu": prefill["per_gpu_gib"]["total"],
            "decode_total_gib_per_gpu": decode["per_gpu_gib"]["total"],
            "prefill_kv_gib_per_gpu": prefill["per_gpu_gib"]["kv_cache"],
            "decode_kv_gib_per_gpu": decode["per_gpu_gib"]["kv_cache"],
            "prefill_kv_transfer_gib_per_gpu": prefill["per_gpu_gib"].get("kv_transfer", 0.0),
            "decode_kv_transfer_gib_per_gpu": decode["per_gpu_gib"].get("kv_transfer", 0.0),
            "prefill_comm_buffer_gib_per_gpu": prefill["per_gpu_gib"].get("comm_buffer", 0.0),
            "decode_comm_buffer_gib_per_gpu": decode["per_gpu_gib"].get("comm_buffer", 0.0),
            "prefill_weight_gib_per_gpu": prefill["per_gpu_gib"]["weights"],
            "decode_weight_gib_per_gpu": decode["per_gpu_gib"]["weights"],
            "prefill_fit": prefill["fit"],
            "decode_fit": decode["fit"],
            "prefill_tp_dp_pp_ep": {
                "tp": prefill["deployment"]["tp"],
                "dp": prefill["deployment"]["dp"],
                "pp": prefill["deployment"]["pp"],
                "ep": prefill["deployment"]["ep"],
            },
            "decode_tp_dp_pp_ep": {
                "tp": decode["deployment"]["tp"],
                "dp": decode["deployment"]["dp"],
                "pp": decode["deployment"]["pp"],
                "ep": decode["deployment"]["ep"],
            },
        },
        "notes": [
            "Prefill nodes are usually configured around prompt throughput and activation/KV production.",
            "Decode nodes are usually configured around resident KV capacity and may use higher DP.",
            "PD KV transfer and communication buffers are modeled explicitly as kv_transfer and comm_buffer.",
        ],
    }
