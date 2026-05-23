from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .deployment import Deployment
from .capacity import build_pd_capacity_report, find_max_context_len, print_capacity_human
from .printing import print_human, print_pd_human
from .registry import model_names, resolve_config_target, resolve_model
from .report import build_report
from .scenarios.pd import build_pd_report, is_pd_deployment, pd_deployments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_deployment(args: argparse.Namespace, deployment_path: Path | None = None) -> Deployment:
    data: Dict[str, Any] = {}
    if deployment_path:
        data.update(json.loads(deployment_path.read_text(encoding="utf-8")))
    for key in [
        "tp",
        "dp",
        "pp",
        "dtype",
        "kv_cache_dtype",
        "quantization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "gpu_memory_utilization",
        "gpu_memory_gib",
        "activation_peak_gib",
        "runtime_overhead_gib",
        "lora_rank",
        "lora_count",
    ]:
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.ep:
        data["ep"] = True
    return Deployment.from_mapping(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate vLLM per-GPU VRAM from a model config.")
    parser.add_argument("target", nargs="?", help="Path to config.json or registered model name")
    parser.add_argument("--model", help="Registered model name, e.g. qwen3.5-35b-a3b")
    parser.add_argument("--list-models", action="store_true", help="List registered model names")
    parser.add_argument("--deployment", help="Optional JSON file with deployment settings")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--max-context", action="store_true", help="Find the largest max_model_len that fits in GPU memory")
    parser.add_argument("--capacity-upper", type=int, help="Upper bound for --max-context search")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--ep", action="store_true", help="Enable expert parallelism for MoE expert weights")
    parser.add_argument("--dtype")
    parser.add_argument("--kv-cache-dtype", dest="kv_cache_dtype")
    parser.add_argument("--quantization")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int)
    parser.add_argument("--max-num-seqs", dest="max_num_seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", dest="max_num_batched_tokens", type=int)
    parser.add_argument("--block-size", dest="block_size", type=int)
    parser.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float)
    parser.add_argument("--gpu-memory-gib", dest="gpu_memory_gib", type=float)
    parser.add_argument("--activation-peak-gib", dest="activation_peak_gib", type=float)
    parser.add_argument("--runtime-overhead-gib", dest="runtime_overhead_gib", type=float)
    parser.add_argument("--lora-rank", dest="lora_rank", type=int)
    parser.add_argument("--lora-count", dest="lora_count", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        for name in model_names():
            print(name)
        return

    target = args.model or args.target
    if not target:
        parser.error("provide a config path, a registered model name, or --model")

    config_path = resolve_config_target(target, PROJECT_ROOT)
    if not config_path:
        parser.error(f"unknown config/model target: {target}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    deployment_data: Dict[str, Any] = {}
    deployment_path = Path(args.deployment) if args.deployment else None
    if deployment_path and not deployment_path.is_absolute():
        deployment_path = PROJECT_ROOT / deployment_path

    if not deployment_path:
        entry = resolve_model(target)
        if entry and entry.default_deployment:
            deployment_path = PROJECT_ROOT / entry.default_deployment

    if deployment_path:
        deployment_data = json.loads(deployment_path.read_text(encoding="utf-8"))

    if deployment_data and is_pd_deployment(deployment_data):
        if args.max_context:
            prefill_deploy, decode_deploy = pd_deployments(deployment_data)
            report = build_pd_capacity_report(
                config,
                prefill_deploy,
                decode_deploy,
                config_path,
                upper=args.capacity_upper,
            )
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print("Scenario: PD disaggregated max context capacity")
                print()
                print_capacity_human(report["prefill"], "Prefill nodes")
                print()
                print_capacity_human(report["decode"], "Decode nodes")
            return
        report = build_pd_report(config, deployment_data, config_path)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_pd_human(report)
        return

    deploy = load_deployment(args, deployment_path)
    if args.max_context:
        report = find_max_context_len(config, deploy, config_path, upper=args.capacity_upper)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_capacity_human(report)
        return

    report = build_report(config, deploy, config_path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
