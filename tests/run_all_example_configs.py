from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "vllm_vram_estimator.py"


CASES = [
    (
        "deepseek_moe_like/config.json",
        ["--deployment", "examples/deployments/deepseek_moe_like/ep.json"],
    ),
    (
        "deepseek_v4_flash_base/config.json",
        ["--deployment", "examples/deployments/deepseek_v4_flash_base/tp16.json"],
    ),
    (
        "deepseek_v4_pro_base/config.json",
        ["--deployment", "examples/deployments/deepseek_v4_pro_base/tp64.json"],
    ),
    (
        "glm_5_1/config.json",
        ["--deployment", "examples/deployments/glm_5_1/tp16.json"],
    ),
    (
        "glm_5_1_w4a8/config.json",
        ["--deployment", "examples/deployments/glm_5_1_w4a8/tp8.json"],
    ),
    (
        "openpangu_ultra_moe_718b/config.json",
        ["--deployment", "examples/deployments/openpangu_ultra_moe_718b/tp32.json"],
    ),
    (
        "qwen2_7b/config.json",
        ["--tp", "1", "--dtype", "bf16", "--max-model-len", "32768", "--max-num-seqs", "16", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_8b/config.json",
        ["--tp", "1", "--dtype", "bf16", "--max-model-len", "40960", "--max-num-seqs", "8", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_8b_vl/config.json",
        ["--tp", "1", "--dtype", "bf16", "--max-model-len", "32768", "--max-num-seqs", "1", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_32b_vl/config.json",
        ["--tp", "4", "--dtype", "bf16", "--max-model-len", "32768", "--max-num-seqs", "1", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_30b_a3b/config.json",
        ["--tp", "4", "--dp", "1", "--ep", "--dtype", "bf16", "--max-model-len", "40960", "--max-num-seqs", "4", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_30b_a3b_vl/config.json",
        ["--tp", "4", "--dp", "1", "--ep", "--dtype", "bf16", "--max-model-len", "32768", "--max-num-seqs", "1", "--gpu-memory-gib", "80"],
    ),
    (
        "qwen3_6_27b/config.json",
        ["--deployment", "examples/deployments/qwen3_6_27b/tp4.json"],
    ),
    (
        "qwen3_6_35b_a3b/config.json",
        ["--deployment", "examples/deployments/qwen3_6_35b_a3b/tp2.json"],
    ),
    (
        "qwen3_6_35b_a3b/config.json",
        ["--deployment", "examples/deployments/qwen3_6_35b_a3b/pd.json"],
    ),
]

QWEN35_CASES = [
    (path.parent.name, path.relative_to(ROOT / "examples/configs").as_posix())
    for path in sorted((ROOT / "examples/configs").glob("qwen3_5_*/config.json"))
]


def run_case(config_name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(CLI), f"examples/configs/{config_name}", *args, "--json"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{config_name} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    report = json.loads(proc.stdout)
    if report.get("scenario") == "pd_disaggregated":
        assert report["prefill"]["per_gpu_gib"]["total"] > 0
        assert report["decode"]["per_gpu_gib"]["total"] > 0
        label = f"{config_name} [PD]"
        total = f"P={report['prefill']['per_gpu_gib']['total']:.2f}GiB D={report['decode']['per_gpu_gib']['total']:.2f}GiB"
    else:
        assert report["per_gpu_gib"]["total"] > 0
        label = config_name
        total = f"{report['per_gpu_gib']['total']:.2f}GiB"
    print(f"ok {label}: {total}")


def main() -> None:
    for config_name, args in CASES:
        run_case(config_name, args)
    for label, config_path in QWEN35_CASES:
        run_case(
            config_path,
            ["--tp", "4", "--dp", "1", "--ep", "--dtype", "bf16", "--max-model-len", "32768", "--max-num-seqs", "1", "--gpu-memory-gib", "80"],
        )
    print(f"validated {len(CASES) + len(QWEN35_CASES)} example runs")


if __name__ == "__main__":
    main()
