from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vram_estimator import Deployment, build_report  # noqa: E402
from vram_estimator.capacity import build_pd_capacity_report, find_max_context_len  # noqa: E402
from vram_estimator.registry import model_names, resolve_model  # noqa: E402
from vram_estimator.scenarios.pd import build_pd_report  # noqa: E402


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_qwen3_6_moe_active_params() -> None:
    config_path = ROOT / "examples/configs/qwen3_6_35b_a3b/config.json"
    report = build_report(
        read_json("examples/configs/qwen3_6_35b_a3b/config.json"),
        Deployment.from_mapping(read_json("examples/deployments/qwen3_6_35b_a3b/tp2.json")),
        config_path,
    )

    assert report["model"]["model_type"] == "qwen3_5_moe_text"
    assert report["model"]["full_attention_layers"] == 10
    assert round(report["params"]["total"] / 1e9, 3) == 35.015
    assert round(report["active_params"]["per_token"] / 1e9, 3) == 3.0
    assert report["per_gpu_breakdown"]["kv_cache"]["children"]["cache_layout"] == "standard_kv"


def test_qwen3_vl_configs() -> None:
    dense_path = ROOT / "examples/configs/qwen3_8b_vl/config.json"
    dense = build_report(read_json("examples/configs/qwen3_8b_vl/config.json"), Deployment(), dense_path)
    assert dense["model"]["model_type"] == "qwen3_vl_text"
    assert dense["model"]["wrapper_model_type"] == "qwen3_vl"
    assert round(dense["params"]["vision"] / 1e9, 3) == 0.453

    moe_path = ROOT / "examples/configs/qwen3_30b_a3b_vl/config.json"
    moe = build_report(read_json("examples/configs/qwen3_30b_a3b_vl/config.json"), Deployment(), moe_path)
    assert moe["model"]["model_type"] == "qwen3_vl_moe_text"
    assert moe["model"]["wrapper_model_type"] == "qwen3_vl_moe"
    assert round(moe["active_params"]["per_token"] / 1e9, 3) == 3.0


def test_deepseek_v4_flash_compressed_kv() -> None:
    config_path = ROOT / "examples/configs/deepseek_v4_flash_base/config.json"
    report = build_report(
        read_json("examples/configs/deepseek_v4_flash_base/config.json"),
        Deployment.from_mapping(read_json("examples/deployments/deepseek_v4_flash_base/tp16.json")),
        config_path,
    )

    assert report["model"]["model_type"] == "deepseek_v4"
    assert round(report["active_params"]["per_token"] / 1e9, 3) == 13.0
    assert report["per_gpu_breakdown"]["kv_cache"]["children"]["effective_layer_tokens"] < (
        report["per_gpu_breakdown"]["kv_cache"]["children"]["reserved_tokens"] * report["model"]["num_layers"]
    )


def test_glm_5_1_kv_lora_layout() -> None:
    config_path = ROOT / "examples/configs/glm_5_1/config.json"
    report = build_report(
        read_json("examples/configs/glm_5_1/config.json"),
        Deployment.from_mapping(read_json("examples/deployments/glm_5_1/tp16.json")),
        config_path,
    )

    kv = report["per_gpu_breakdown"]["kv_cache"]["children"]
    assert report["model"]["model_type"] == "glm_moe_dsa"
    assert kv["cache_layout"] == "compressed_kv_lora"
    assert kv["cache_dim_per_token_layer"] == 576


def test_pd_disaggregated_report() -> None:
    config_path = ROOT / "examples/configs/qwen3_6_35b_a3b/config.json"
    report = build_pd_report(
        read_json("examples/configs/qwen3_6_35b_a3b/config.json"),
        read_json("examples/deployments/qwen3_6_35b_a3b/pd.json"),
        config_path,
    )

    assert report["scenario"] == "pd_disaggregated"
    assert report["prefill"]["deployment"]["dp"] == 1
    assert report["decode"]["deployment"]["dp"] == 8
    assert report["decode"]["per_gpu_gib"]["kv_cache"] > report["prefill"]["per_gpu_gib"]["kv_cache"]
    assert report["decode"]["per_gpu_gib"]["weights"] < report["prefill"]["per_gpu_gib"]["weights"]
    assert round(report["prefill"]["per_gpu_gib"]["kv_transfer"], 3) == 0.931
    assert report["prefill"]["notes"]["pd_kv_transfer_sizing_method"] == "fixed_kv_buffer_size_bytes"


def test_qwen35_registry() -> None:
    qwen35_names = [name for name in model_names() if name.startswith("qwen3.5-")]
    assert len(qwen35_names) == 21
    assert resolve_model("Qwen/Qwen3.5-35B-A3B").config.endswith("qwen3_5_35b_a3b/config.json")
    assert resolve_model("qwen3_5_27b_fp8").config.endswith("qwen3_5_27b_fp8/config.json")


def test_non_qwen35_registry() -> None:
    assert resolve_model("qwen3-8b").config.endswith("qwen3_8b/config.json")
    assert resolve_model("Qwen/Qwen3-VL-8B-Instruct").config.endswith("qwen3_8b_vl/config.json")
    assert resolve_model("qwen3_32b_vl").config.endswith("qwen3_32b_vl/config.json")
    assert resolve_model("Qwen/Qwen3-VL-30B-A3B-Instruct").config.endswith("qwen3_30b_a3b_vl/config.json")
    assert resolve_model("qwen3.6-35b-a3b-pd").default_deployment.endswith("qwen3_6_35b_a3b/pd.json")
    assert resolve_model("deepseek-ai/DeepSeek-V4-Pro-Base").config.endswith("deepseek_v4_pro_base/config.json")
    assert resolve_model("zai-org/GLM-5.1").config.endswith("glm_5_1/config.json")
    assert resolve_model("glm-5.1-w4a8").config.endswith("glm_5_1_w4a8/config.json")
    assert resolve_model("openpangu-ultra-moe-718b").config.endswith("openpangu_ultra_moe_718b/config.json")


def test_max_context_capacity() -> None:
    config_path = ROOT / "examples/configs/qwen3_6_35b_a3b/config.json"
    deploy = Deployment.from_mapping(read_json("examples/deployments/qwen3_6_35b_a3b/tp2.json"))
    result = find_max_context_len(
        read_json("examples/configs/qwen3_6_35b_a3b/config.json"),
        deploy,
        config_path,
        upper=262144,
    )

    assert result["mode"] == "max_model_len"
    assert result["max_model_len"] == 136843
    assert result["fit"] is True
    assert result["model_max_position_embeddings"] == 262144


def test_max_context_defaults_to_model_context_limit() -> None:
    config_path = ROOT / "examples/configs/qwen3_5_35b_a3b/config.json"
    deploy = Deployment.from_mapping(read_json("examples/deployments/qwen3_5_35b_a3b/tp4.json"))
    result = find_max_context_len(
        read_json("examples/configs/qwen3_5_35b_a3b/config.json"),
        deploy,
        config_path,
    )

    assert result["max_model_len"] == 262144
    assert result["limited_by_model_context"] is True


def test_pd_max_context_capacity() -> None:
    config_path = ROOT / "examples/configs/qwen3_6_35b_a3b/config.json"
    pd = read_json("examples/deployments/qwen3_6_35b_a3b/pd.json")
    from vram_estimator.scenarios.pd import pd_deployments

    prefill, decode = pd_deployments(pd)
    result = build_pd_capacity_report(
        read_json("examples/configs/qwen3_6_35b_a3b/config.json"),
        prefill,
        decode,
        config_path,
        upper=262144,
    )

    assert result["scenario"] == "pd_disaggregated_capacity"
    assert result["prefill"]["mode"] == "fixed_token_budget"
    assert result["decode"]["max_model_len"] == 262144


if __name__ == "__main__":
    test_qwen3_6_moe_active_params()
    test_qwen3_vl_configs()
    test_deepseek_v4_flash_compressed_kv()
    test_glm_5_1_kv_lora_layout()
    test_pd_disaggregated_report()
    test_qwen35_registry()
    test_non_qwen35_registry()
    test_max_context_capacity()
    test_max_context_defaults_to_model_context_limit()
    test_pd_max_context_capacity()
    print("estimator smoke tests passed")
