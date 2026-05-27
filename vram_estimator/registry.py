from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class ModelEntry:
    name: str
    config: str
    default_deployment: Optional[str] = None
    hf_id: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "qwen2-7b": ModelEntry("qwen2-7b", "examples/configs/qwen2_7b/config.json", hf_id="Qwen/Qwen2-7B"),
    "qwen3-8b": ModelEntry("qwen3-8b", "examples/configs/qwen3_8b/config.json", hf_id="Qwen/Qwen3-8B"),
    "qwen3-8b-vl": ModelEntry("qwen3-8b-vl", "examples/configs/qwen3_8b_vl/config.json", hf_id="Qwen/Qwen3-VL-8B-Instruct"),
    "qwen3-32b-vl": ModelEntry("qwen3-32b-vl", "examples/configs/qwen3_32b_vl/config.json", hf_id="Qwen/Qwen3-VL-32B-Instruct"),
    "qwen3-30b-a3b": ModelEntry("qwen3-30b-a3b", "examples/configs/qwen3_30b_a3b/config.json", hf_id="Qwen/Qwen3-30B-A3B"),
    "qwen3-30b-a3b-vl": ModelEntry("qwen3-30b-a3b-vl", "examples/configs/qwen3_30b_a3b_vl/config.json", hf_id="Qwen/Qwen3-VL-30B-A3B-Instruct"),
    "qwen3.6-27b": ModelEntry("qwen3.6-27b", "examples/configs/qwen3_6_27b/config.json", "examples/deployments/qwen3_6_27b/tp4.json"),
    "qwen3.6-35b-a3b": ModelEntry("qwen3.6-35b-a3b", "examples/configs/qwen3_6_35b_a3b/config.json", "examples/deployments/qwen3_6_35b_a3b/tp2.json"),
    "qwen3.6-35b-a3b-pd": ModelEntry("qwen3.6-35b-a3b-pd", "examples/configs/qwen3_6_35b_a3b/config.json", "examples/deployments/qwen3_6_35b_a3b/pd.json"),
    "glm-5.1": ModelEntry("glm-5.1", "examples/configs/glm_5_1/config.json", "examples/deployments/glm_5_1/tp16.json"),
    "glm-5.1-tp16": ModelEntry("glm-5.1-tp16", "examples/configs/glm_5_1/config.json", "examples/deployments/glm_5_1/tp16.json"),
    "glm-5.1-fp8": ModelEntry("glm-5.1-fp8", "examples/configs/glm_5_1_fp8/config.json"),
    "glm-5.1-w8a8": ModelEntry("glm-5.1-w8a8", "examples/configs/glm_5_1_fp8/config.json"),
    "glm-5.1-w4a8": ModelEntry("glm-5.1-w4a8", "examples/configs/glm_5_1_w4a8/config.json", "examples/deployments/glm_5_1_w4a8/tp8.json", "Eco-Tech/GLM-5.1-w4a8"),
    "deepseek-moe-like": ModelEntry("deepseek-moe-like", "examples/configs/deepseek_moe_like/config.json", "examples/deployments/deepseek_moe_like/ep.json"),
    "deepseek-v4-flash-base": ModelEntry("deepseek-v4-flash-base", "examples/configs/deepseek_v4_flash_base/config.json", "examples/deployments/deepseek_v4_flash_base/tp16.json"),
    "deepseek-v4-pro-base": ModelEntry("deepseek-v4-pro-base", "examples/configs/deepseek_v4_pro_base/config.json", "examples/deployments/deepseek_v4_pro_base/tp64.json"),
    "openpangu-ultra-moe-718b": ModelEntry("openpangu-ultra-moe-718b", "examples/configs/openpangu_ultra_moe_718b/config.json", "examples/deployments/openpangu_ultra_moe_718b/tp32.json", "FreedomIntelligence/openPangu-Ultra-MoE-718B"),
    "pangu-v2-moe": ModelEntry("pangu-v2-moe", "examples/configs/pangu_v2_moe/config.json", "examples/deployments/pangu_v2_moe/tp32.json"),
}


def _qwen35(file_stem: str, hf_suffix: str, deployment: Optional[str] = None) -> ModelEntry:
    name = hf_suffix.lower()
    return ModelEntry(
        name=f"qwen3.5-{name}",
        config=f"examples/configs/qwen3_5_{file_stem}/config.json",
        default_deployment=deployment,
        hf_id=f"Qwen/Qwen3.5-{hf_suffix}",
    )


for stem, suffix, deployment in [
    ("0_8b", "0.8B", None),
    ("0_8b_base", "0.8B-Base", None),
    ("2b", "2B", None),
    ("2b_base", "2B-Base", None),
    ("4b", "4B", None),
    ("4b_base", "4B-Base", None),
    ("9b", "9B", None),
    ("9b_base", "9B-Base", None),
    ("27b", "27B", None),
    ("27b_fp8", "27B-FP8", None),
    ("27b_gptq_int4", "27B-GPTQ-Int4", None),
    ("35b_a3b", "35B-A3B", "examples/deployments/qwen3_5_35b_a3b/tp4.json"),
    ("35b_a3b_base", "35B-A3B-Base", "examples/deployments/qwen3_5_35b_a3b/tp4.json"),
    ("35b_a3b_fp8", "35B-A3B-FP8", "examples/deployments/qwen3_5_35b_a3b/tp4.json"),
    ("35b_a3b_gptq_int4", "35B-A3B-GPTQ-Int4", "examples/deployments/qwen3_5_35b_a3b/tp4.json"),
    ("122b_a10b", "122B-A10B", None),
    ("122b_a10b_fp8", "122B-A10B-FP8", None),
    ("122b_a10b_gptq_int4", "122B-A10B-GPTQ-Int4", None),
    ("397b_a17b", "397B-A17B", None),
    ("397b_a17b_fp8", "397B-A17B-FP8", None),
    ("397b_a17b_gptq_int4", "397B-A17B-GPTQ-Int4", None),
]:
    entry = _qwen35(stem, suffix, deployment)
    MODEL_REGISTRY[entry.name] = entry


def _loose_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


ALIASES = {}
for key, entry in MODEL_REGISTRY.items():
    ALIASES[key.replace(".", "_")] = key
    ALIASES[key.replace(".", "-")] = key
    ALIASES[_loose_key(key)] = key
    if entry.hf_id:
        hf_key = entry.hf_id.lower()
        for prefix in ("qwen/", "zai-org/", "deepseek-ai/"):
            hf_key = hf_key.removeprefix(prefix)
        ALIASES[hf_key] = key
        ALIASES[_loose_key(hf_key)] = key

ALIASES.update(
    {
        "qwen3-vl-8b": "qwen3-8b-vl",
        "qwen3_vl_8b": "qwen3-8b-vl",
        "qwen3-vl-32b": "qwen3-32b-vl",
        "qwen3_vl_32b": "qwen3-32b-vl",
        "qwen3-vl-30b-a3b": "qwen3-30b-a3b-vl",
        "qwen3_vl_30b_a3b": "qwen3-30b-a3b-vl",
    }
)


def model_names() -> Iterable[str]:
    return sorted(MODEL_REGISTRY)


def resolve_model(name: str) -> Optional[ModelEntry]:
    normalized = name.strip().lower()
    normalized = normalized.removeprefix("qwen/")
    normalized = normalized.removeprefix("zai-org/")
    normalized = normalized.removeprefix("deepseek-ai/")
    key = normalized if normalized in MODEL_REGISTRY else ALIASES.get(normalized) or ALIASES.get(_loose_key(normalized))
    return MODEL_REGISTRY.get(key) if key else None


def resolve_config_target(target: str, root: Path) -> Optional[Path]:
    path = Path(target)
    if path.exists():
        return path
    if not path.is_absolute() and (root / path).exists():
        return root / path
    entry = resolve_model(target)
    return root / entry.config if entry else None
