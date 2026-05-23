from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Deployment:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: bool = False
    dtype: str = "bf16"
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None
    quant_overhead_ratio: float = 0.03
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    gpu_memory_gib: Optional[float] = None
    activation_peak_gib: Optional[float] = None
    activation_factor: float = 0.08
    runtime_overhead_gib: float = 1.5
    lora_rank: int = 0
    lora_count: int = 0
    speculative: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "Deployment":
        fields = {field_name for field_name in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in fields}
        return cls(**values)
