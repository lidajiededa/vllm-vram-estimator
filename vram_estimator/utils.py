from __future__ import annotations

import math
from typing import Any, Dict, Iterable


BYTES_BY_DTYPE = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "half": 2.0,
    "fp8": 1.0,
    "float8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "w4a8": 0.5,
    "modelslimw4a8": 0.5,
    "ascendw4a8": 0.5,
    "awq": 0.5,
    "gptq": 0.5,
}

GIB = 1024**3


def pick(config: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return default


def ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor))


def as_gib(num_bytes: float) -> float:
    return num_bytes / GIB


def pct(part: float, total: float) -> float:
    return 0.0 if total == 0 else part / total * 100.0


def parse_dtype_bytes(dtype: str) -> float:
    key = dtype.lower().replace("-", "").replace("_", "")
    normalized = {
        "float32": "fp32",
        "bfloat16": "bf16",
        "float16": "fp16",
        "float8": "fp8",
    }.get(key, key)
    if normalized not in BYTES_BY_DTYPE:
        raise ValueError(f"Unsupported dtype '{dtype}'. Known: {', '.join(sorted(BYTES_BY_DTYPE))}")
    return BYTES_BY_DTYPE[normalized]
