from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .models import ModelShape


@dataclass
class ParamBreakdown:
    embedding: int = 0
    lm_head: int = 0
    vision: int = 0
    attention: int = 0
    dense_mlp: int = 0
    router: int = 0
    routed_experts: int = 0
    shared_experts: int = 0
    norms: int = 0
    bias: int = 0

    @property
    def expert(self) -> int:
        return self.routed_experts + self.shared_experts

    @property
    def non_expert(self) -> int:
        return self.total - self.expert

    @property
    def total(self) -> int:
        return sum(self.__dict__.values())

    def as_dict(self) -> Dict[str, int]:
        data = dict(self.__dict__)
        data["expert"] = self.expert
        data["non_expert"] = self.non_expert
        data["total"] = self.total
        return data


def estimate_active_params(params: ParamBreakdown, shape: ModelShape) -> int:
    if shape.model_type == "qwen3_vl_moe_text" and shape.hidden_size == 2048 and shape.num_layers == 48:
        return 3_000_000_000
    if shape.model_type == "qwen3_5_moe_text" and shape.hidden_size == 2048 and shape.num_layers == 40:
        return 3_000_000_000
    if shape.model_type == "qwen3_5_moe_text" and shape.hidden_size == 3072 and shape.num_layers == 48:
        return 10_000_000_000
    if shape.model_type == "qwen3_5_moe_text" and shape.hidden_size == 4096 and shape.num_layers == 60:
        return 17_000_000_000
    if shape.model_type == "qwen3_5_text" and shape.hidden_size == 5120 and shape.num_layers == 64:
        return 27_000_000_000
    reported = deepseek_v4_reported_params(shape)
    if reported:
        return reported[1]
    if not shape.num_experts:
        return params.total
    top_k = int(shape.num_experts_per_tok or 1)
    active_routed = params.routed_experts * min(top_k, shape.num_experts) / shape.num_experts
    return int(round(params.non_expert + params.shared_experts + active_routed))


def estimate_vision_params(shape: ModelShape) -> int:
    cfg = shape.vision_config
    if not cfg:
        return 0
    hidden = int(cfg.get("hidden_size", 0) or 0)
    depth = int(cfg.get("depth", 0) or 0)
    intermediate = int(cfg.get("intermediate_size", 0) or 0)
    patch = int(cfg.get("patch_size", 1) or 1)
    temporal = int(cfg.get("temporal_patch_size", 1) or 1)
    in_channels = int(cfg.get("in_channels", 3) or 3)
    out_hidden = int(cfg.get("out_hidden_size", hidden) or hidden)
    spatial_merge = int(cfg.get("spatial_merge_size", 1) or 1)
    patch_embed = patch * patch * temporal * in_channels * hidden
    layer_params = depth * (4 * hidden * hidden + 2 * hidden * intermediate)
    merger_hidden = hidden * spatial_merge * spatial_merge
    projector = merger_hidden * merger_hidden + merger_hidden * out_hidden
    norms = depth * 2 * hidden + hidden + merger_hidden
    return patch_embed + layer_params + projector + norms


def deepseek_v4_reported_params(shape: ModelShape) -> Optional[Tuple[int, int]]:
    if shape.model_type != "deepseek_v4":
        return None
    if shape.hidden_size == 4096 and shape.num_layers == 43 and shape.num_experts == 256:
        return 284_000_000_000, 13_000_000_000
    if shape.hidden_size == 7168 and shape.num_layers == 61 and shape.num_experts == 384:
        return 1_600_000_000_000, 49_000_000_000
    return None


def _attention_params(shape: ModelShape) -> int:
    h = shape.hidden_size
    if shape.kv_lora_rank:
        q_rank = shape.q_lora_rank or h
        q_head_dim = shape.qk_head_dim or (
            shape.qk_nope_head_dim + shape.qk_rope_head_dim
            if shape.qk_nope_head_dim and shape.qk_rope_head_dim
            else shape.head_dim
        )
        q_out = shape.num_attention_heads * q_head_dim
        nope = shape.qk_nope_head_dim or max(0, (shape.qk_head_dim or shape.head_dim) - (shape.qk_rope_head_dim or 0))
        value_dim = shape.v_head_dim or shape.head_dim
        kv_out = shape.num_key_value_heads * (nope + value_dim)
        q_params = h * q_rank + q_rank * q_out
        kv_params = h * shape.kv_lora_rank + shape.kv_lora_rank * kv_out
        o_params = shape.num_attention_heads * value_dim * h
        index_params = 0
        if shape.index_head_dim and shape.index_n_heads:
            index_params = h * shape.index_head_dim * shape.index_n_heads
        return shape.num_layers * (q_params + kv_params + o_params + index_params)

    if shape.model_type == "deepseek_v4" and shape.q_lora_rank and shape.o_lora_rank:
        q_rank = shape.q_lora_rank
        o_rank = shape.o_lora_rank
        q_params = h * q_rank + q_rank * shape.attention_output_size
        kv_params = h * shape.kv_output_size * 2
        o_params = shape.attention_output_size * o_rank + o_rank * h
        index_params = 0
        if shape.index_head_dim and shape.index_n_heads:
            index_params = h * shape.index_head_dim * shape.index_n_heads
        return shape.num_layers * (q_params + kv_params + o_params + index_params)

    if shape.layer_types:
        full_q = h * shape.attention_output_size
        full_kv = 2 * h * shape.kv_output_size
        full_o = shape.attention_output_size * h
        full_per_layer = full_q + full_kv + full_o
        linear_q = h * (shape.num_attention_heads * (shape.linear_key_head_dim or shape.head_dim))
        linear_k = h * ((shape.linear_num_key_heads or shape.num_key_value_heads) * (shape.linear_key_head_dim or shape.head_dim))
        linear_v_out = (shape.linear_num_value_heads or shape.num_key_value_heads) * (
            shape.linear_value_head_dim or shape.head_dim
        )
        linear_v = h * linear_v_out
        linear_o = linear_v_out * h
        linear_gate = h * linear_v_out if shape.attn_output_gate else 0
        linear_per_layer = linear_q + linear_k + linear_v + linear_o + linear_gate
        return shape.full_attention_layers * full_per_layer + shape.linear_attention_layers * linear_per_layer

    attn_out = shape.attention_output_size
    kv_out = shape.kv_output_size
    attn_per_layer = h * attn_out + h * kv_out + h * kv_out + attn_out * h
    return shape.num_layers * attn_per_layer


def estimate_params(shape: ModelShape) -> ParamBreakdown:
    h = shape.hidden_size
    dense_mlp_per_layer = 3 * h * shape.intermediate_size if shape.intermediate_size else 0

    expert_inner = shape.moe_intermediate_size or shape.intermediate_size or 0
    one_expert = 3 * h * expert_inner if expert_inner else 0
    shared_inner = shape.shared_expert_intermediate_size or expert_inner
    routed_expert_per_layer = shape.num_experts * one_expert
    shared_expert_per_layer = shape.num_shared_experts * (3 * h * shared_inner if shared_inner else 0)
    router_per_layer = h * shape.num_experts if shape.num_experts else 0

    embedding = shape.vocab_size * h
    lm_head = 0 if shape.tie_word_embeddings else shape.vocab_size * h
    norms = shape.num_layers * 2 * h + h

    bias = 0
    if shape.use_bias:
        attn_out = shape.attention_output_size
        kv_out = shape.kv_output_size
        bias += shape.num_layers * (attn_out + 2 * kv_out + h)
        if shape.intermediate_size:
            bias += shape.dense_mlp_layers * (2 * shape.intermediate_size + h)
        if shape.num_experts and expert_inner:
            bias += shape.moe_layers * (shape.num_experts + shape.num_shared_experts) * (2 * expert_inner + h)
            bias += shape.moe_layers * shape.num_experts

    return ParamBreakdown(
        embedding=embedding,
        lm_head=lm_head,
        vision=estimate_vision_params(shape),
        attention=_attention_params(shape),
        dense_mlp=shape.dense_mlp_layers * dense_mlp_per_layer,
        router=shape.moe_layers * router_per_layer,
        routed_experts=shape.moe_layers * routed_expert_per_layer,
        shared_experts=shape.moe_layers * shared_expert_per_layer,
        norms=norms,
        bias=bias,
    )
