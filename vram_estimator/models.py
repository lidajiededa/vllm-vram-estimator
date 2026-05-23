from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import pick


@dataclass
class ModelShape:
    model_type: str
    wrapper_model_type: Optional[str] = None
    language_model_only: bool = True
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    max_position_embeddings: Optional[int] = None
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    num_experts: int = 0
    num_shared_experts: int = 0
    shared_expert_intermediate_size: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    decoder_sparse_step: Optional[int] = None
    mlp_only_layers: Optional[List[int]] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    o_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_n_heads: Optional[int] = None
    num_nextn_predict_layers: int = 0
    expert_dtype: Optional[str] = None
    config_quant_method: Optional[str] = None
    quant_modules_to_not_convert: List[str] = field(default_factory=list)
    compress_ratios: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    attn_output_gate: bool = False
    vision_config: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    use_bias: bool = False
    raw_architectures: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelShape":
        if isinstance(config.get("text_config"), dict):
            text_config = dict(config["text_config"])
            text_config.setdefault("architectures", config.get("architectures", []))
            text_config["_wrapper_model_type"] = config.get("model_type")
            text_config["_language_model_only"] = bool(config.get("language_model_only", False))
            text_config["_vision_config"] = config.get("vision_config")
            text_config.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", False))
            text_config.setdefault("quantization_config", config.get("quantization_config"))
            config = text_config

        hidden_size = int(pick(config, ["hidden_size", "n_embd", "d_model"]))
        num_layers = int(pick(config, ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]))
        num_attention_heads = int(pick(config, ["num_attention_heads", "n_head", "num_heads"]))
        num_key_value_heads = int(
            pick(config, ["num_key_value_heads", "n_kv_heads", "num_kv_heads"], num_attention_heads)
        )
        head_dim = int(pick(config, ["head_dim", "qk_rope_head_dim"], hidden_size // num_attention_heads))
        vocab_size = int(pick(config, ["vocab_size", "padded_vocab_size"]))
        max_position_embeddings = pick(config, ["max_position_embeddings", "n_positions", "seq_length"], None)

        num_experts = int(
            pick(
                config,
                [
                    "num_experts",
                    "n_routed_experts",
                    "num_local_experts",
                    "moe_num_experts",
                    "num_experts_per_layer",
                ],
                0,
            )
        )
        num_shared_experts = int(pick(config, ["n_shared_experts", "num_shared_experts"], 0) or 0)
        shared_expert_intermediate_size = pick(config, ["shared_expert_intermediate_size"], None)
        if shared_expert_intermediate_size is not None and num_shared_experts == 0:
            num_shared_experts = 1

        intermediate_size = pick(config, ["intermediate_size", "ffn_hidden_size", "n_inner"])
        moe_intermediate_size = pick(
            config,
            ["moe_intermediate_size", "moe_ffn_hidden_size", "moe_hidden_size", "expert_intermediate_size"],
            None,
        )
        if moe_intermediate_size is None and num_experts:
            moe_intermediate_size = intermediate_size

        return cls(
            model_type=str(config.get("model_type", "unknown")),
            wrapper_model_type=config.get("_wrapper_model_type"),
            language_model_only=bool(config.get("_language_model_only", True)),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_position_embeddings=(
                int(max_position_embeddings) if max_position_embeddings is not None else None
            ),
            intermediate_size=int(intermediate_size) if intermediate_size is not None else None,
            moe_intermediate_size=int(moe_intermediate_size) if moe_intermediate_size is not None else None,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=(
                int(shared_expert_intermediate_size) if shared_expert_intermediate_size is not None else None
            ),
            num_experts_per_tok=pick(config, ["num_experts_per_tok", "moe_top_k", "top_k"], None),
            first_k_dense_replace=int(pick(config, ["first_k_dense_replace"], 0) or 0),
            moe_layer_freq=int(pick(config, ["moe_layer_freq"], 1) or 1),
            decoder_sparse_step=(
                int(config["decoder_sparse_step"]) if config.get("decoder_sparse_step") is not None else None
            ),
            mlp_only_layers=(
                [int(layer) for layer in config["mlp_only_layers"]]
                if config.get("mlp_only_layers") is not None
                else None
            ),
            q_lora_rank=(int(config["q_lora_rank"]) if config.get("q_lora_rank") is not None else None),
            kv_lora_rank=(int(config["kv_lora_rank"]) if config.get("kv_lora_rank") is not None else None),
            o_lora_rank=(int(config["o_lora_rank"]) if config.get("o_lora_rank") is not None else None),
            qk_head_dim=(int(config["qk_head_dim"]) if config.get("qk_head_dim") is not None else None),
            qk_nope_head_dim=(int(config["qk_nope_head_dim"]) if config.get("qk_nope_head_dim") is not None else None),
            qk_rope_head_dim=(int(config["qk_rope_head_dim"]) if config.get("qk_rope_head_dim") is not None else None),
            v_head_dim=(int(config["v_head_dim"]) if config.get("v_head_dim") is not None else None),
            index_head_dim=(int(config["index_head_dim"]) if config.get("index_head_dim") is not None else None),
            index_n_heads=(int(config["index_n_heads"]) if config.get("index_n_heads") is not None else None),
            num_nextn_predict_layers=int(config.get("num_nextn_predict_layers", 0) or 0),
            expert_dtype=config.get("expert_dtype"),
            config_quant_method=(
                config.get("quantization_config", {}).get("quant_method")
                if isinstance(config.get("quantization_config"), dict)
                else None
            ),
            quant_modules_to_not_convert=(
                list(config.get("quantization_config", {}).get("modules_to_not_convert", []))
                if isinstance(config.get("quantization_config"), dict)
                else []
            ),
            compress_ratios=(
                [int(ratio) for ratio in config["compress_ratios"]]
                if config.get("compress_ratios") is not None
                else None
            ),
            layer_types=list(config.get("layer_types", [])) if config.get("layer_types") is not None else None,
            linear_key_head_dim=(
                int(config["linear_key_head_dim"]) if config.get("linear_key_head_dim") is not None else None
            ),
            linear_value_head_dim=(
                int(config["linear_value_head_dim"]) if config.get("linear_value_head_dim") is not None else None
            ),
            linear_num_key_heads=(
                int(config["linear_num_key_heads"]) if config.get("linear_num_key_heads") is not None else None
            ),
            linear_num_value_heads=(
                int(config["linear_num_value_heads"]) if config.get("linear_num_value_heads") is not None else None
            ),
            attn_output_gate=bool(config.get("attn_output_gate", False)),
            vision_config=config.get("_vision_config"),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            use_bias=bool(pick(config, ["attention_bias", "mlp_bias", "use_bias"], False)),
            raw_architectures=list(config.get("architectures", [])),
        )

    @property
    def attention_output_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_output_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def moe_layer_indices(self) -> List[int]:
        if not self.num_experts:
            return []
        if self.decoder_sparse_step is not None or self.mlp_only_layers is not None:
            sparse_step = max(1, self.decoder_sparse_step or 1)
            dense_layers = set(self.mlp_only_layers or [])
            return [
                layer_idx
                for layer_idx in range(self.num_layers)
                if layer_idx not in dense_layers and (layer_idx + 1) % sparse_step == 0
            ]
        return list(range(self.first_k_dense_replace, self.num_layers, max(1, self.moe_layer_freq)))

    @property
    def moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    @property
    def dense_mlp_layers(self) -> int:
        return self.num_layers - self.moe_layers

    @property
    def full_attention_layers(self) -> int:
        if not self.layer_types:
            return self.num_layers
        return sum(1 for layer_type in self.layer_types if layer_type == "full_attention")

    @property
    def linear_attention_layers(self) -> int:
        if not self.layer_types:
            return 0
        return sum(1 for layer_type in self.layer_types if layer_type == "linear_attention")
