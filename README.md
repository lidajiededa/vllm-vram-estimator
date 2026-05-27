# vLLM 显存估算工具

这是一个 Python CLI。输入 Hugging Face `config.json`、已注册模型名或部署参数，它会估算每张 GPU 的显存占用，并拆成权重、KV cache、activation 峰值、runtime reserve、LoRA、投机推理等部分。

## 推荐用法

优先使用模型名，不必手写 config 路径：

```bash
python3 vllm_vram_estimator.py --model qwen3.5-35b-a3b
```

也可以把模型名作为第一个参数：

```bash
python3 vllm_vram_estimator.py Qwen/Qwen3.5-27B-FP8 --tp 4 --max-model-len 262144 --max-num-seqs 1 --gpu-memory-gib 80
```

查看所有已注册模型：

```bash
python3 vllm_vram_estimator.py --list-models
```

仍然支持直接传 config 路径：

```bash
python3 vllm_vram_estimator.py examples/configs/qwen2_7b/config.json --tp 1 --dtype bf16 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-gib 80
```

输出 JSON：

```bash
python3 vllm_vram_estimator.py --model deepseek-v4-flash-base --json
```

反推当前配置下不 OOM 的最大上下文长度：

```bash
python3 vllm_vram_estimator.py --model qwen3.5-35b-a3b --max-context
```

限制搜索上界：

```bash
python3 vllm_vram_estimator.py --model qwen3.5-35b-a3b --max-context --capacity-upper 262144
```

## 已注册模型

Qwen / Qwen3：

```text
qwen2-7b
qwen3-8b
qwen3-8b-vl
qwen3-32b-vl
qwen3-30b-a3b
qwen3-30b-a3b-vl
```

Qwen3.5 官方系列 21 个模型：

```text
qwen3.5-0.8b
qwen3.5-0.8b-base
qwen3.5-2b
qwen3.5-2b-base
qwen3.5-4b
qwen3.5-4b-base
qwen3.5-9b
qwen3.5-9b-base
qwen3.5-27b
qwen3.5-27b-fp8
qwen3.5-27b-gptq-int4
qwen3.5-35b-a3b
qwen3.5-35b-a3b-base
qwen3.5-35b-a3b-fp8
qwen3.5-35b-a3b-gptq-int4
qwen3.5-122b-a10b
qwen3.5-122b-a10b-fp8
qwen3.5-122b-a10b-gptq-int4
qwen3.5-397b-a17b
qwen3.5-397b-a17b-fp8
qwen3.5-397b-a17b-gptq-int4
```

Qwen3.6 / GLM / DeepSeek / Pangu：

```text
qwen3.6-27b
qwen3.6-35b-a3b
qwen3.6-35b-a3b-pd
glm-5.1
glm-5.1-tp16
glm-5.1-w4a8
deepseek-v4-flash-base
deepseek-v4-pro-base
deepseek-moe-like
openpangu-ultra-moe-718b
pangu-v2-moe
```

模型名支持宽松写法，例如 `Qwen/Qwen3.5-35B-A3B`、`qwen3_5_35b_a3b`、`deepseek-ai/DeepSeek-V4-Pro-Base` 都可以解析。

## PD 分离

PD 分离 deployment JSON 顶层可以放公共参数，`prefill` 和 `decode` 分别覆盖各自配置。D 节点通常可以设置更高的 `dp`，P 节点通常用 `max_num_batched_tokens` 和更保守的 `activation_peak_gib` 描述 prefill 压力。

```bash
python3 vllm_vram_estimator.py --model qwen3.6-35b-a3b-pd
```

PD 分离也支持反推最大上下文长度，会分别输出 P 节点和 D 节点：

```bash
python3 vllm_vram_estimator.py --model qwen3.6-35b-a3b-pd --max-context --capacity-upper 262144
```

如果 P 节点设置了 `max_num_batched_tokens`，工具会提示该节点 KV 显存由固定 token budget 控制，而不是由 `max_model_len` 控制。

PD 场景会额外估算 KV transfer 和通信缓冲：

```json
{
  "kv_transfer": {
    "enabled": true,
    "mode": "vllm_buffer",
    "kv_buffer_size_bytes": 1000000000,
    "comm_overhead_gib": 0.0
  }
}
```

默认 `mode=vllm_buffer`，按 vLLM `KVTransferConfig.kv_buffer_size` 的默认 `1e9 bytes` 估算每卡 connector buffer，输出为 `kv_transfer/GPU`。如果要按一次搬运的 KV payload 估算，可以改成：

```json
{
  "kv_transfer": {
    "mode": "payload",
    "tokens": 8192,
    "buffer_factor": 2.0,
    "comm_overhead_gib": 1.0
  }
}
```

payload 模式按当前角色的本卡 KV layout 计算：

```text
transfer_bytes =
  ceil(tokens / block_size) * block_size
  * kv_bytes_per_token_per_gpu
  * buffer_factor
```

因此 P/D 的 TP、MLA compressed KV、linear/full attention 层数差异都会反映到 transfer buffer 里。`comm_overhead_gib` 用于额外预留 HCCL/NCCL/NIC/connector workspace。

等价于：

```bash
python3 vllm_vram_estimator.py examples/configs/qwen3_6_35b_a3b/config.json --deployment examples/deployments/qwen3_6_35b_a3b/pd.json
```

## 报告字段

总显存按下面几项相加：

```text
VRAM_per_gpu =
  model_weights_shard
+ kv_cache_reserved
+ activation_peak
+ runtime_overhead
+ feature_extra
```

JSON 里保留 `per_gpu_gib`，并提供更细的 `per_gpu_breakdown`：

```text
weights.children.non_expert     本卡非专家权重
weights.children.expert         本卡专家权重
weights.children.quant_overhead 量化 scale/zero/group 元数据预留
kv_cache.children               KV token 预算、block、local layers、local KV heads、每 1k tokens 显存
```

MoE 模型会同时输出：

```text
total params       常驻总参数，决定权重显存
active/token       每 token 激活参数，例如 A3B 里的 3B
```

注意：active 参数量不能直接拿来估权重显存；vLLM 部署时本 rank 持有的专家权重仍然要常驻显存。

## 特殊模型处理

Qwen3.5：

```text
自动展开 text_config。
FP8 / GPTQ-Int4 会从 quantization_config 自动读取权重 dtype。
A3B / A10B / A17B 的 active/token 分别按官方口径校准为 3B / 10B / 17B。
```

Qwen3.6：

```text
自动展开 text_config。
vision_config 会估算为 vision 权重。
layer_types 区分 linear_attention / full_attention，KV cache 只按 full attention 层估算。
Qwen3.6-35B-A3B 的 active/token 使用官方 A3B 口径校准为 3B。
```

Qwen3-VL：

```text
支持 qwen3-8b-vl、qwen3-32b-vl、qwen3-30b-a3b-vl，并兼容 Hugging Face 名称 Qwen/Qwen3-VL-*-Instruct。
会自动展开 text_config，vision_config 估算为 vision 权重。
Qwen3-VL-30B-A3B 的 active/token 按 A3B 口径校准为 3B。
```

DeepSeek-V4：

```text
q_lora_rank / o_lora_rank 按低秩 Q/O attention 参数估算。
expert_dtype=fp8 和 quantization_config.quant_method=fp8 会让权重默认按 FP8 存储估算。
compress_ratios 会用于 KV cache 的 token-wise compression。
Flash/Pro 的 active/token 使用官方 rounded 值校准。
```

GLM-5.1：

```text
kv_cache_dim_per_token_layer = kv_lora_rank + qk_rope_head_dim
```

因此会输出 `kv_cache_layout=compressed_kv_lora`，而不是按普通 `2 * kv_heads * head_dim` 估算。
`glm-5.1-w4a8` 来自 ModelScope `Eco-Tech/GLM-5.1-w4a8`，默认按 ModelSlim/昇腾 W4A8 的 4bit 权重和额外 scale/offset 元数据估算。该仓库的量化权重 index 标注 `total_size=419847499776 bytes`，因此默认 `quant_overhead_ratio=0.13` 以贴近真实文件体积。

openPangu Ultra MoE 718B：

```text
来自 ModelScope `FreedomIntelligence/openPangu-Ultra-MoE-718B`。
结构是 MoE + MLA/kv_lora compressed KV，按 `kv_lora_rank + qk_rope_head_dim` 估算 KV cache。
示例默认使用 TP=32、EP=true、BF16、80GiB 卡。
```

Pangu V2 MoE：

```text
支持 `pangu-v2-moe`。结构是 MoE + MLA/kv_lora compressed KV，同时包含 DSA full-context 层和 SWA sliding-window 层。
KV cache 会对 `swa_layers` 按 `sliding_window_list` 截断 token 预算，其余 DSA/普通层按完整 token budget 估算。
示例默认使用 TP=32、EP=true、BF16、64GiB 卡。
```

## 代码结构

```text
vram_estimator/
  cli.py          命令行参数和入口
  deployment.py   vLLM 部署参数
  models.py       config 解析和模型结构归一化
  registry.py     模型名到 config/deployment 的映射
  params.py       参数量、active/token、vision 参数估算
  memory.py       权重、KV cache、LoRA、投机推理显存估算
  report.py       JSON 报告组装
  printing.py     human-readable 输出
  capacity.py     反推给定显存预算下的最大上下文长度
  scenarios/pd.py PD 分离场景
  utils.py        dtype、单位和小工具
examples/
  configs/        每个模型一个目录，目录内放 config.json 和模型侧辅助文件
  deployments/    每个模型一个目录，目录内放 tp4.json、pd.json 等部署参数
```

## 注册新模型

最小注册流程如下：

1. 保存模型 config。

   推荐把 Hugging Face / ModelScope 的 `config.json` 放到 `examples/configs/<model_name>/` 下，例如：

   ```text
   examples/configs/my_model/config.json
   ```

   如果是 wrapper config，例如 VL 模型常见的外层 `model_type` + 内层 `text_config`，工具会优先展开 `text_config`，并保留外层 `vision_config` 用于估算 vision 权重。

2. 可选：保存默认 deployment。

   如果这个模型有推荐部署方式，可以放到 `examples/deployments/<model_name>/` 下：

   ```text
   examples/deployments/my_model/tp8.json
   ```

   示例：

   ```json
   {
     "tp": 8,
     "pp": 1,
     "dp": 1,
     "ep": false,
     "dtype": "bf16",
     "kv_cache_dtype": "bf16",
     "max_model_len": 32768,
     "max_num_seqs": 1,
     "gpu_memory_gib": 80,
     "gpu_memory_utilization": 0.9
   }
   ```

3. 在 `vram_estimator/registry.py` 注册模型名。

   ```python
   "my-model": ModelEntry(
       "my-model",
       "examples/configs/my_model/config.json",
       "examples/deployments/my_model/tp8.json",
       hf_id="Org/My-Model",
   ),
   ```

   `default_deployment` 可以不填。`hf_id` 建议填写，方便支持 `Org/My-Model` 这种命令行写法。

4. 如果新模型结构字段已有通用含义，通常不需要改代码。

   当前已经支持常见字段：

   ```text
   text_config / vision_config
   hidden_size / num_hidden_layers / num_attention_heads / num_key_value_heads
   intermediate_size / moe_intermediate_size
   num_experts / n_routed_experts / num_experts_per_tok
   n_shared_experts / shared_expert_intermediate_size
   kv_lora_rank / q_lora_rank / qk_rope_head_dim
   layer_types / linear_attention / full_attention
   quantization_config.quant_method
   ```

   如果模型有特殊结构，例如新的 MLA 变体、特殊 KV 压缩、非标准 MoE 字段、vision projector 字段不一致，才需要扩展：

   ```text
   vram_estimator/models.py   解析 config 字段
   vram_estimator/params.py   参数量和 active/token 估算
   vram_estimator/memory.py   权重、KV cache、量化显存估算
   ```

5. 添加测试并验证。

   推荐在 `tests/test_estimator_smoke.py` 加 registry 和关键参数断言；如果要加入全量样例运行，再把 config 加到 `tests/run_all_example_configs.py`。

   ```bash
   python3 -m compileall -q vram_estimator vllm_vram_estimator.py
   python3 tests/test_estimator_smoke.py
   python3 tests/run_all_example_configs.py
   ```

后续建议新增模块：

```text
scenarios/spec.py     更完整的投机采样、draft/EAGLE/MTP 拆账
quant/modelslim.py    ModelSlim/昇腾量化模型读取和 scale 元数据估算
capacity.py           在给定显存预算下反推最大上下文长度
```

## 测试

```bash
cd /mnt/d/workspace/vllm-vram-estimator
source /mnt/d/workspace/.venv/Scripts/activate
python3 -m compileall -q vram_estimator vllm_vram_estimator.py
python3 tests/test_estimator_smoke.py
python3 tests/run_all_example_configs.py
```
