---
license: MIT License
tasks:
  - text-generation
frameworks: PyTorch
base_model:
  - ZhipuAI/GLM-5.1
base_model_relation: quantized
---
# GLM-5.1-w4a8

## 1. 基本信息
| 项目 | 信息 |
|:------:|:------:|
| 原始模型名  | GLM-5.1 |
| 原始模型链接  | https://huggingface.co/zai-org/GLM-5.1 |
| 测试机型  | Atlas 800T A3 1台|
| 版本  | vllm-ascend:v0.18.0rc1 |
| 链接  | quay.io/ascend/vllm-ascend:v0.18.0rc1 |


## 2 模型推理指导：

## 介绍

[GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) 采用混合专家（MoE）架构，其编码能力远强于前代产品。在 SWE-Bench Pro 上实现了最先进的性能，并在 NL2Repo（代码库生成）和 Terminal-Bench 2.0（真实世界终端任务）上大幅领先于 GLM-5。

本文档将展示该模型的主要验证步骤，包括支持特性、特性配置、环境准备、单节点与多节点部署、精度评估及性能评估。

## 支持特性

请参阅 [支持特性](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html) 获取模型支持的特性矩阵。

请参阅 [特性指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html) 获取特性配置说明。

## 环境准备

### 模型权重

- `GLM-5.1`（BF16 版本）：[下载模型权重](https://www.modelscope.cn/models/ZhipuAI/GLM-5.1)
- `GLM-5-w4a8`：[下载模型权重](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8)
- 可使用 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 对模型进行基础量化。

建议将模型权重下载至多节点共享目录，例如 `/root/.cache/`。

### 安装

vLLM 与 vLLM-ascend 仅在主分支支持 GLM-5。您可使用官方 Docker 镜像，并升级 vLLM 和 vLLM-ascend 进行推理。

```{code-block} bash
# 根据您的设备更新 --device（Atlas A3：/dev/davinci[0-15]）。
# 根据您的环境更新 vllm-ascend 镜像。
# 注意：您需要提前将权重下载至 /root/.cache。
# 更新 vllm-ascend 镜像，alm5-a3 可替换为：glm5;glm5-openeuler;glm5-a3-openeuler
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0rc1
export NAME=vllm-ascend

# 使用定义的变量运行容器
# 注意：若使用 Docker 桥接网络，请提前开放可供多节点通信的端口
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```


如需部署多节点环境，您需要在每个节点上分别完成环境配置。

## 部署

### 单节点部署

**A2 系列**

尚未测试。

**A3 系列**

- 量化模型 `glm-5.1-w4a8` 可部署于单台 Atlas 800 A3（64G × 16）。

执行以下脚本进行在线推理。

```shell
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5.1-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 16 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5.1 \
--max-num-seqs 8 \
--max-model-len 66600 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"multistream_overlap_shared_expert":true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' 
```

**注意：**
参数说明如下：

- 对于单节点部署，低延迟场景下我们推荐使用 `dp1tp16` 并关闭专家并行。
- `--async-scheduling`：异步调度是一种优化推理效率的技术，允许非阻塞的任务调度，以提高并发性和吞吐量，尤其在处理大规模模型时效果明显。

### 多节点部署

**A2 系列**

尚未测试。

**A3 系列**

- `glm-5.1-bf16`：至少需要 2 台 Atlas 800 A3（64G × 16）。

在两台节点上分别执行以下脚本。

**节点 0**

```shell
# 通过 ifconfig 获取本机信息
# nic_name 为当前节点 local_ip 对应的网卡接口名称
nic_name="xxx"
local_ip="xxx"

# node0_ip 的值必须与节点0（主节点）中设置的 local_ip 一致
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5.1-bf16 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12890 \
--tensor-parallel-size 16 \
--quantization ascend \
--seed 1024 \
--served-model-name glm-5.1 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

**节点 1**

```shell
# 通过 ifconfig 获取本机信息
# nic_name 为当前节点 local_ip 对应的网卡接口名称
nic_name="xxx"
local_ip="xxx"

# node0_ip 的值必须与节点0（主节点）中设置的 local_ip 一致
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5.1-bf16 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12890 \
--tensor-parallel-size 16 \
--quantization ascend \
--seed 1024 \
--served-model-name glm-5.1 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

### 前缀与解码分离

尚未测试。

## 精度评估

这里提供两种精度评估方法。

### 使用 AISBench

1. 详细步骤请参阅 [使用 AISBench 进行精度评估](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html)。
2. 执行后即可获得评估结果。

| 模型名 |  量化格式 | 数据集 | 测试精度 % | 官方精度 % |
|:------:|:------:|:------:|:------:|:------:|
| GLM-5.1-w4a8 | w4a8 | gpqa | 87.37 | 86.2 |

### 使用语言模型评估工具（Language Model Evaluation Harness）

尚未测试。

## 性能

### 使用 AISBench

详细步骤请参阅 [使用 AISBench 进行性能评估](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html#execute-performance-evaluation)。

### 使用 vLLM 基准测试工具

更多信息请参考 [vLLM 基准测试](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)。
