# 大模型显存估算：从原理到工具 演讲稿

## Slide 1 大模型显存估算：从原理到工具

大家好，今天想和大家分享一个比较工程化的话题：大模型部署时，显存到底应该怎么估。

我们平时拿到一个模型，最常见的问题不是“这个模型有多强”，而是更朴素也更紧急的问题：这几张卡到底能不能跑起来？如果能跑，最大上下文能开到多少？如果是 MoE、PD 分离、量化模型，显存又应该怎么算？

所以这次分享会从基础原理开始，把单卡显存拆成几块：权重、KV cache、activation、runtime 和通信缓冲。最后再介绍我们这个 `vllm-vram-estimator` 工具，它的目标就是：拿到 model config 和部署参数后，先在部署前估一个 per-GPU 显存，避免靠反复启动试错。

过渡：我们先不看具体模型，先看总公式。

## Slide 2 显存不是一个数，而是几类占用的叠加

估显存最重要的一步，是不要把显存当成一个黑盒数字。

对推理部署来说，单卡显存大致可以拆成四类。第一类是 model weights，也就是模型权重，它通常是常驻显存。第二类是 KV cache，它是为了保存上下文里每个 token 的 key/value 状态，通常随着 token 数线性增长。第三类是 activation peak，主要在 prefill、大 batch、长 prompt 时出现。第四类是 runtime reserve，包括 allocator 碎片、kernel workspace、graph、通信缓冲等。

所以我们的总公式就是：

`VRAM_per_gpu = weights + KV cache + activation peak + runtime reserve`

后面所有复杂情况，本质上都是在问：这四项分别应该怎么从 config 和部署参数里推出来。

过渡：先看最稳定、也最容易理解的一项，权重显存。

## Slide 3 权重显存从参数量开始，再乘以 dtype 并切分

权重显存的基本逻辑很直接：先估参数量，再乘以每个参数占多少字节，然后看并行策略怎么切。

参数量主要来自 embedding、attention、MLP 或 experts、norm 和 bias。对于 dense 模型，通常总参数量就比较直接。比如 bf16 大概每个参数 2 bytes，fp8 约 1 byte，int4 约 0.5 byte。当然低 bit 量化还会有 scale、zero point、group metadata 等额外开销。

并行切分这里要稍微小心。TP 会切 attention 和 MLP 的矩阵，所以非专家层通常可以按 TP 分。MoE 如果开启 EP，专家权重还会按 DP 进一步分摊。因此在 EP=true 的情况下，可以近似写成：

`weights/GPU = non_expert / TP + expert / (TP × DP)`

这里我们当前工具里，PP 在示例中固定为 1，不用 PP 再切层。

过渡：权重决定模型能不能放下，但上下文长度往往是 KV cache 决定的。

## Slide 4 KV cache 往往决定最大上下文长度

KV cache 是大模型服务里非常关键的一块显存，因为它直接和 token budget 相关。

标准 attention 下，每个 token、每层都要保存 K 和 V。单层单 token 的大小大致是：

`2 × local_kv_heads × head_dim × bytes`

再乘以 tokens 和 local layers，就是这张卡上 KV cache 的显存。

这里有几个变量很重要。TP 会影响 local KV heads，PP 如果使用会影响 local layers，不过我们这里 PP 固定为 1。`max_model_len`、`max_num_seqs`、`max_num_batched_tokens` 会影响 token 预算。

另外，现在很多模型不是标准 KV。比如 GLM、OpenPangu 这类 MLA 或 compressed KV 结构，不能再按 `2 × kv_heads × head_dim` 算，而要按类似：

`kv_lora_rank + qk_rope_head_dim`

来估。如果这个地方不适配，KV cache 可能会被严重高估或低估。

过渡：接下来看看 MoE，因为 MoE 里“参数量”和“激活参数量”很容易混淆。

## Slide 5 MoE 的 active 参数不等于权重显存

MoE 模型经常会写 A3B、A10B、A17B 这样的口径。这里的 active 参数指的是每个 token 实际激活参与计算的参数量，比如 A3B 就是每个 token 大约激活 3B 参数。

但部署显存不是按 active 参数算的。显存里常驻的是这个 rank 持有的权重，包括非专家层和分到本 rank 的专家层。

所以 MoE 至少要拆两个口径。一个是 active/token，它更接近推理计算量和 token latency。另一个是 resident weights，也就是常驻权重，它才决定部署时这张卡要放多少模型参数。

工具里会同时输出 total params、expert/non-expert、active/token。这样我们既能看模型计算口径，也能看实际显存口径。

过渡：有了权重和 KV 的基础之后，我们再看并行方式怎样改变每张卡持有的东西。

## Slide 6 并行方式改变的是“每张卡持有哪些东西”

TP、DP、EP、PP 这几个概念容易混在一起，但从显存角度看，可以理解成它们分别改变了“每张卡持有什么”。

TP 是张量并行，会切矩阵和 attention heads，因此权重会下降，KV 里的 local KV heads 也可能下降。它常用于让大模型能放进单卡预算。

DP 是数据并行，本质是多个服务副本。普通 dense 权重不会因为 DP 下降，因为每个副本都有一份模型。但在 MoE + EP 场景下，DP 也会参与专家切分。

EP 是专家并行，主要影响 MoE experts。它能显著降低每张卡上的专家权重。

PP 是流水并行，会切层，也会影响 local layers 和 KV，但我们这个工具当前的部署建议里 PP 固定为 1，避免把问题复杂化，也贴近现在用户要求的场景。

实践上，通常先用 TP/EP 让权重 fit，再看 KV cache 是否限制上下文。

过渡：接下来讲 vLLM 部署时为什么会“预分配”显存。

## Slide 7 vLLM 会先确定可用预算，再预留 KV cache

vLLM 启动时不是等请求来了再随便增长显存，而是会先根据 `gpu_memory_utilization` 确定一个可用预算。

大致流程是：先加载模型权重，然后 profile 或预留一部分 runtime 和 activation 空间，接着计算剩下多少显存可以给 KV cache，最后按 block size 分配 block cache。

所以 `max_model_len`、`max_num_seqs`、`max_num_batched_tokens` 会直接影响启动时 KV 的预留。

尤其是 `max_num_batched_tokens`，如果设置了它，KV 预留会优先看这个 token budget。否则通常会从 `max_model_len × max_num_seqs` 这类配置推导。

这也是为什么我们工具里有“反推最大上下文长度”的功能：在给定权重、runtime、activation、显存预算后，看 KV 还能容纳多少 token。

过渡：单体部署讲完后，我们看一个更真实的大模型服务场景，PD 分离。

## Slide 8 PD 分离让 P 与 D 节点按不同瓶颈配置

PD 分离的核心思想是：prefill 和 decode 的瓶颈不同，所以可以分成两类节点分别配置。

Prefill 节点更关注 prompt 吞吐和 activation 峰值。它通常会有较大的 `max_num_batched_tokens`，TP 也可能比较高，因为要处理大 prompt，并产生 KV。

Decode 节点更关注常驻 KV cache 和持续生成。它通常可以开更高 DP，因为 decode 侧每步计算比较小，但会长期持有上下文 KV。

所以 PD 场景不能只算一套 deployment。P 节点和 D 节点要分别算权重、KV、activation、runtime，然后分别看 fit 和 headroom。

过渡：PD 分离还有一块之前容易漏掉的显存，就是 KV transfer。

## Slide 9 PD 还需要显式预留 KV transfer 与通信缓冲

在 PD 分离里，P 节点产生 KV 后，需要把 KV 传给 D 节点。如果不单独算这部分 buffer，P/D 的总显存会偏乐观。

工具里现在支持两种估算模式。

第一种是 vLLM buffer 模式，按 `KVTransferConfig.kv_buffer_size` 来算。vLLM 里这个默认值是 `1e9 bytes`，也就是每卡大约 0.93 GiB。

第二种是 payload 模式。如果我们想更保守，或者在 vLLM-Ascend、HCCL、Mooncake 这类实现里希望按一次搬运的 token payload 来估，可以用：

`tokens × bytes_per_token × buffer_factor`

这里的 `bytes_per_token` 会根据当前 P 或 D 节点的 KV layout 自动计算，因此 TP 不同、MLA compressed KV 不同，结果都会不同。

另外还有 `comm_buffer/GPU`，可以额外预留 HCCL、NCCL、NIC 或 connector workspace。

过渡：到这里，原理部分差不多了。下面看这个工具怎么把这些东西落成报告。

## Slide 10 工具把模型 config 和部署参数变成可解释报告

这个工具的输入有两类。

第一类是模型配置，也就是 `config.json`。工具会从里面读 hidden size、层数、attention heads、KV heads、MoE experts、MLA 参数、vision config、quantization config 等。

第二类是部署配置，比如 TP、DP、EP、dtype、max model length、max num seqs，以及 PD 里的 prefill/decode 配置。

中间的估算核心主要分成几个模块：`params.py` 估参数量，`memory.py` 估权重和 KV，`scenarios/pd.py` 处理 PD 分离和 KV transfer。

输出不是只有一个 total，而是包括 `per_gpu_gib`、breakdown、fit、headroom。这样你能知道为什么 OOM，是权重太大，还是 KV 太大，还是 activation 或 transfer buffer 太保守。

命令也尽量简单，例如：

`python3 vllm_vram_estimator.py --model qwen3.5-35b-a3b --max-context`

过渡：我们用一个真实的大模型配置看一下显存瓶颈是怎么暴露出来的。

## Slide 11 OpenPangu 1P1D 示例：显存瓶颈会暴露在 breakdown 里

这里是 OpenPangu 718B 的 1P1D 示例。假设每张卡 64GiB，`gpu_memory_utilization=0.9`，那么实际预算是：

`64 × 0.9 = 57.60 GiB`

工具输出 P 侧单卡里，权重大约 41.85 GiB，KV cache 大约 8.58 GiB，activation 预留 6 GiB，KV transfer 大约 0.93 GiB。

加起来 total 约 59.36 GiB，超过预算 57.60 GiB，也就是 headroom 约 -1.76 GiB。所以这个配置下 P 侧会 OOM。

这个结果的意义在于，它不是简单告诉我们“不行”，而是告诉我们“为什么不行”。下一步可以选择降低 P 侧 token budget、减少 activation reserve，或者增加 TP、换更大显存卡。

过渡：最后总结一下实际使用时的流程。

## Slide 12 显存估算的正确姿势：先拆账，再调参

最后总结一下。

第一，先看权重。确认 dtype、量化方式、TP/EP 是否能让模型权重放下。

第二，再看 KV。用上下文长度、并发数和 batched tokens 估 token budget，因为最大上下文通常就是被 KV 限制。

第三，PD 要单独看。P/D 分别计算，不同 TP/DP 下显存结构会完全不同。

第四，保留余量。runtime、activation、KV transfer 和通信缓冲都要显式写进配置里，不要只靠一个总的经验值。

最后要强调，显存估算不是替代线上 profile。它的价值是让部署方案在上线前更快收敛。真正部署时，还是要用 vLLM 或 vLLM-Ascend 的启动日志和实际 profile 来校准 `activation_peak_gib` 与 `runtime_overhead_gib`。

结束语：

这就是今天的分享。简单说，我们希望把显存估算从“拍脑袋试启动”变成“拆账、估算、验证、校准”的流程。这个工具就是为了把这件事做得更透明、更可解释。
