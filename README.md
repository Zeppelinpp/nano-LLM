# nano-LLM

From scratch 探索、理解、实现 LLM 在建模、训练、推理过程中的核心细节和算法。

> Python >= 3.12 | PyTorch >= 2.10 | 支持 Apple Silicon (MPS)

---

## 项目结构

```
.
├── attention/
│   └── attn.py                  # Multi-Head Attention (支持 KV Cache)
├── model/
│   ├── llms.py                  # 完整 LLM 建模 (RoPE, RMSNorm, MHA, MLP)
│   ├── rope.py                  # RoPE 位置编码独立实现
│   └── bert.py                  # BERT 模型加载示例
├── train/
│   ├── pretrain.py              # 预训练脚本 (支持 MPS/CUDA)
│   ├── preprocess/
│   │   ├── download_openwebtext.py   # 下载完整 OpenWebText
│   │   ├── download_subset.py        # 下载子集 (推荐先用这个)
│   │   └── prepare_subset.py         # 数据预处理
│   └── README.md                # 训练详细指南
├── test/
│   ├── test_attn.py             # Attention 功能测试
│   ├── test_kvcache.py          # KV Cache 验证 & 性能对比
│   └── test_torch.py            # Torch 常用操作示例
├── docs/
│   ├── torch_basic.md           # PyTorch 基础操作笔记
│   ├── kv_cache_benchmark_mps.md    # KV Cache 性能测试报告
│   ├── RoPE.md                  # RoPE 位置编码原理
│   └── gradient_accumulation.md # 梯度累积 & Batch Size 分析
├── examples/
│   └── pretrain_mps.sh          # MPS 预训练示例脚本
└── pyproject.toml
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行 Attention 性能对比 (KV Cache vs 无缓存)
uv run attention/attn.py

# 运行测试
uv run test/test_attn.py
uv run test/test_kvcache.py
```

### 预训练 (OpenWebText)

详见 [训练指南](./train/README.md)，快速体验：

```bash
# 1. 下载数据子集 (~304MB)
uv run train/preprocess/download_subset.py --output_dir ./data/openwebtext_subset

# 2. 预处理
uv run train/preprocess/prepare_subset.py --data_dir ./data/openwebtext_subset

# 3. 开始训练
uv run train/pretrain.py \
    --data_dir ./data/openwebtext_subset \
    --batch_size 4 \
    --max_length 256 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --output_dir ./checkpoints_test

# 4. 查看训练曲线
tensorboard --logdir ./checkpoints_test/tensorboard
```

## 核心模块

### Attention & KV Cache
- `AttentionLayer` 支持 Prefill (全序列处理) 和 Decoding (单 token + 缓存) 两阶段
- KV Cache 避免重复计算历史 token 的 Key/Value 投影，推理加速 2-3x (CPU)
- 相关文件：[attn.py](./attention/attn.py) | [KV Cache 测试](./test/test_kvcache.py) | [性能报告](./docs/kv_cache_benchmark_mps.md)

### LLM 建模
- 实现了一个完整的 Decoder-Only Transformer：RMSNorm → Multi-Head Attention → Residual → MLP
- RoPE 旋转位置编码，支持任意序列长度外推
- ~124M 参数 (768 hidden, 12 layers, 12 heads)
- 相关文件：[llms.py](./model/llms.py) | [rope.py](./model/rope.py) | [RoPE 原理](./docs/RoPE.md)

### 预训练
- 基于 OpenWebText 数据集，使用 GPT-2 Tokenizer
- 支持梯度累积、学习率 Warmup、TensorBoard 日志
- 针对 Apple Silicon MPS 做了内存优化
- 相关文件：[pretrain.py](./train/pretrain.py) | [梯度累积分析](./docs/gradient_accumulation.md)

## 已完成 & 待办

- [x] Multi-Head Attention 的完整计算过程 — [attn.py](./attention/attn.py)
- [x] Prefill 和 Decoding 两个阶段的不同计算行为
- [x] KV Cache 实现与性能验证
  - [KV Cache 推理性能提升](./docs/kv_cache_benchmark_mps.md)
  - [KV Cache 测试](./test/test_kvcache.py)
- [x] 常见 LLM 结构建模 (RMSNorm, MHA, Residual, RoPE) — [llms.py](./model/llms.py)
- [x] RoPE 旋转位置编码原理与实现 — [RoPE.md](./docs/RoPE.md) | [rope.py](./model/rope.py)
- [x] PyTorch 高频矩阵操作整理 (持续更新)
  - [Torch Basics](./docs/torch_basic.md) | [Torch 操作测试](./test/test_torch.py)
- [x] OpenWebText 预训练流程 — [训练指南](./train/README.md)
- [x] 梯度累积与 Batch Size 分析 — [gradient_accumulation.md](./docs/gradient_accumulation.md)
- [ ] 推理 Generate 实现 (Top-K, Top-P, Temperature)
- [ ] Tokenizer 原理与 BPE 实现
- [ ] Flash Attention 实现
