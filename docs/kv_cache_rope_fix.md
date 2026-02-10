---
tags:
  - kv-cache
  - attention
  - rope
  - bugfix
createTime: 2026-02-10
Last Modified: 2026-02-10
---

# KV Cache 与 RoPE 集成问题修复

## 问题概述

在实现带有 KV Cache 的 Attention 机制时，RoPE（Rotary Position Embedding）的应用存在三个关键问题，导致推理阶段位置编码失效。

## 发现的问题

### 问题 1：RoPE 在推理阶段位置编码错误 [严重]

**受影响文件**：`model/llms.py:130-133`

**问题描述**：

原始实现：
```python
# Apply RoPE if enabled
if self.use_rope:
    q = self.rope(q, seq_len)  # seq_len 是新输入的长度（通常为 1）
    k = self.rope(k, seq_len)
```

**根本原因**：
- 训练阶段处理完整序列（如 512 tokens），位置从 0 到 511，RoPE 工作正常
- 推理阶段每次处理单个 token，`seq_len = 1`，RoPE 始终从位置 0 编码
- 使用 KV cache 后，第 10 个 token 应该使用位置 10 的编码，但实际使用位置 0
- **结果**：所有新 token 的位置信息完全相同，破坏了模型的位置感知能力

**影响范围**：
- 训练阶段：无影响（始终处理完整序列）
- 推理阶段：位置信息完全错误，模型无法区分不同位置的 token

**修复方案**：

1. 修改 `RoPE.forward()` 接受显式的 `position_ids` 参数：
```python
def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        position_ids: Position indices of shape (seq_len,) or (batch, seq_len)
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    # 使用显式位置计算角度
    angles = position_ids.unsqueeze(-1).float() @ self.inv_freq.unsqueeze(0)
    # ...
```

2. 在 `AttentionBlock.forward()` 中计算正确的位置偏移：
```python
# 计算缓存长度
cache_len = 0
if use_kv_cache and kv_cache is not None and "keys" in kv_cache:
    cache_len = kv_cache["keys"].shape[1]

# 创建从 cache_len 开始的位置 IDs
if position_ids is None:
    position_ids = torch.arange(
        cache_len, cache_len + seq_len, dtype=torch.long, device=device
    )

# 应用 RoPE
if self.use_rope:
    q = self.rope(q, position_ids)  # 使用正确的位置
    k = self.rope(k, position_ids)
```

**验证示例**：
```python
# 推理第 1 步：处理 prompt（10 tokens）
position_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 正确
cache_len = 0

# 推理第 2 步：生成第 1 个 token
position_ids = [10]  # 正确（cache_len=10, seq_len=1）
cache_len = 10

# 推理第 3 步：生成第 2 个 token
position_ids = [11]  # 正确（cache_len=11, seq_len=1）
cache_len = 11
```

---

### 问题 2：Causal Mask 形状不匹配 [重要]

**受影响文件**：`model/llms.py:154-158`

**问题描述**：

原始实现：
```python
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype), diagonal=1
)
```

**根本原因**：
- 使用 KV cache 后，k/v 的序列长度是 `cache_len + seq_len`（历史 + 新增）
- attention scores 的形状是 `(batch, num_heads, seq_len, cache_len + seq_len)`
- causal mask 的形状是 `(seq_len, seq_len)`，维度不匹配
- 广播机制会导致 mask 应用错误或运行时错误

**修复方案**：

```python
# 获取 k 的实际序列长度（包含缓存）
kv_seq_len = k.shape[2]

# Apply causal mask
if use_kv_cache and cache_len > 0:
    # 推理阶段：新 token 可以 attend 到所有历史 token（包括自己）
    # mask 全为 0（不遮蔽任何位置）
    causal_mask = torch.zeros(
        seq_len, kv_seq_len, device=device, dtype=scores.dtype
    )
else:
    # 训练阶段：标准因果 mask（上三角为 -inf）
    causal_mask = torch.triu(
        torch.ones(seq_len, kv_seq_len, device=device, dtype=scores.dtype),
        diagonal=1
    )
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
```

**推理阶段的因果性说明**：
- 推理时每次只生成 1 个新 token
- 这个新 token 可以 attend 到所有已生成的历史 token（它们都在"过去"）
- 因此不需要 mask，所有位置都应该是可见的
- 因果性已经由自回归生成过程保证（token 只能看到之前生成的内容）

---

### 问题 3：Position Embeddings 与 RoPE 冲突 [重要]

**受影响文件**：`model/llms.py:301, 336`

**问题描述**：

原始实现同时使用两种位置编码：
```python
# __init__
self.position_embed = nn.Embedding(max_position_embeddings, hidden_size)

# forward
x = self.token_embed(input_ids) + self.position_embed(position_ids)  # Learned PE
# ... 然后在 attention 层又应用 RoPE
```

**根本原因**：
- **Learned Position Embeddings**：绝对位置编码，通过可学习参数将位置信息加到 embeddings
- **RoPE**：相对位置编码，通过旋转 Q/K 向量注入位置信息
- 同时使用会导致位置信息重复编码，干扰 RoPE 的相对位置优势
- RoPE 的外推能力（处理超过训练长度的序列）会被 learned embeddings 限制

**修复方案**：

```python
def __init__(self, ..., use_rope: bool = True, ...):
    # Token embeddings
    self.token_embed = nn.Embedding(vocab_size, hidden_size)

    # Position embeddings - 只在不使用 RoPE 时使用
    if not use_rope:
        self.position_embed = nn.Embedding(max_position_embeddings, hidden_size)
    else:
        self.position_embed = None

def forward(self, input_ids, ...):
    x = self.token_embed(input_ids)

    # Add position embeddings only if not using RoPE
    if self.position_embed is not None:
        x = x + self.position_embed(position_ids)
```

**设计原则**：
- 使用 RoPE：不添加 position embeddings，位置信息完全由 RoPE 提供
- 不使用 RoPE：使用 learned position embeddings（类似 GPT-2）
- 两者互斥，避免位置信息重复

---

## 修复总结

### 核心变更

| 组件 | 原始行为 | 修复后行为 |
|------|---------|-----------|
| **RoPE** | 接受 `seq_len`，从 0 开始编码 | 接受 `position_ids`，使用正确绝对位置 |
| **Causal Mask** | 固定 `(seq_len, seq_len)` | 动态 `(seq_len, kv_seq_len)`，推理时全为 0 |
| **Position Embed** | 总是添加 | 仅在 `use_rope=False` 时添加 |

### 训练 vs 推理对比

| 阶段 | Position IDs | Causal Mask | Position Embeddings |
|------|-------------|-------------|---------------------|
| **训练** | `[0, 1, 2, ..., 511]` | 上三角 `-inf` | 不添加（use_rope=True） |
| **推理第1步** | `[0, 1, ..., 9]` (prefill) | 上三角 `-inf` | 不添加 |
| **推理第2步** | `[10]` (decode) | 全为 0 | 不添加 |
| **推理第N步** | `[10+N-1]` | 全为 0 | 不添加 |

### 代码流程图

**修复前（错误）**：
```
推理第 10 步：
input_ids [shape: (1, 1)]
  ↓
RoPE(position=0)  ← 错误！应该是 position=10
  ↓
concat with cache (positions 0-9)
  ↓
attention: position 0 与 positions 0-9 交互  ← 位置信息混乱
```

**修复后（正确）**：
```
推理第 10 步：
input_ids [shape: (1, 1)]
  ↓
cache_len = 10
  ↓
position_ids = [10]  ← 正确位置
  ↓
RoPE(position=10)
  ↓
concat with cache (positions 0-9, 已应用 RoPE)
  ↓
attention: position 10 与 positions 0-10 交互  ← 位置信息正确
```

参考可视化示意图：[RoPE 与 Causal Mask 施加示意图](./assets/rope_causal_mask_visualization.html)

---

## 性能影响

### 计算开销
- 无额外计算：修复仅改变位置参数，不增加计算量
- 优化：RoPE 的 `inv_freq` 预计算并注册为 buffer，避免重复计算

### 内存使用
- 无变化：KV cache 大小不变
- 移除冗余：不再同时存储 position embeddings 和 RoPE 参数

---

## 测试建议

### 单元测试
```python
def test_rope_positions_with_kv_cache():
    model = GeneralLLM(use_rope=True)

    # 第 1 步：prefill（10 tokens）
    input_ids = torch.randint(0, 50257, (1, 10))
    logits, cache = model(input_ids, use_kv_cache=True)

    # 第 2 步：生成第 1 个 token
    new_token = torch.randint(0, 50257, (1, 1))
    logits, cache = model(new_token, use_kv_cache=True, kv_cache_list=cache)

    # 验证：cache 中的 keys 应该有 11 个位置（0-10）
    assert cache[0]["keys"].shape[1] == 11

    # 第 3 步：生成第 2 个 token
    new_token = torch.randint(0, 50257, (1, 1))
    logits, cache = model(new_token, use_kv_cache=True, kv_cache_list=cache)

    # 验证：cache 应该有 12 个位置
    assert cache[0]["keys"].shape[1] == 12
```

### 一致性测试
```python
def test_output_consistency():
    """验证使用和不使用 KV cache 的输出一致性"""
    model = GeneralLLM(use_rope=True)
    input_ids = torch.randint(0, 50257, (1, 20))

    # 不使用 cache：一次性处理所有 tokens
    logits_no_cache, _ = model(input_ids, use_kv_cache=False)

    # 使用 cache：分步处理
    logits_prefill, cache = model(input_ids[:, :10], use_kv_cache=True)

    logits_list = [logits_prefill]
    for i in range(10, 20):
        logits, cache = model(
            input_ids[:, i:i+1],
            use_kv_cache=True,
            kv_cache_list=cache
        )
        logits_list.append(logits)

    logits_with_cache = torch.cat(logits_list, dim=1)

    # 验证输出一致性（允许数值误差）
    assert torch.allclose(logits_no_cache, logits_with_cache, atol=1e-5)
```

---

## 相关文件

- **修复文件**：`model/llms.py`
- **受影响模块**：
  - `RoPE`：行 7-64
  - `AttentionBlock`：行 67-189
  - `TransformerBlock`：行 240-275
  - `GeneralLLM`：行 278-354

---

## 参考文档

- [RoPE 原理](./RoPE.md)
- [KV Cache Benchmark](./kv_cache_benchmark_mps.md)
- [RoFormer 论文](https://arxiv.org/abs/2104.09864) - RoPE 原始论文

---

## 版本历史

- **2026-02-10**：首次记录修复方案
  - 修复 RoPE 位置编码问题
  - 修复 Causal Mask 形状错误
  - 移除 Position Embeddings 与 RoPE 的冲突
