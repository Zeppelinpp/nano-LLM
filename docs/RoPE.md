---
tags:
  - infra
  - attention
  - positional-encoding
createTime: 2026-02-09T19:24:00
Last Modified: 2026-02-09 19:25
---
# RoPE (Rotary Position Embedding)

## 基本概念

RoPE（旋转位置编码）是一种相对位置编码方法，通过旋转 Query 和 Key 向量来注入位置信息。与传统的绝对位置编码不同，RoPE 利用复数性质，让注意力计算天然具备相对位置感知能力。

数学原理：将向量视为复数，乘以旋转因子 $e^{i\theta}$ 实现旋转

$$
v' = R_{\theta} v =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta &  \cos\theta
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
$$

其中 $\theta$ 与位置相关：$\theta = \text{position} \times \text{frequency}$

## 实现步骤

### 1. 计算频率

首先计算逆频率（inverse frequencies），用于确定每个维度的旋转速度：

```python
rope_base = 10000
position_ids = torch.arange(seq_len)  # [seq_len]
inverse_frequencies = 1.0 / (
    rope_base ** (torch.arange(0, head_dim, 2) / head_dim)
)  # [head_dim // 2]
```

逆频率决定了不同维度的旋转速度，高频维度旋转快（变化剧烈），低频维度旋转慢（变化平缓）。

### 2. 计算位置频率并生成 sin/cos

将位置 ID 与逆频率相乘，得到每个位置在每个维度上的旋转角度：

```python
# 计算频率矩阵 [seq_len, head_dim // 2]
freqs = position_ids.unsqueeze(-1) * inverse_frequencies.unsqueeze(0)

# 计算 sin 和 cos
sin, cos = freqs.sin(), freqs.cos()

# 扩展到完整 head_dim 维度 [seq_len, head_dim]
sin = torch.cat((sin, sin), dim=-1)
cos = torch.cat((cos, cos), dim=-1)
```

> **实现细节**：
> - `position_ids.unsqueeze(-1)` 形状为 `[seq_len, 1]`
> - `inverse_frequencies.unsqueeze(0)` 形状为 `[1, head_dim // 2]`
> - 相乘后得到 `[seq_len, head_dim // 2]`
> - 最后 concatenate 扩展到 `[seq_len, head_dim]`

与 Transformers 官方实现的等价性：
```python
# Transformers库的做法（数学上完全等价）:
# freqs = position_ids.unsqueeze(-1) * inverse_frequencies.unsqueeze(0)
# emb = torch.cat((freqs, freqs), dim=-1)  # 先concat freqs
# cos, sin = emb.cos(), emb.sin()          # 再做cos，sin运算
```

两种方法数学上等价，区别在于 sin/cos 运算的时机。代码中先算 sin/cos 再 concat，官方先 concat 再算 sin/cos，结果一致。

### 3. 实现 rotate_half 函数

块式旋转的核心：将向量分成两半并交换位置，同时取负值：

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

> **数学意义**：
> - 输入向量：`[v1, v2]`（两半）
> - 旋转后：`[-v2, v1]`
> - 这种变换等价于复数旋转：$(v_1 + iv_2) \times e^{i\theta}$

### 4. 应用 RoPE 到 Q 和 K

最终调用 apply_rotary_pos_emb 函数（或直接用公式）：

```python
# split half of q,k
q1, q2 = (q_head[..., : q_head.shape[-1] // 2], q_head[..., q_head.shape[-1] // 2 :])
k1, k2 = (k_head[..., : k_head.shape[-1] // 2], k_head[..., k_head.shape[-1] // 2 :])

rotate_half_q = torch.cat((-q2, q1), dim=-1)
rotate_half_k = torch.cat((-k2, k1), dim=-1)

# Apply rope emb
q = q_head * cos + rotate_half_q * sin
k = k_head * cos + rotate_half_k * sin
```

> **数学解释**：
> - 原始向量乘以 cos: `[v1*cos, v2*cos]`
> - rotate_half 向量乘以 sin: `[-v2*sin, v1*sin]`
> - 两者相加: `[v1*cos - v2*sin, v2*cos + v1*sin]`
> - 这正是二维旋转矩阵的应用结果！

## Transformers 官方实现详解

### Qwen3 的 apply_rotary_pos_emb 实现

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze
            cos[position_ids] and sin[position_ids] so that they can be properly broadcasted
            to the dimensions of q and k. For example, note that cos[position_ids] and
            sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q
            and k have the shape [batch_size, heads, seq_len, head_dim], then setting
            unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids] broadcastable
            to the shapes of q and k. Similarly, if q and k have the shape
            [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using
        the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # 增加维度以支持广播
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

### 关键解析

**broadcast 机制**：`unsqueeze_dim` 参数用于适配不同张量形状

- Q/K 形状：`[batch_size, heads, seq_len, head_dim]` 或 `[batch_size, seq_len, heads, head_dim]`
- sin/cos 形状：`[seq_len, head_dim]`
- 通过 `unsqueeze(1)` 使 sin/cos 变为 `[1, seq_len, head_dim]`，可广播到 Q/K 的 `heads` 维度

**核心公式**：
```
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

这个公式实现了复数乘法 $(a + bi) \times (cos\theta + i sin\theta)$，其中：
- `q * cos` 对应实部乘以 cos
- `rotate_half(q) * sin` 对应旋转后的虚部乘以 sin

---
## 数学验证

向量旋转的有效性：

```
vector = [v1, v2]
rotate_half_vector = [-v2, v1]

# 旋转后结果
result = vector * cos + rotate_half_vector * sin
       = [v1 * cos - v2 * sin, v2 * cos + v1 * sin]
```

这正是二维旋转矩阵 $R_{\theta}$ 的应用，说明向量确实被旋转了角度 $\theta$！

## Reference

- [rope.py](file:///Users/ruipu/projects/4Fun/infra/model/rope.py)
