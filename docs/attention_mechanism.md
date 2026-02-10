---
tags:
  - algorithm
  - torch
  - llm
  - attention
createTime: 2026-02-11 02:05
Last Modified: 2026-02-11 02:08
---
# Attention Mechanism

---

## MHA - Multi Heads Attention

- KV 头数 = Q 头数

```python
class MHA(nn.Module):
  def __init__(self, ...):
    ...
    # Q, K, V
	self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
    self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
    self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim)
    ...
  def forward(self, ...):
    ...
    q = q.view(batch_size, seq_len, self.num_heads, self,head_dim)
    k = k.view(batch_size, seq_len, self.num_heads, self,head_dim)
    v = v.view(batch_size, seq_len, self.num_heads, self,head_dim)
```

---

## MQA - Multi Query Attention

- 多个Q，KV只有一个头

```python
class MQA(nn.Module):
  def __init__(self, ...):
    self.num_kv_heads = 1
    self.num_queries_per_kv = self.num_heads // 1
    
    self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    self.k_proj = nn.Linear(self.embed_dim, 1 * self.head_dim, bias=False)
    self.v_proj = nn.Linear(self.embed_dim, 1 * self.head_dim, bias=False)
    ...
  def forward(self, ...):
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, 1, self.head_dim)
    v = v.view(batch_size, seq_len, 1, self.head_dim)
    
    k = repeat_kv(k, self.num_queries_per_kv)
    v = repead_kv(v, self.num_queries_per_kv)
    ...
  def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
      return x
    return x.repead_interleave(n_rep, dim=2) # x.shape -> (bz, sq, 1, head_dim) -> (bz, sq, num_heads, head_dim)
```

---

## GQA - Group Query Attention

- 多组 KV 头

```python
class MQA(nn.Module):
  def __init__(self, ...):
    self.num_kv_heads = num_kv_heads
    self.num_queries_per_kv = self.num_heads // self.num_kv_heads
    
    self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
    ...
  def forward(self, ...):
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    
    k = repeat_kv(k, self.num_queries_per_kv)
    v = repead_kv(v, self.num_queries_per_kv)
    ...
  def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
      return x
    return x.repead_interleave(n_rep, dim=2) # x.shape -> (bz, sq, num_kv_heads, head_dim) -> (bz, sq, num_heads, head_dim)
```

---

## 总结

|                | MHA               | MQA                | GQA                      |
| -------------- | ----------------- | ------------------ | ------------------------ |
| num_kv_heads   | 等于 num_heads    | 等于 1             | 介于二者之间（如 8）     |
| k/v_proj 维度  | embed_dim         | head_dim           | num_kv_heads * head_dim  |
| repeat_kv 逻辑 | 不执行（n_rep=1） | 将 1 个头重复 N 次 | 将 G 个头每个重复 N/G 次 |

