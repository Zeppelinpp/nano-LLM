"""
Exploring torch:
- view: change shape but don't copy
- transpose: swap dim of matrix, ignore order of input args
- tril or triu: lower or upper triangular matrix
- @: matmul
- contiguous: check if the tensor is contiguous in GPU memory
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


device = "mps"
tensor = torch.randn(2, 10, 1024).to(device)
print(f"Original shape: {tensor.shape}")

c_attn = nn.Linear(1024, 3 * 1024, bias=False).to(device)

qkv = c_attn(tensor)
print(f"After attn, combined qkv's shape {qkv.shape}")

# Split qkv to separate q, k, v
q, k, v = qkv.split(1024, dim=2)
q = q.view(2, 10, 8, 1024 // 8).transpose(1, 2)
k = k.view(2, 10, 8, 1024 // 8).transpose(1, 2)
v = v.view(2, 10, 8, 1024 // 8).transpose(1, 2)
print(f"After split attention heads, each heads: {q.shape}")
# batch_size, num_heads, seq_len, head_dim

att = q @ k.transpose(2, 3) * (1.0 / math.sqrt(k.size(-1)))
casual_mask = torch.tril(torch.ones(10, 10)).view(1, 1, 10, 10).bool().to(device)
att = att.masked_fill(casual_mask, float("-inf"))
att = F.softmax(att, dim=-1)
y = att @ v
print(f"After attention scores computation and apply to v on each heads: {y.shape}")

y = y.transpose(1, 2).contiguous().view(2, 10, 1024)
print(f"Final shape after transform it back via transpose & view: {y.shape}")
