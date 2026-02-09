import torch

# q,k,v -> (batch_size, seq_len, num_heads, head_dim)

# Setup
torch.device = "mps"
bz, seq_len, num_heads, head_dim = 2, 10, 12, 64

q_head = torch.randn(bz, seq_len, head_dim)
k_head = torch.randn(bz, seq_len, head_dim)

rope_base = 10000
position_ids = torch.arange(seq_len)
inverse_freqencies = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2) / head_dim))

print(f"posistion_ids's shape: {position_ids.shape}")
print(f"inverse_freqencies's shape: {inverse_freqencies.shape}")
# position_ids's shape -> [10] and inv_freq's shape -> [32]

sin, cos = (
    (position_ids.unsqueeze(-1) * inverse_freqencies.unsqueeze(0)).sin(),
    (position_ids.unsqueeze(-1) * inverse_freqencies.unsqueeze(0)).cos(),
)
print(f"sin, cos shape(arange with step:2): {sin.shape}")
print("Concat (sin, sin) -> shape[-1] == head_dim ...")
sin = torch.cat((sin, sin), dim=-1)
cos = torch.cat((cos, cos), dim=-1)
print(f"Concated sin: {sin}")
# [10, 1] @ [1, 32] -> [10, 32]
"""
Transformers库的做法:
freqs = position_ids.unsqueeze(-1) * inverse_freqencies.unsqueeze(0)
emb = torch.cat((freqs, freqs), dim=-1) # 先concat freqs
cos, sin = emb.cos(), emb.sin() # 再做cos，sin运算
数学上完全等价
"""

# split half of q,k
q1, q2 = (q_head[..., : q_head.shape[-1] // 2], q_head[..., q_head.shape[-1] // 2 :])
k1, k2 = (k_head[..., : k_head.shape[-1] // 2], k_head[..., k_head.shape[-1] // 2 :])
# print(q1.shape)
# print(q2.shape)
# [2, 10, 32]
if torch.allclose(torch.cat((q1, q2), dim=-1), q_head):
    print("Identical")
rotate_half_q = torch.cat((-q2, q1), dim=-1)
rotate_half_k = torch.cat((-k2, k1), dim=-1)

"""
vector = [v1, v2]
rotate_half_vector = [-v2, v1]
vector * cos + rotate_half_vector = [v1 * cos - v2 * sin, v2 * sin - v1 * cos]
Vector rotated by a angel !!
"""

# Apply rope emb
q = q_head * cos + rotate_half_q * sin
k = k_head * cos + rotate_half_k * sin
