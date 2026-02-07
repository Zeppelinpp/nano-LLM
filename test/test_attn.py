import torch
from attention.attn import AttentionLayer


# Demonstrate torch.transpose operation
print("=== torch.transpose demonstration ===")

# Create a sample tensor similar to our attention use case
# Shape: (batch_size, seq_len, hidden_size) = (2, 3, 4)
batch_size, seq_len, hidden_size = 2, 3, 4
k = torch.randn(batch_size, seq_len, hidden_size)
print(f"Original K tensor shape: {k.shape}")
print(f"K tensor:\n{k}\n")

# In attention, we need K^T for matrix multiplication Q @ K^T
# The last two dimensions need to be transposed: (..., seq_len, hidden_size) -> (..., hidden_size, seq_len)
k_transposed = k.transpose(-2, -1)
print(f"Transposed K tensor shape: {k_transposed.shape}")
print(f"K transposed:\n{k_transposed}\n")

# Verify the transpose operation
print("Verification:")
print(f"k[0, 1, 2] = {k[0, 1, 2].item():.4f}")
print(f"k_transposed[0, 2, 1] = {k_transposed[0, 2, 1].item():.4f}")
print(f"Are they equal? {torch.allclose(k[0, 1, 2], k_transposed[0, 2, 1])}")

# Show how it's used in attention computation
print("\n=== Attention computation example ===")
q = torch.randn(batch_size, seq_len, hidden_size)
print(f"Q shape: {q.shape}")
print(f"K^T shape: {k_transposed.shape}")

# Compute attention scores: Q @ K^T
attention_scores = torch.matmul(q, k_transposed)
print(f"Attention scores shape: {attention_scores.shape}")
print(f"This gives us (batch_size, seq_len, seq_len) - each query attends to all keys")

# Simple test with the AttentionLayer
print("\n=== Testing AttentionLayer ===")
attn_layer = AttentionLayer(hidden_size=hidden_size)

# Prefill phase
x_prefill = torch.randn(1, 5, hidden_size)  # batch=1, seq_len=5
output_prefill = attn_layer(x_prefill, decoding=False, use_kv_cache=True)
print(f"Prefill output shape: {output_prefill.shape}")

# Decoding phase
x_decode = torch.randn(1, 1, hidden_size)  # batch=1, seq_len=1
output_decode = attn_layer(x_decode, decoding=True, use_kv_cache=True)
print(f"Decoding output shape: {output_decode.shape}")
print(
    f"Cache size after decoding: K={attn_layer.k_cache.shape}, V={attn_layer.v_cache.shape}"
)

# test_tensor = torch.randn(2, 3, 4)
# transposed_test_tensor = test_tensor.transpose(1, 0)

# print(
#     f"Original: \n{test_tensor.shape}\n, Transposed:\n{transposed_test_tensor.shape}\n"
# )
