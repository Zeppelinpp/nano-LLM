import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=4):
        """
        Mock attention layer:
        - hidden_size : 4
        - decoding : bool
        - use_kv_cache: bool

            q = q_proj @ x (seq_len x hidden_size) + bias
            k = k_proj @ x + bias
            v = v_proj @ x + bias
        attention score = q.kT/sqrt(dk)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # KV cache for decoding phase
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        """Reset KV cache for new sequence"""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, decoding=False, use_kv_cache=False):
        """
        Forward pass for attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
               - In prefill phase: seq_len > 1 (full sequence)
               - In decoding phase: seq_len = 1 (single token)
            decoding: Boolean flag indicating decoding phase
            use_kv_cache: Boolean flag to enable/disable KV caching for benchmarking

        Returns:
            output: Attention output of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(x)  # (batch_size, seq_len, hidden_size)
        v = self.v_proj(x)  # (batch_size, seq_len, hidden_size)

        if decoding:
            # Decoding phase: single token input
            if use_kv_cache:
                # Use KV cache optimization
                if self.k_cache is None:
                    # First token in sequence
                    self.k_cache = k
                    self.v_cache = v
                else:
                    # Append to existing cache
                    self.k_cache = torch.cat([self.k_cache, k], dim=1)
                    self.v_cache = torch.cat([self.v_cache, v], dim=1)

                # Use full cache for attention
                k_full = self.k_cache  # (batch_size, cached_seq_len, hidden_size)
                v_full = self.v_cache  # (batch_size, cached_seq_len, hidden_size)

                # Compute attention with single query against full key cache
                scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(
                    self.hidden_size
                )
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v_full)
            else:
                # No KV cache - for benchmarking purposes
                # Note: In a real scenario without cache, you'd need the full past context
                # For this mock implementation, we'll just compute attention on current token
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                    self.hidden_size
                )
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)

        else:
            # Prefill phase: full sequence input
            # No caching needed, compute full attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)

            # Initialize cache with prefill tokens for subsequent decoding (if caching enabled)
            if use_kv_cache:
                self.k_cache = k
                self.v_cache = v

        return output


if __name__ == "__main__":
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
    print(
        f"This gives us (batch_size, seq_len, seq_len) - each query attends to all keys"
    )

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
