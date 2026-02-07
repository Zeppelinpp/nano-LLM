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

            # Casual Mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()  # triangaular upper including diagonal -> bool
            scores.masked_fill_(mask, float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)

            # Initialize cache with prefill tokens for subsequent decoding (if caching enabled)
            if use_kv_cache:
                self.k_cache = k
                self.v_cache = v
            else:
                # When not using cache, don't store anything
                self.k_cache = None
                self.v_cache = None

        return output


import time
import torch.nn.functional as F


class MultiLayerAttention(nn.Module):
    """Stack of multiple attention layers"""

    def __init__(self, hidden_size=1024, num_layers=20):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionLayer(hidden_size) for _ in range(num_layers)]
        )

    def reset_cache(self):
        """Reset cache for all layers"""
        for layer in self.layers:
            layer.reset_cache()

    def forward(self, x, decoding=False, use_kv_cache=False):
        """Forward through all layers"""
        for layer in self.layers:
            x = layer(x, decoding=decoding, use_kv_cache=use_kv_cache)
        return x


def verify_output_consistency():
    """
    Verify that outputs are identical between KV cache and non-cache modes
    """
    print("=== Output Consistency Verification ===")

    # Use CPU for reproducible results (MPS/CUDA can have non-deterministic behavior)
    device = torch.device("cpu")
    torch.manual_seed(42)  # Ensure reproducible results

    # Initialize two identical models
    model_cache = MultiLayerAttention(hidden_size=64, num_layers=4).to(device)
    model_no_cache = MultiLayerAttention(hidden_size=64, num_layers=4).to(device)

    # Copy weights to ensure identical models
    model_no_cache.load_state_dict(model_cache.state_dict())

    batch_size = 1
    prefill_len = 5
    hidden_size = 64

    # Generate deterministic input
    x_prefill = torch.randn(batch_size, prefill_len, hidden_size).to(device)

    # Test prefill phase consistency
    print("Testing prefill phase consistency...")
    model_cache.reset_cache()
    model_no_cache.reset_cache()

    with torch.no_grad():
        output_cache = model_cache(x_prefill, decoding=False, use_kv_cache=True)
        output_no_cache = model_no_cache(x_prefill, decoding=False, use_kv_cache=False)

    prefill_diff = torch.max(torch.abs(output_cache - output_no_cache)).item()
    print(f"Prefill max absolute difference: {prefill_diff:.2e}")
    print(
        f"Prefill outputs identical: {torch.allclose(output_cache, output_no_cache, atol=1e-6)}"
    )

    # Test decoding phase consistency
    print("\nTesting decoding phase consistency...")
    torch.manual_seed(123)  # Different seed for decoding tokens
    decoding_outputs_cache = []
    decoding_outputs_no_cache = []

    # With cache
    model_cache.reset_cache()
    _ = model_cache(x_prefill, decoding=False, use_kv_cache=True)  # Prefill

    # Without cache - maintain full sequence
    full_sequence = x_prefill.clone()

    for step in range(10):  # Test 10 decoding steps
        x_token = torch.randn(batch_size, 1, hidden_size).to(device)

        # With cache
        with torch.no_grad():
            out_cache = model_cache(x_token, decoding=True, use_kv_cache=True)
        decoding_outputs_cache.append(out_cache.clone())

        # Without cache: reset model state and process full sequence
        model_no_cache.reset_cache()  # Reset any internal state
        full_sequence = torch.cat([full_sequence, x_token], dim=1)
        with torch.no_grad():
            out_full = model_no_cache(full_sequence, decoding=False, use_kv_cache=False)
        # Extract only the last token output
        out_no_cache = out_full[:, -1:, :]
        decoding_outputs_no_cache.append(out_no_cache.clone())

    # Compare decoding outputs
    all_differences = []
    for i, (out_cache, out_no_cache) in enumerate(
        zip(decoding_outputs_cache, decoding_outputs_no_cache)
    ):
        diff = torch.max(torch.abs(out_cache - out_no_cache)).item()
        all_differences.append(diff)
        print(
            f"Step {i + 1}: max diff = {diff:.2e}, identical = {torch.allclose(out_cache, out_no_cache, atol=1e-6)}"
        )

    max_decoding_diff = max(all_differences)
    avg_decoding_diff = sum(all_differences) / len(all_differences)
    print(f"\nDecoding phase summary:")
    print(f"Max difference across all steps: {max_decoding_diff:.2e}")
    print(f"Avg difference across all steps: {avg_decoding_diff:.2e}")
    print(f"All steps within tolerance (1e-6): {max_decoding_diff < 1e-6}")

    return {
        "prefill_consistent": torch.allclose(output_cache, output_no_cache, atol=1e-6),
        "decoding_consistent": max_decoding_diff < 1e-6,
        "prefill_max_diff": prefill_diff,
        "decoding_max_diff": max_decoding_diff,
    }


def benchmark():
    """
    Test out throughputs difference with or without kv cache

    Configuration:
    - hidden_size: 1024
    - num_layers: 20
    - autoregressive steps: 100
    - batch_size: 1 (typical for inference)
    """
    print("=== KV Cache Benchmark ===")
    print(f"Configuration:")
    print(f"- Hidden size: 1024")
    print(f"- Number of layers: 20")
    print(f"- Autoregressive steps: 100")
    print(f"- Batch size: 1")
    print()

    # Set device - properly handle Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = MultiLayerAttention(hidden_size=1024, num_layers=20).to(device)

    # Prefill phase - generate initial context
    batch_size = 1
    prefill_len = 10  # Initial context length
    x_prefill = torch.randn(batch_size, prefill_len, 1024).to(device)

    # Warm up both modes
    print("Warming up models...")
    with torch.no_grad():
        # Warm up with cache
        model.reset_cache()
        _ = model(x_prefill, decoding=False, use_kv_cache=True)
        for _ in range(5):
            x_token = torch.randn(batch_size, 1, 1024).to(device)
            _ = model(x_token, decoding=True, use_kv_cache=True)

        # Warm up without cache
        model.reset_cache()
        _ = model(x_prefill, decoding=False, use_kv_cache=False)
        for _ in range(5):
            x_token = torch.randn(batch_size, 1, 1024).to(device)
            _ = model(x_token, decoding=True, use_kv_cache=False)

    # Benchmark with KV cache
    print("\nRunning benchmark with KV cache...")
    model.reset_cache()

    # Prefill with cache
    start_time = time.time()
    with torch.no_grad():
        output = model(x_prefill, decoding=False, use_kv_cache=True)
    prefill_time_cache = time.time() - start_time

    # Autoregressive generation with cache
    total_decode_time_cache = 0
    tokens_generated_cache = 0

    start_total_time = time.time()
    with torch.no_grad():
        for step in range(100):
            x_token = torch.randn(batch_size, 1, 1024).to(device)
            start_step = time.time()
            output = model(x_token, decoding=True, use_kv_cache=True)
            step_time = time.time() - start_step
            total_decode_time_cache += step_time
            tokens_generated_cache += 1
    total_time_cache = time.time() - start_total_time

    # Benchmark without KV cache
    print("Running benchmark without KV cache...")
    model.reset_cache()

    # Prefill without cache
    start_time = time.time()
    with torch.no_grad():
        output = model(x_prefill, decoding=False, use_kv_cache=False)
    prefill_time_no_cache = time.time() - start_time

    # Autoregressive generation without cache
    # Note: Without cache, we need to recompute everything each time
    # For fair comparison, we'll simulate the same computational load
    total_decode_time_no_cache = 0
    tokens_generated_no_cache = 0

    # We'll maintain a growing sequence to simulate the same attention context
    current_sequence = x_prefill.clone()

    start_total_time = time.time()
    with torch.no_grad():
        for step in range(100):
            x_token = torch.randn(batch_size, 1, 1024).to(device)
            # Append new token to sequence (simulating full context)
            current_sequence = torch.cat([current_sequence, x_token], dim=1)

            start_step = time.time()
            # Process the entire sequence (this is what would happen without caching)
            output = model(current_sequence, decoding=False, use_kv_cache=False)
            step_time = time.time() - start_step
            total_decode_time_no_cache += step_time
            tokens_generated_no_cache += 1
    total_time_no_cache = time.time() - start_total_time

    # Calculate metrics
    tokens_per_sec_cache = (
        tokens_generated_cache / total_decode_time_cache
        if total_decode_time_cache > 0
        else 0
    )
    tokens_per_sec_no_cache = (
        tokens_generated_no_cache / total_decode_time_no_cache
        if total_decode_time_no_cache > 0
        else 0
    )

    speedup = (
        tokens_per_sec_cache / tokens_per_sec_no_cache
        if tokens_per_sec_no_cache > 0
        else float("inf")
    )

    # Print results
    print("\n=== BENCHMARK RESULTS ===")
    print(
        f"{'Metric':<25} {'With KV Cache':<20} {'Without KV Cache':<20} {'Speedup':<10}"
    )
    print("-" * 80)
    print(
        f"{'Prefill time (s)':<25} {prefill_time_cache:<20.4f} {prefill_time_no_cache:<20.4f} {'-':<10}"
    )
    print(
        f"{'Total decode time (s)':<25} {total_decode_time_cache:<20.4f} {total_decode_time_no_cache:<20.4f} {'-':<10}"
    )
    print(
        f"{'Tokens/sec (decode)':<25} {tokens_per_sec_cache:<20.2f} {tokens_per_sec_no_cache:<20.2f} {speedup:<10.2f}x"
    )
    print(
        f"{'Total time (s)':<25} {total_time_cache:<20.4f} {total_time_no_cache:<20.4f} {'-':<10}"
    )

    print(f"\nKV Cache provides {speedup:.2f}x speedup in decoding throughput!")

    return {
        "with_cache": {
            "prefill_time": prefill_time_cache,
            "decode_time": total_decode_time_cache,
            "tokens_per_sec": tokens_per_sec_cache,
            "total_time": total_time_cache,
        },
        "without_cache": {
            "prefill_time": prefill_time_no_cache,
            "decode_time": total_decode_time_no_cache,
            "tokens_per_sec": tokens_per_sec_no_cache,
            "total_time": total_time_no_cache,
        },
        "speedup": speedup,
    }


if __name__ == "__main__":
    # First verify output consistency
    consistency_results = verify_output_consistency()
    print()

    # Then run performance benchmark
    benchmark()
