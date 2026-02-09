import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Applies rotary position encoding to query and key tensors
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply RoPE to input tensor

        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            seq_len: Sequence length

        Returns:
            Tensor with RoPE applied
        """
        device = x.device
        dtype = x.dtype

        # Create position indices
        positions = torch.arange(seq_len, device=device, dtype=dtype)

        # Create theta for each dimension (apply to half of the dimensions)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))

        # Calculate angles: (seq_len, dim/2)
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)

        # Create sin and cos
        sin = angles.sin()
        cos = angles.cos()

        # Reshape for broadcasting
        sin = sin.view(seq_len, 1, self.dim // 2)
        cos = cos.view(seq_len, 1, self.dim // 2)

        # Split x into two halves
        x_reshaped = x.view(*x.shape[:-1], 2, self.dim // 2)
        x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Combine back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-2)
        rotated = rotated.view_as(x)

        return rotated


class AttentionBlock(nn.Module):
    """
    Decoder-only Causal AttentionBlock with optional KV caching
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Rotary Position Embedding
        if use_rope:
            self.rope = RoPE(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_kv_cache: bool = False,
        kv_cache: dict = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of AttentionBlock

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Attention mask (optional)
            use_kv_cache: Whether to use KV caching
            kv_cache: Dictionary containing cached keys and values

        Returns:
            Tuple of (output_tensor, updated_kv_cache)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch, seq_len, embed_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q, seq_len)
            k = self.rope(k, seq_len)

        # Handle KV caching
        if use_kv_cache and kv_cache is not None:
            # Append new keys and values to cache
            if 'keys' in kv_cache and 'values' in kv_cache:
                k = torch.cat([kv_cache['keys'], k], dim=1)
                v = torch.cat([kv_cache['values'], v], dim=1)

            # Update cache
            kv_cache = {'keys': k, 'values': v}

        # Transpose for attention calculation: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply external attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output, kv_cache if use_kv_cache else None


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Apply normalization and scale
        return x * rms * self.weight


class GeneralLLM(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class RMSNorm(nn.Module):



class GeneralLLM(nn.Module):
    def __init__(self):
        
