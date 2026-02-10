# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based infrastructure project focused on implementing and benchmarking attention mechanisms with KV caching optimization. The codebase demonstrates the difference between prefill and decoding phases in transformer-based models, with specific focus on performance optimization through KV caching.

## Development Commands

### Environment Setup
- Python version: >=3.12 (specified in `.python-version`)
- Package manager: `uv` (ultra-fast Python package installer)
- Dependencies: `torch>=2.10.0`, `transformers>=5.1.0`

### Running Code
```bash
# Run any Python script using uv
uv run /path/to/script.py

# Examples:
uv run attention/attn.py          # Run attention layer benchmark
uv run main.py                    # Run main application
```

### Testing
```bash
# Run specific test files
uv run test/test_attn.py          # Test attention layer functionality
uv run test/test_kvcache.py       # Test KV cache implementation

# The test files contain demonstrations and benchmarks rather than traditional unit tests
```

## Code Architecture

### Core Components
- **`attention/attn.py`**: Main implementation of `AttentionLayer` class with KV caching support
  - Supports both prefill phase (full sequence processing) and decoding phase (single token with cache)
  - Includes `use_kv_cache` parameter for performance benchmarking
  - Implements causal masking for autoregressive generation
  - Contains benchmark functions for measuring KV cache performance benefits

- **`MultiLayerAttention`**: Stack of multiple attention layers for realistic model simulation
  - Used in benchmarking to demonstrate real-world performance differences

### Key Concepts
- **Prefill Phase**: Processes complete input sequence, initializes KV cache
- **Decoding Phase**: Processes single tokens, utilizes cached KV from previous steps
- **KV Caching**: Critical optimization that avoids recomputing key/value projections for historical tokens
- **Causal Masking**: Ensures autoregressive property by preventing attention to future tokens

### Directory Structure
- `attention/`: Attention mechanism implementations
- `test/`: Test and demonstration scripts
- `docs/`: Documentation on PyTorch basics and operations
- `model/`: Model definitions (currently minimal)

## Apple Silicon Support

The codebase properly supports Apple Silicon (M1/M2/M3/M4) chips through Metal Performance Shaders (MPS):
- Automatically detects and uses MPS backend when available
- Falls back to CPU if MPS is not available
- Device selection logic: `mps` → `cuda` → `cpu`

## Benchmarking

The primary benchmark compares:
- **With KV Cache**: Efficient decoding with cached key/value projections
- **Without KV Cache**: Naive approach recomputing all projections each step

Expected performance improvement: 2-3x speedup on CPU, 1.2-1.5x on MPS (varies by hardware).

## Documentation Style Guide

When creating or updating documentation in the `docs/` directory:
- **No emojis**: Do not use emojis in documentation files
- **Clean formatting**: Use markdown formatting (headers, lists, code blocks, tables)
- **Visual aids**: Use ASCII diagrams, flow charts, or algorithmic art visualizations when helpful
- **YAML frontmatter**: Include tags, createTime, and Last Modified metadata
- **Technical precision**: Focus on clear, concise technical explanations