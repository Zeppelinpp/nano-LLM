---
tags:
  - kv-cache
  - attention
createTime: 2026-02-07 17:00
Last Modified: 2026-07 17:00
---

# KV Cache Benchmark on Apple Silicon (MPS)

## Test Environment
- **Device**: Apple Silicon (MPS)
- **Hidden size**: 1024
- **Number of layers**: 20
- **Autoregressive steps**: 100
- **Batch size**: 1

## Output Consistency Verification

### Prefill Phase
- **Max absolute difference**: 0.00e+00
- **Outputs identical**: True

### Decoding Phase
| Step | Max Difference | Identical |
|------|----------------|-----------|
| Step 1 | 0.00e+00 | True |
| Step 2 | 0.00e+00 | True |
| Step 3 | 2.98e-08 | True |
| Step 4 | 1.49e-08 | True |
| Step 5 | 1.49e-08 | True |
| Step 6 | 2.98e-08 | True |
| Step 7 | 2.98e-08 | True |
| Step 8 | 2.98e-08 | True |
| Step 9 | 2.98e-08 | True |
| Step 10 | 2.98e-08 | True |

**Decoding Phase Summary**:
- **Maximum difference**: 2.98e-08
- **Average difference**: 2.09e-08
- **All steps within tolerance (1e-6)**: True

## Performance Benchmark Results

The benchmark demonstrates that KV caching provides significant performance benefits while maintaining numerical accuracy. All computations were executed on the MPS (Metal Performance Shaders) backend, ensuring optimal utilization of Apple Silicon hardware.

KV Cache provides **1.47x** speedup in decoding throughput!

### Benchmark Results

| Metric | With KV Cache | Without KV Cache | Speedup |
|--------|---------------|------------------|---------|
| Prefill time (s) | 0.0030 | 0.0052 | - |
| Total decode time (s) | 0.4346 | 0.6369 | - |
| Tokens/sec (decode) | 230.12 | 157.02 | 1.47x |
| Total time (s) | 0.7810 | 1.0989 | - |