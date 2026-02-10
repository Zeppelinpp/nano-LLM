# Gradient Accumulation & Batch Size

## 配置对比

假设：
- 数据集大小: 10,000 样本
- 目标 effective batch size: 32

### 方案对比表

| 方案 | Batch Size | Grad Accum | Effective Batch | Forward次数/epoch | 更新次数/epoch | 显存占用 |
|------|-----------|-----------|----------------|------------------|---------------|---------|
| A    | 32        | 1         | 32             | 312              | 312           |  高    |
| B    | 16        | 2         | 32             | 625              | 312           |  中    |
| C    | 8         | 4         | 32             | 1,250            | 312           |  低    |
| D    | 4         | 8         | 32             | 2,500            | 312           |  很低  |

**结论**：
- 所有方案的**参数更新次数相同**（训练效果理论上相同）
- Batch size 越小，显存占用越低
- Gradient accumulation 越大，训练速度越慢（更多 forward/backward）

## 显存占用详细分析

```python
# 模型显存分解（以 GPT-2 Small 为例，124M 参数）

组件                    | 显存占用                          | 受 batch_size 影响？
--------------------|--------------------------------|------------------
模型参数               | ~500MB (124M × 4 bytes)        |  固定
优化器状态 (AdamW)     | ~1GB (2倍参数用于动量)           |  固定
梯度                  | ~500MB (与参数相同)              |  固定
激活值                 | batch_size 相关                |  线性增长

# 示例计算（sequence_length=1024, hidden_size=768）
batch_size=4:  激活值 ~2GB
batch_size=8:  激活值 ~4GB
batch_size=16: 激活值 ~8GB
```

## 训练速度影响

```python
# 伪代码对比时间消耗

# 方案 A: batch_size=32, grad_accum=1
for batch in dataloader:  # 312 iterations
    forward(batch_32)      # 耗时: t
    backward()             # 耗时: t
    optimizer.step()       # 耗时: s
# 总耗时 ≈ 312 × (2t + s)

# 方案 D: batch_size=4, grad_accum=8
for batch in dataloader:  # 2500 iterations
    forward(batch_4)       # 耗时: t/8 (batch小了8倍)
    backward()             # 耗时: t/8
    if step % 8 == 0:
        optimizer.step()   # 耗时: s (每8步执行一次)
# 总耗时 ≈ 2500 × (2t/8) + 312 × s = 625t + 312s

# 对比：
# 方案 A: 624t + 312s
# 方案 D: 625t + 312s
# 结论：理论上几乎相同！但实际上方案D可能稍慢（overhead）
```

---

## 最佳实践建议

### 1. 如果显存充足
```python
batch_size = 32  # 尽可能大
gradient_accumulation_steps = 1  # 不需要累积
```
**优点**：训练最快

### 2. 如果显存不足
```python
# 先测试最大可用 batch_size
batch_size = 8  # 测试找到的最大值
gradient_accumulation_steps = 4  # 调整达到目标 effective batch
```
**优点**：显存友好

---

## TensorBoard 中如何查看？

```python
# 在训练开始时记录
writer.add_text("Config/effective_batch_size",
                f"{args.batch_size} × {args.gradient_accumulation_steps} = "
                f"{args.batch_size * args.gradient_accumulation_steps}")
```


