# Torch Basic

## 矩阵操作
- `torch.randn()`: 随机初始化对应形状的tensor
- `torch.transpose`: 矩阵转置
  - `dim1`, `dim2`: 这两个维度进行交换
```python
# tensor_1.shape -> (2, 3, 4)
transposed = tensor_1.transposed(-2, -1) # 倒数第一维度和倒数第二维度呼唤
# transposed.shape -> (2, 4, 3), 参数的顺序没有影响
```


