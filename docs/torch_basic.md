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

- `torch.triu`: Triangular Upper, 保留主对角线右上方部分
```python
mask = torch.triu(
  torch.ones(3, 3), diagonal=1
) # diagonal: 1 不包含对角线，0包含对角线
"""
torch.ones(3,3):
[1,1,1
 1,1,1
 1,1,1]

diagonal:1
[0,1,1
 0,0,1
 0,0,0]

diagonal:0
[1,1,1
 0,1,1
 0,0,1]
"""
```

- `torch.cat`: 拼接tensor，输入为
  - `Tensor List`: 要拼接的tensor列表
  - `dim`: 按哪个维度拼接

```python
tensor_1 = torch.randn(3, 4)
tensor_2 = torch.randn(1, 4)

concated = torch.cat([tensor_1, tensor_2], dim=0) # 在第一个维度拼接
# concated.shape -> (4, 4)
# 注意拼接有顺序，按照tensor列表的顺序
```

