# OpenWebText 数据集下载和训练指南

## 快速开始（推荐 - 单个 304MB 文件）

完整数据集有 80 个文件（共 24GB），下载耗时且易失败。推荐先用单个文件测试：

### 步骤 1: 下载单个 parquet 文件（304MB）
```bash
uv run train/download_subset.py --output_dir ./data/openwebtext_subset
```

### 步骤 2: 准备数据集
```bash
uv run train/prepare_subset.py --data_dir ./data/openwebtext_subset
```

### 步骤 3: 运行预训练测试（1 epoch）
```bash
uv run train/pretrain.py \
    --data_dir ./data/openwebtext_subset \
    --batch_size 4 \
    --max_length 256 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --output_dir ./checkpoints_test
```

### 查看 TensorBoard
```bash
tensorboard --logdir ./checkpoints_test/tensorboard
```

预期结果：
- 下载时间：1-5 分钟（取决于网络）
- 训练时间：10-30 分钟（1 epoch，MPS）
- 验证训练流程是否正常工作

## 完整数据集（所有 80 个文件，24GB）

完整数据集下载注意事项：
- 总大小：约 24GB
- 下载时间：1-2 小时（取决于网络）
- 可能遇到网络超时，需要多次重试

### 下载完整数据集
```bash
uv run train/download_openwebtext.py --output_dir ./data/openwebtext
```

### 训练完整数据集
```bash
uv run train/pretrain.py \
    --data_dir ./data/openwebtext \
    --batch_size 4 \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --output_dir ./checkpoints
```

下载完成后，数据集结构：
```
data/openwebtext/
├── train/          # 训练数据
├── validation/     # 验证数据
└── dataset_info.txt
```

## 参数说明

### 模型参数
- `--vocab_size`: 词表大小 (默认: 50257，GPT-2)
- `--hidden_size`: 隐藏层大小 (默认: 768)
- `--num_layers`: Transformer 层数 (默认: 12)
- `--num_heads`: 注意力头数 (默认: 12)
- `--max_position_embeddings`: 最大位置编码长度 (默认: 1024)

### 训练参数 (MPS 优化)
- `--batch_size`: 批次大小 (推荐: 4-8，16GB MPS)
- `--max_length`: 序列长度 (推荐: 512-1024)
- `--gradient_accumulation_steps`: 梯度累积步数 (推荐: 4-8)
- `--learning_rate`: 学习率 (默认: 5e-5)
- `--num_epochs`: 训练轮数

### 数据参数
- `--data_dir`: 本地数据集目录 (必需)
- `--tokenizer_name`: Tokenizer 名称 (默认: gpt2)

## MPS 内存优化建议

对于 16GB MPS 设备：
1. **batch_size=4**: 保守设置，避免 OOM
2. **max_length=512**: 减少内存占用
3. **gradient_accumulation_steps=4**: 等效 batch_size=16
4. 如果仍然 OOM，可以进一步降低 batch_size 到 2

## 输出文件

训练完成后，`checkpoints/` 目录包含：
- `config.json`: 训练配置
- `checkpoint_epoch_{N}.pt`: 每个 epoch 的 checkpoint
- `best_model.pt`: 验证集上表现最好的模型

## 故障排查

### 问题: "Dataset not found"
**解决**: 先运行下载脚本
```bash
uv run train/download_openwebtext.py --output_dir ./data/openwebtext
```

### 问题: OOM (显存不足)
**解决**: 减小 batch_size 和 max_length
```bash
uv run train/pretrain.py \
    --batch_size 2 \
    --max_length 256 \
    ...
```

### 问题: 下载速度慢
**解决**: 可以添加 HF_TOKEN 提高限速
```bash
export HF_TOKEN=your_huggingface_token
uv run train/download_openwebtext.py ...
```

## 模型信息

- **参数量**: ~124M (768 hidden_size, 12 layers)
- **结构**: Transformer with RoPE, MLP, RMSNorm
- **词表**: GPT-2 tokenizer (50,257 tokens)
