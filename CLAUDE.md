# Claude Code 使用指南

## Python 脚本执行

在本项目中，使用 `uv` 工具来运行 Python 脚本。`uv` 是一个快速的 Python 包安装器和解析器。

### 基本用法

```bash
# 使用 uv run 执行 Python 脚本
uv run /path/to/script.py

# 例如，运行 attention 模块中的脚本
uv run attention/attn.py

# 运行主脚本
uv run main.py
```

### 项目结构

- `attention/` - 注意力机制相关代码
- `model/` - 模型定义
- `main.py` - 主入口文件

### 开发工作流

1. 编写或修改 Python 脚本
2. 使用 `uv run` 命令执行脚本进行测试
3. 查看输出结果并进行调试

### 示例

```bash
# 测试注意力层实现
uv run attention/attn.py

# 运行主程序
uv run main.py
```

确保在项目根目录下执行这些命令，这样 `uv` 可以正确识别项目的 `pyproject.toml` 和依赖配置。
