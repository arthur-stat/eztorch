# eztorch

> [Click for English](README.md)

一个基于 NumPy 的轻量深度学习练习框架，借鉴 PyTorch 的风格。包含简单的 Transformer、训练器，以及可插拔的分词/向量化适配器，便于快速做 NLP Demo。

## 特性
- 基础组件：`Sequential`、`Linear`、`Conv2d`、`BatchNorm1d`、`LayerNorm`、`Residual`、常用激活（ReLU/LeakyReLU/Tanh/Sigmoid/Softmax）；
- 训练工具：`Trainer` 配合 SGD/Adam，支持 CE/MSE；使用 `Sequential` 组装模型（如 MLP/CNN 等）；
- 注意力/Transformer：多头自注意力层、编码器/译码器层、`DefaultTransformer`（正弦位置编码 + 生成头）；
- 示例与测试：分类/回归 MLP，NLP demo（`test/test_nlp.py`）。

## 安装
建议使用虚拟环境（例如 Windows 下 `python -m venv .venv && .venv\Scripts\activate`）后：
```bash
pip install -e .
```
若使用 `uv`，可替换为 `uv pip install -e .`。

## 快速上手
### MLP 分类（logits + 交叉熵）
```python
import numpy as np
from eztorch.layers.linear import Linear
from eztorch.layers.sequential import Sequential
from eztorch.functions.activations import ReLU
from eztorch.models.sequential_model import SequentialModel
from eztorch.optim.adam import Adam
from eztorch.utils.trainer import Trainer

X = np.random.randn(128, 2)
y = np.random.randint(0, 3, size=128)
mlp = SequentialModel(Sequential([Linear(2, 16), ReLU(), Linear(16, 3)]))
trainer = Trainer(model=mlp.model, forward=mlp.forward, optimizer=Adam(lr=0.01))
losses = trainer.fit(X, y, batch_size=32, max_steps=200, log_every=50)
print("final loss:", losses[-1])
```

### NLP demo（参见 `test/test_nlp.py`）
使用外部的空格分词 + one-hot 向量化适配器，跑一个极小的中英文翻译 Demo：
```bash
python test/test_nlp.py
```
训练几条样本后输出预测结果，并对训练集外的输入做推理。

## 测试 / 演示
- `test/test_mlp_classification.py`：moons 数据集分类，含归一化 + ReLU。
- `test/test_mlp_regression.py`：合成回归 + MSE。
- `test/test_nlp.py`：默认 Transformer + 外部分词/向量化的翻译 Demo。

## 说明
- 框架基于 NumPy，为实现简便起见计算图被嵌在层内，定位为学习/实验用途，而非生产训练。
- 交叉注意力已回传到编码器记忆，编码器参数可从解码器信号中学习。
