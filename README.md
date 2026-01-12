# eztorch

> [点击此处跳转至中文版本](README.zh.md)

Minimal, NumPy-first deep learning playground with PyTorch-inspired APIs. Includes a tiny Transformer stack, simple trainers, and pluggable tokenization/vectorization for NLP demos.

## Features
- Core building blocks: `Sequential`, `Linear`, `Conv2d`, `BatchNorm1d`, `LayerNorm`, `Residual`, activations (ReLU, LeakyReLU, Tanh, Sigmoid, Softmax).
- Training utilities: `Trainer` with SGD/Adam, CE/MSE losses, logits-first `MLP` wrapper.
- Attention/Transformer: multi-head self-attention layer, encoder/decoder layers, and a `DefaultTransformer` (encoder-decoder with sinusoidal PE and generator head).
- Tests/demos: classification/regression MLPs, toy translation demo (`test/test_nlp.py`).

## Installation
Use your preferred virtual environment (e.g., `python -m venv .venv && .venv\Scripts\activate` on Windows) and install:
```bash
pip install -e .
```
If you use `uv`, you can replace with `uv pip install -e .`.

## Quickstart
### MLP classification (logits + cross entropy)
```python
import numpy as np
from eztorch.layers.linear import Linear
from eztorch.layers.sequential import Sequential
from eztorch.functions.activations import ReLU
from eztorch.models.mlp import MLP
from eztorch.optim.adam import Adam
from eztorch.utils.trainer import Trainer

X = np.random.randn(128, 2)
y = np.random.randint(0, 3, size=128)
model = MLP(Sequential([Linear(2, 16), ReLU(), Linear(16, 3)]))
trainer = Trainer(model=model.model, forward=model.forward, optimizer=Adam(lr=0.01))
losses = trainer.fit(X, y, batch_size=32, max_steps=200, log_every=50)
print("final loss:", losses[-1])
```

### Transformer translation demo (see `test/test_nlp.py`)
Uses the simple whitespace tokenizer/one-hot vectorizer from `src/external/` and the built-in `DefaultTransformer`:
```bash
python test/test_nlp.py
```
This trains on a few tiny source/target pairs and prints decoded predictions, plus an inference on an unseen source sentence.

## Tests / demos
- `test/test_mlp_classification.py` – moons dataset classification with norms + ReLU.
- `test/test_mlp_regression.py` – synthetic regression with MSE.
- `test/test_nlp.py` – tiny translation toy using the default Transformer and external tokenizer/vectorizer.

## Notes
- The framework is NumPy-based and keeps graphs inside layer implementations; it is intended for learning and experimentation, not production-scale training.
- Cross-attention currently does not propagate gradients back to encoder memory; extend `MultiHeadCrossAttention` if you need that behavior.
