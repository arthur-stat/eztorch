# eztorch

> [点击此处跳转至中文版本](README.zh.md)

Minimal, NumPy-first deep learning playground with PyTorch-inspired APIs. Includes a tiny Transformer stack, simple trainers, and pluggable tokenization/vectorization for NLP demos.

## Features
- Core building blocks: `Sequential`, `Linear`, `Conv2d`, `BatchNorm1d`, `LayerNorm`, `Residual`, activations (ReLU, LeakyReLU, Tanh, Sigmoid, Softmax).
- Training utilities: `Trainer` with SGD/Adam, CE/MSE losses; build models with `Sequential`（MLP/CNN 等）。
- Attention/Transformer: multi-head self-attention layer, encoder/decoder layers, and a `DefaultTransformer` (encoder-decoder with sinusoidal PE and generator head).
- Tests/demos: classification/regression MLPs, toy translation demo (`test/test_nlp.py`).

## Installation
Use your preferred virtual environment and install only the core by default:
```bash
# pip
pip install -e .
# or uv
uv pip install -e .
```

Optional extras to run demos/tests without pulling heavy deps into the core:
```bash
# Visualization (matplotlib)
pip install -e .[viz]
# scikit-learn (datasets, demos)
pip install -e .[sklearn]
# tokenizers (modern NLP tokenization)
pip install -e .[nlp]

# everything used by examples/tests
pip install -e .[all]
```

## Quickstart
### MLP classification (logits + cross entropy)
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

### Transformer translation demo (see `test/test_nlp.py`)
Uses the simple whitespace tokenizer from `src/external/` and the built-in `DefaultTransformer`:
```bash
python test/test_nlp.py
```
This trains on a few tiny source/target pairs and prints decoded predictions, plus an inference on an unseen source sentence.

## Tests / demos
- `test/test_mlp_classification.py` – moons dataset classification with norms + ReLU.
- `test/test_mlp_regression.py` – synthetic regression with MSE.
- `test/test_nlp.py` – tiny translation toy using the default Transformer and external tokenizer/vectorizer.
