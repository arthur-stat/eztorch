from typing import List

import numpy as np

from eztorch.typing import FloatArray, IntArray


class Embedding:
    """Simple embedding layer: maps token ids to vectors.

    Parameters
    - vocab_size: number of tokens in the vocabulary
    - d_model: embedding dimension
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        # weights: (vocab_size, d_model)
        self.weights: FloatArray = np.random.randn(self.vocab_size, self.d_model).astype(float) * 0.02
        self.grad_weights: FloatArray = np.zeros_like(self.weights)
        self._last_ids: IntArray | None = None

    def __call__(self, ids: IntArray) -> FloatArray:
        # ids: (batch, seq) int array
        self._last_ids = np.asarray(ids, dtype=np.int64)
        out = self.weights[self._last_ids]
        return out.astype(float)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        # grad_output: (batch, seq, d_model)
        if self._last_ids is None:
            raise RuntimeError("Embedding.backward called before forward")
        self.grad_weights[...] = 0.0
        ids_flat = self._last_ids.reshape(-1)
        grads_flat = np.asarray(grad_output, dtype=float).reshape(-1, self.d_model)
        # Accumulate gradients into corresponding rows
        np.add.at(self.grad_weights, ids_flat, grads_flat)
        # No gradient flows to ids; return zeros for interface
        return np.zeros_like(self._last_ids, dtype=float)

    def parameters(self) -> List[FloatArray]:
        return [self.weights]

    def grads(self) -> List[FloatArray]:
        return [self.grad_weights]