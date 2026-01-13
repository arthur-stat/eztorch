import numpy as np

from eztorch.layers.sequential import Sequential
from eztorch.optim.base import Optimizer
from eztorch.typing import FloatArray, IntArray


class SequentialModel:
    """Basic sequential model, which can be MLP, CNN etc."""

    def __init__(self, seq: Sequential) -> None:
        self.model = seq

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def forward(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def classify(self, x: FloatArray) -> IntArray:
        logits: FloatArray = self.forward(x)
        return np.argmax(logits, axis=1)

    def probability(self, x: FloatArray) -> FloatArray:
        logits: FloatArray = self.forward(x)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(logits - logits_max)
        probs: FloatArray = exps / np.sum(exps, axis=-1, keepdims=True)
        return probs

    def backward(self, x: FloatArray, y: IntArray) -> None:
        batch_size: int = x.shape[0]
        logits: FloatArray = self.forward(x)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_scores = np.exp(logits - logits_max)
        probs: FloatArray = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        grad_output: FloatArray = probs
        grad_output[np.arange(batch_size), y] -= 1
        grad_output /= batch_size

        for layer in reversed(self.model.layers):
            grad_output = getattr(layer, "backward")(grad_output)

    def train_step(self, x: FloatArray, y: IntArray, optimizer: Optimizer) -> None:
        self.backward(x, y)
        optimizer.step(self.model.parameters(), self.model.grads())