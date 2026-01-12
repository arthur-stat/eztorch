from __future__ import annotations

from typing import Callable

import numpy as np

from eztorch.typing import FloatArray, IntArray
from eztorch.layers.sequential import Sequential
from eztorch.optim.base import Optimizer, zeroed


class Trainer:

    def __init__(
        self,
        model: Sequential,
        forward: Callable[[FloatArray], FloatArray],
        optimizer: Optimizer,
        epsilon: float = 1e-12,
    ) -> None:
        self.model = model
        self.forward = forward
        self.optimizer = optimizer
        self.epsilon = epsilon

    def step(self, X_batch: FloatArray, y_batch: IntArray) -> float:
        logits: FloatArray = self.forward(X_batch)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        e_x = np.exp(logits - logits_max)
        probs: FloatArray = e_x / e_x.sum(axis=-1, keepdims=True)

        batch_size: int = X_batch.shape[0]
        loss = float(-np.mean(np.log(probs[np.arange(batch_size), y_batch] + self.epsilon)))

        grad_output: FloatArray = probs.copy()
        grad_output[np.arange(batch_size), y_batch] -= 1
        grad_output /= batch_size

        with zeroed(self.model.grads()):
            for layer in reversed(self.model.layers):
                grad_output = layer.backward(grad_output)
            self.optimizer.step(self.model.parameters(), self.model.grads())

        return loss

    def fit(self, X: FloatArray, y: IntArray, batch_size: int, max_steps: int, log_every: int = 10000) -> list[float]:
        losses: list[float] = []
        for step in range(max_steps):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuff = X[indices]
            y_shuff = y[indices]

            last_loss = 0.0
            for i in range(0, X_shuff.shape[0], batch_size):
                X_batch = X_shuff[i:i+batch_size]
                y_batch = y_shuff[i:i+batch_size]
                last_loss = self.step(X_batch, y_batch)

            if log_every and step % log_every == 0:
                losses.append(last_loss)
                print(f"Step {step}, Loss: {last_loss:.6f}")
        return losses