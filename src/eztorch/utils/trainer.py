from typing import Callable

import numpy as np

from eztorch.functions.losses import LossFunction, CrossEntropyLoss
from eztorch.layers.sequential import Sequential
from eztorch.optim.base import Optimizer, zeroed
from eztorch.typing import FloatArray, IntArray


class Trainer:

    def __init__(
        self,
        model: Sequential,
        forward: Callable[[FloatArray], FloatArray],
        optimizer: Optimizer,
        loss_fn: LossFunction | None = None,
        epsilon: float = 1e-12,
    ) -> None:
        self.model = model
        self.forward = forward
        self.optimizer = optimizer
        self.loss_fn = loss_fn or CrossEntropyLoss(epsilon=epsilon)
        self.epsilon = epsilon

    def step(self, X_batch: FloatArray, y_batch: FloatArray | IntArray) -> float:
        preds: FloatArray = self.forward(X_batch)
        loss, grad_output = self.loss_fn(preds, y_batch)

        with zeroed(self.model.grads()):
            for layer in reversed(self.model.layers):
                grad_output = layer.backward(grad_output)
            self.optimizer.step(self.model.parameters(), self.model.grads())

        return loss

    def fit(self, X: FloatArray, y: FloatArray | IntArray, batch_size: int, max_steps: int, log_every: int) -> list[float]:
        losses: list[float] = []
        for step in range(max_steps):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuff = X[indices]
            y_shuff = y[indices]

            batch_losses: list[float] = []
            for i in range(0, X_shuff.shape[0], batch_size):
                X_batch = X_shuff[i:i+batch_size]
                y_batch = y_shuff[i:i+batch_size]
                batch_losses.append(self.step(X_batch, y_batch))

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            if log_every and step % log_every == 0:
                losses.append(epoch_loss)
                print(f"Step {step}, Loss: {epoch_loss:.6f}")
        return losses