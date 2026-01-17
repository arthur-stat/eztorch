import inspect
from typing import Callable

import numpy as np

from eztorch.functions.losses import LossFunction, CrossEntropyLoss
from eztorch.optim.base import Optimizer, zeroed
from eztorch.typing import FloatArray, IntArray


class Trainer:

    def __init__(
        self,
        model: object,
        optimizer: Optimizer,
        forward: Callable[[FloatArray], FloatArray] | None = None,
        loss_fn: LossFunction | None = None,
        epsilon: float = 1e-12,
    ) -> None:
        self.model = model
        self._uses_model_forward = forward is None
        if forward is None:
            forward = getattr(model, "forward", model)
        self.forward = forward
        self.optimizer = optimizer
        self.loss_fn = loss_fn or CrossEntropyLoss(epsilon=epsilon)
        self.epsilon = epsilon

    def step(self, X_batch: FloatArray, y_batch: FloatArray | IntArray) -> float:
        train_mode = getattr(self.model, "train", None)
        if callable(train_mode):
            train_mode()
        preds: FloatArray = self.forward(X_batch)
        loss, grad_output = self.loss_fn(preds, y_batch)

        params = self._parameters()
        with zeroed(self._grads()):
            self._backward(grad_output, X_batch, y_batch)
            self.optimizer.step(params, self._grads())

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

    def _parameters(self) -> list[FloatArray]:
        params = getattr(self.model, "parameters", None)
        if callable(params):
            return params()
        nested = getattr(self.model, "model", None)
        if nested is not None:
            nested_params = getattr(nested, "parameters", None)
            if callable(nested_params):
                return nested_params()
        raise AttributeError("Trainer requires model.parameters().")

    def _grads(self) -> list[FloatArray]:
        grads = getattr(self.model, "grads", None)
        if callable(grads):
            return grads()
        nested = getattr(self.model, "model", None)
        if nested is not None:
            nested_grads = getattr(nested, "grads", None)
            if callable(nested_grads):
                return nested_grads()
        raise AttributeError("Trainer requires model.grads().")

    def _layers(self) -> list | None:
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            return layers
        nested = getattr(self.model, "model", None)
        if nested is not None:
            nested_layers = getattr(nested, "layers", None)
            if nested_layers is not None:
                return nested_layers
        return None

    def _try_model_backward(self, grad_output: FloatArray, X_batch: FloatArray, y_batch: FloatArray | IntArray) -> bool:
        if not self._uses_model_forward:
            return False
        backward = getattr(self.model, "backward", None)
        if not callable(backward):
            return False
        try:
            sig = inspect.signature(backward)
        except (TypeError, ValueError):
            return False
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(params) == 1:
            backward(grad_output)
            return True
        if len(params) == 2 and isinstance(self.loss_fn, CrossEntropyLoss):
            backward(X_batch, y_batch)
            return True
        return False

    def _backward(self, grad_output: FloatArray, X_batch: FloatArray, y_batch: FloatArray | IntArray) -> None:
        if self._try_model_backward(grad_output, X_batch, y_batch):
            return
        layers = self._layers()
        if layers is None:
            raise RuntimeError("Trainer requires a model with backward(...) or a .layers list.")
        g = grad_output
        for layer in reversed(layers):
            g = layer.backward(g)