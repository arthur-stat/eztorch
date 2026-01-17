import numpy as np

from eztorch.typing import FloatArray


class Dropout:
    """Dropout layer with inverted scaling.

    - p: drop probability
    - training: if False, acts as identity
    """

    def __init__(self, p: float = 0.1, training: bool = True) -> None:
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p = float(p)
        self.training = training
        self._mask: FloatArray | None = None

    def __call__(self, x: FloatArray) -> FloatArray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.p
        self._mask = (np.random.rand(*x.shape) < keep_prob).astype(float)
        return (x * self._mask) / keep_prob

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._mask is None or not self.training or self.p == 0.0:
            return grad_output
        keep_prob = 1.0 - self.p
        return (grad_output * self._mask) / keep_prob

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []