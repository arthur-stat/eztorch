import numpy as np

from eztorch.typing import FloatArray


class LeakyReLU:

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input = x
        return np.where(x > 0, x, self.negative_slope * x)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        grad_input = np.where(self.input > 0, 1.0, self.negative_slope) * grad_output
        return grad_input

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []