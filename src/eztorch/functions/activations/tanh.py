import numpy as np

from eztorch.typing import FloatArray


class Tanh:

    def __call__(self, x: FloatArray) -> FloatArray:
        self.output: FloatArray = np.tanh(x)
        return self.output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        return grad_output * (1.0 - self.output ** 2)

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []