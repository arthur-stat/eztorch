import numpy as np
from eztorch.typing import FloatArray


class Sigmoid:
    
    def __call__(self, x: FloatArray) -> FloatArray:
        self.output: FloatArray = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output: FloatArray, learning_rate: float) -> FloatArray:
        grad_input: FloatArray = grad_output * self.output * (1 - self.output)
        return grad_input

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []