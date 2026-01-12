import numpy as np
from eztorch.typing import FloatArray


class Softmax:
    
    def __call__(self, x: FloatArray) -> FloatArray:
        x_max: FloatArray = np.max(x, axis=-1, keepdims=True)
        exps: FloatArray = np.exp(x - x_max)
        self.output: FloatArray = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output: FloatArray, learning_rate: float) -> FloatArray:
        s: FloatArray = (grad_output * self.output).sum(axis=-1, keepdims=True)
        grad_input: FloatArray = (grad_output - s) * self.output
        return grad_input