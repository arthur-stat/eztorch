from typing import List, Optional

import numpy as np
from eztorch.typing import FloatArray


class Linear:
    
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        self.weights: FloatArray = np.random.randn(input_features, output_features) / np.sqrt(max(1, input_features))
        self.bias: Optional[FloatArray] = np.zeros(output_features, dtype=float) if bias else None

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input: FloatArray = x
        self.output: FloatArray = x @ self.weights
        if self.bias is not None:
            self.output += self.bias
        return self.output

    def parameters(self) -> List[FloatArray]:
        params: List[FloatArray] = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def backward(self, grad_output: FloatArray, learning_rate: float) -> FloatArray:
        grad_input: FloatArray = grad_output @ self.weights.T
        grad_weights: FloatArray = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0) if self.bias is not None else None

        self.weights -= learning_rate * grad_weights
        if self.bias is not None and grad_bias is not None:
            self.bias -= learning_rate * grad_bias

        return grad_input