from typing import Optional

import numpy as np

from eztorch.typing import FloatArray


class Linear:
    
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        self.weights: FloatArray = np.random.randn(input_features, output_features) / np.sqrt(max(1, input_features))
        self.bias: Optional[FloatArray] = np.zeros(output_features, dtype=float) if bias else None
        self.grad_weights: FloatArray = np.zeros_like(self.weights)
        self.grad_bias: Optional[FloatArray] = np.zeros_like(self.bias) if self.bias is not None else None

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input: FloatArray = x
        self.output: FloatArray = x @ self.weights
        if self.bias is not None:
            self.output += self.bias
        return self.output

    def parameters(self) -> list[FloatArray]:
        params: list[FloatArray] = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def backward(self, grad_output: FloatArray) -> FloatArray:
        # Flatten leading dims
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        input_flat = self.input.reshape(-1, self.input.shape[-1])

        grad_input: FloatArray = grad_output @ self.weights.T
        self.grad_weights = input_flat.T @ grad_output_flat
        self.grad_bias = np.sum(grad_output_flat, axis=0) if self.bias is not None else None
        return grad_input

    def grads(self) -> list[FloatArray]:
        grads: list[FloatArray] = [self.grad_weights]
        if self.grad_bias is not None:
            grads.append(self.grad_bias)
        return grads