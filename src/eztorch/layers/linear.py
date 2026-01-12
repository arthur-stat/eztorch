import numpy as np


class Linear:
    
    def __init__(self, input_features, output_features, bias=True):
        self.weights = np.random.randn(input_features, output_features) / np.sqrt(max(1, input_features))
        self.bias = np.zeros(output_features, dtype=float) if bias else None

    def __call__(self, x):
        self.input = x
        self.output = x @ self.weights
        if self.bias is not None:
            self.output += self.bias
        return self.output

    def parameters(self):
        params = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output @ self.weights.T
        grad_weights = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0) if self.bias is not None else None

        self.weights -= learning_rate * grad_weights
        if self.bias is not None:
            self.bias -= learning_rate * grad_bias

        return grad_input