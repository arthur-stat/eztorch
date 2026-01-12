import numpy as np
from eztorch.typing import FloatArray


class BatchNorm1d:

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        self.num_features = num_features
        self.eps = eps
        self.gamma: FloatArray = np.ones(num_features, dtype=float)
        self.beta: FloatArray = np.zeros(num_features, dtype=float)
        self.grad_gamma: FloatArray = np.zeros_like(self.gamma)
        self.grad_beta: FloatArray = np.zeros_like(self.beta)

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input: FloatArray = x
        self.mean: FloatArray = np.mean(x, axis=0)
        self.var: FloatArray = np.var(x, axis=0)
        self.std_inv: FloatArray = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat: FloatArray = (x - self.mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output: FloatArray) -> FloatArray:
        batch_size = grad_output.shape[0]

        self.grad_gamma = np.sum(grad_output * self.x_hat, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        grad_x_hat = grad_output * self.gamma
        grad_var = np.sum(grad_x_hat * (self.input - self.mean) * -0.5 * self.std_inv**3, axis=0)
        grad_mean = np.sum(grad_x_hat * -self.std_inv, axis=0) + grad_var * np.mean(-2.0 * (self.input - self.mean), axis=0)

        grad_input = grad_x_hat * self.std_inv + grad_var * 2.0 * (self.input - self.mean) / batch_size + grad_mean / batch_size
        return grad_input

    def parameters(self) -> list[FloatArray]:
        return [self.gamma, self.beta]

    def grads(self) -> list[FloatArray]:
        return [self.grad_gamma, self.grad_beta]


class LayerNorm:

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        self.num_features = num_features
        self.eps = eps
        self.gamma: FloatArray = np.ones(num_features, dtype=float)
        self.beta: FloatArray = np.zeros(num_features, dtype=float)
        self.grad_gamma: FloatArray = np.zeros_like(self.gamma)
        self.grad_beta: FloatArray = np.zeros_like(self.beta)

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input: FloatArray = x
        self.mean: FloatArray = np.mean(x, axis=1, keepdims=True)
        self.var: FloatArray = np.var(x, axis=1, keepdims=True)
        self.std_inv: FloatArray = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat: FloatArray = (x - self.mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_output: FloatArray) -> FloatArray:
        self.grad_gamma = np.sum(grad_output * self.x_hat, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        grad_x_hat = grad_output * self.gamma
        mean_grad = np.mean(grad_x_hat, axis=1, keepdims=True)
        mean_grad_xhat = np.mean(grad_x_hat * self.x_hat, axis=1, keepdims=True)

        grad_input = (grad_x_hat - mean_grad - self.x_hat * mean_grad_xhat) * self.std_inv
        return grad_input

    def parameters(self) -> list[FloatArray]:
        return [self.gamma, self.beta]

    def grads(self) -> list[FloatArray]:
        return [self.grad_gamma, self.grad_beta]