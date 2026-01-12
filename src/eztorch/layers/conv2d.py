import numpy as np

from eztorch.typing import FloatArray


class Conv2d:

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1, bias: bool = True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else int(kernel_size)
        self.padding = padding
        self.stride = stride
        limit = 1 / np.sqrt(in_channels * self.kernel_size * self.kernel_size)
        self.weights: FloatArray = np.random.uniform(-limit, limit, size=(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.bias: FloatArray | None = np.zeros(out_channels, dtype=float) if bias else None
        self.grad_weights: FloatArray = np.zeros_like(self.weights)
        self.grad_bias: FloatArray | None = np.zeros_like(self.bias) if self.bias is not None else None

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input = x
        batch_size, _, h, w = x.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        self.input_padded = x_padded

        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        out = np.zeros((batch_size, self.out_channels, h_out, w_out), dtype=float)

        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * s
                        w_start = j * s
                        region = x_padded[n, :, h_start:h_start + k, w_start:w_start + k]
                        out[n, c_out, i, j] = np.sum(region * self.weights[c_out])
                if self.bias is not None:
                    out[n, c_out] += self.bias[c_out]
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        batch_size, _, h_out, w_out = grad_output.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride

        self.grad_weights[...] = 0
        if self.grad_bias is not None:
            self.grad_bias[...] = 0

        grad_input_padded = np.zeros_like(self.input_padded)

        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * s
                        w_start = j * s
                        region = self.input_padded[n, :, h_start:h_start + k, w_start:w_start + k]
                        grad_val = grad_output[n, c_out, i, j]
                        self.grad_weights[c_out] += grad_val * region
                        grad_input_padded[n, :, h_start:h_start + k, w_start:w_start + k] += grad_val * self.weights[c_out]
                if self.grad_bias is not None:
                    self.grad_bias[c_out] += np.sum(grad_output[n, c_out])

        if p > 0:
            grad_input = grad_input_padded[:, :, p:-p, p:-p]
        else:
            grad_input = grad_input_padded
        return grad_input

    def parameters(self) -> list[FloatArray]:
        params = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def grads(self) -> list[FloatArray]:
        grads = [self.grad_weights]
        if self.grad_bias is not None:
            grads.append(self.grad_bias)
        return grads