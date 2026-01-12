from eztorch.typing import FloatArray


class ReLU:

    def __call__(self, x: FloatArray) -> FloatArray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_output: FloatArray) -> FloatArray:
        return grad_output * self.mask

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []