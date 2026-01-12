from typing import Protocol

from eztorch.typing import FloatArray


class LayerProtocol(Protocol):

    def __call__(self, x: FloatArray) -> FloatArray:
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        ...

    def parameters(self) -> list[FloatArray]:
        """Return trainable parameters if present; empty list otherwise."""
        ...

    def grads(self) -> list[FloatArray]:
        """Return gradients aligned with parameters; empty list if no params."""
        ...