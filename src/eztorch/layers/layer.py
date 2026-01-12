from typing import Protocol, List

from eztorch.typing import FloatArray


class LayerProtocol(Protocol):
    def __call__(self, x: FloatArray) -> FloatArray:
        ...

    def backward(self, grad_output: FloatArray, learning_rate: float) -> FloatArray:
        ...

    def parameters(self) -> List[FloatArray]:
        """Return trainable parameters if present; empty list otherwise."""
        ...