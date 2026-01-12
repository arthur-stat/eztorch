from typing import Protocol

from eztorch.typing import FloatArray, IntArray


class LossFunction(Protocol):
    def __call__(self, preds: FloatArray, targets: FloatArray | IntArray) -> tuple[float, FloatArray]:
        ...


from .cross_entropy import CrossEntropyLoss
from .mse import MSELoss

__all__ = ["LossFunction", "CrossEntropyLoss", "MSELoss"]