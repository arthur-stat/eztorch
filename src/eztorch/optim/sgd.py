from typing import List

from eztorch.typing import FloatArray
from .base import zero_grads_inplace


class SGD:
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, params: List[FloatArray], grads: List[FloatArray]) -> None:
        for p, g in zip(params, grads):
            # Skip frozen (read-only) parameters
            if hasattr(p, "flags") and getattr(p.flags, "writeable", True) is False:
                continue
            p -= self.lr * g

    def zero_grad(self, grads: List[FloatArray]) -> None:
        zero_grads_inplace(grads)