from __future__ import annotations

from contextlib import contextmanager
from typing import Protocol, Iterator

from eztorch.typing import FloatArray


class Optimizer(Protocol):
    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        ...

    def zero_grad(self, grads: list[FloatArray]) -> None:
        ...


def zero_grads_inplace(grads: list[FloatArray]) -> None:
    for g in grads:
        g[...] = 0


@contextmanager
def zeroed(grads: list[FloatArray]) -> Iterator[None]:
    """Zero grads on enter. Useful if backward accumulates gradients."""
    zero_grads_inplace(grads)
    yield