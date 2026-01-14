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
    zero_grads_inplace(grads)
    yield


@contextmanager
def frozen_params(params: list[FloatArray]) -> Iterator[None]:
    """Temporarily freeze parameters by making them read-only.

    Optimizers are expected to skip read-only arrays.
    """

    for p in params:
        # Set writeable False
        try:
            p.flags.writeable = False
        except Exception:
            pass
    try:
        yield
    finally:
        # Restore writeable True
        for p in params:
            try:
                p.flags.writeable = True
            except Exception:
                pass