import numpy as np

from eztorch.typing import FloatArray


class UnsqueezeSeq:
    """Expand 2D input (batch, features) to 3D (batch, 1, features).

    Paired with GlobalAvgPool1d, this acts like an identity.
    """

    def __init__(self) -> None:
        self._seq_len = 1

    def __call__(self, x: FloatArray) -> FloatArray:
        self._last_shape = x.shape
        return np.expand_dims(x, axis=1)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        # grad_output is expected to be (batch, 1, features)
        return np.squeeze(grad_output, axis=1)

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []