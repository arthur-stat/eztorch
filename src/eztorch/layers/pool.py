import numpy as np

from eztorch.typing import FloatArray


class GlobalAvgPool1d:
    """Global average pooling over the sequence dimension.

    Input: (batch, seq, features) -> Output: (batch, features)
    """

    def __init__(self) -> None:
        self._last_seq_len: int | None = None

    def __call__(self, x: FloatArray) -> FloatArray:
        self._last_seq_len = x.shape[1]
        return np.mean(x, axis=1)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._last_seq_len is None:
            raise RuntimeError("GlobalAvgPool1d.backward called before forward")
        # Distribute gradient evenly across seq positions
        batch, feat = grad_output.shape
        grad = np.repeat(grad_output[:, np.newaxis, :], self._last_seq_len, axis=1)
        grad = grad / float(self._last_seq_len)
        return grad

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []