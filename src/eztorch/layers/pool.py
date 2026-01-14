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
        grad = np.repeat(grad_output[:, np.newaxis, :], self._last_seq_len, axis=1)
        grad = grad / float(self._last_seq_len)
        return grad

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []


class MaxPool2d:
    """2D max pooling layer (supports padding).

    Input shape: (batch, channels, height, width)
    Output shape depends on `kernel_size`, `stride`, and `padding`.
    Padding is applied as constant `-inf` to avoid selecting padded positions.
    """

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self._last_input_shape: tuple[int, int, int, int] | None = None
        self._max_indices: np.ndarray | None = None  # (batch, channels, h_out, w_out, 2) in padded coords

    def __call__(self, x: FloatArray) -> FloatArray:
        batch, channels, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        out = np.zeros((batch, channels, h_out, w_out), dtype=float)
        max_idx = np.zeros((batch, channels, h_out, w_out, 2), dtype=np.int64)

        # Pad with -inf so padded positions are never selected
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant", constant_values=-np.inf) if p > 0 else x

        for n in range(batch):
            for c in range(channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * s
                        w_start = j * s
                        region = x_padded[n, c, h_start:h_start + k, w_start:w_start + k]
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        out[n, c, i, j] = region[max_pos]
                        max_idx[n, c, i, j] = (h_start + max_pos[0], w_start + max_pos[1])

        self._last_input_shape = x.shape
        self._max_indices = max_idx
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._last_input_shape is None or self._max_indices is None:
            raise RuntimeError("MaxPool2d.backward called before forward")
        grad_input = np.zeros(self._last_input_shape, dtype=float)
        batch, channels, h_out, w_out = grad_output.shape
        p = self.padding
        _, _, h, w = self._last_input_shape

        for n in range(batch):
            for c in range(channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_idx, w_idx = self._max_indices[n, c, i, j]
                        # Map padded indices back to original coords
                        h_orig = h_idx - p
                        w_orig = w_idx - p
                        if 0 <= h_orig < h and 0 <= w_orig < w:
                            grad_input[n, c, h_orig, w_orig] += grad_output[n, c, i, j]
        return grad_input

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []