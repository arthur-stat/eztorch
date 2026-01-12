import numpy as np

from eztorch.typing import FloatArray


class MSELoss:
    def __call__(self, preds: FloatArray, targets: FloatArray) -> tuple[float, FloatArray]:
        diff: FloatArray = preds - targets
        batch_size: int = preds.shape[0]
        loss = float(np.mean(diff ** 2))
        grad_output: FloatArray = (2.0 / batch_size) * diff
        return loss, grad_output