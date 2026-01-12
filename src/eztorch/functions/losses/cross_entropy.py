import numpy as np

from eztorch.typing import FloatArray, IntArray


class CrossEntropyLoss:
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon

    def __call__(self, logits: FloatArray, targets: IntArray) -> tuple[float, FloatArray]:
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(logits - logits_max)
        probs: FloatArray = exps / np.sum(exps, axis=-1, keepdims=True)

        batch_size: int = logits.shape[0]
        loss = float(-np.mean(np.log(probs[np.arange(batch_size), targets] + self.epsilon)))

        grad_output: FloatArray = probs.copy()
        grad_output[np.arange(batch_size), targets] -= 1
        grad_output /= batch_size
        return loss, grad_output