from __future__ import annotations

from typing import List, Tuple

import numpy as np

from eztorch.typing import FloatArray
from .base import zero_grads_inplace


class Adam:

    def __init__(
        self,
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: List[FloatArray] = []
        self.v: List[FloatArray] = []
        self.t: int = 0

    def _ensure_state(self, params: List[FloatArray]) -> None:
        if len(self.m) != len(params) or any(m.shape != p.shape for m, p in zip(self.m, params)):
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.t = 0

    def step(self, params: List[FloatArray], grads: List[FloatArray]) -> None:
        self._ensure_state(params)
        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, grads: List[FloatArray]) -> None:
        zero_grads_inplace(grads)