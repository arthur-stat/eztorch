import numpy as np


class Softmax:
    def __call__(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exps = np.exp(x - x_max)
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

