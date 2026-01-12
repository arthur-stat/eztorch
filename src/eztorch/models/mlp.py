from typing import List

import numpy as np
from eztorch.typing import FloatArray, IntArray

from eztorch.layers.linear import Linear
from eztorch.layers.sequential import Sequential
from eztorch.functions.sigmoid import Sigmoid
from eztorch.functions.softmax import Softmax


class MLP:
    
    def __init__(self) -> None:
        self.model = Sequential([
            Linear(2, 4), Sigmoid(),
            Linear(4, 4), Sigmoid(),
            Linear(4, 2)
        ])
        self.softmax = Softmax()

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def forward(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def backwardAndGradientDescent(self, x: FloatArray, y: IntArray, learning_rate: float) -> None:
        batch_size: int = x.shape[0]
        logits: FloatArray = self.forward(x)
        probs: FloatArray = self.softmax(logits)

        grad_output: FloatArray = probs.copy()
        grad_output[np.arange(batch_size), y] -= 1
        grad_output /= batch_size

        for layer in reversed(self.model.layers):
            grad_output = getattr(layer, "backward")(grad_output, learning_rate)

    def probability(self, x: FloatArray) -> FloatArray:
        logits: FloatArray = self.forward(x)
        return self.softmax(logits)

    def classify(self, x: FloatArray) -> IntArray:
        return np.argmax(self.probability(x), axis=1)
