import numpy as np
from eztorch.typing import FloatArray, IntArray

from eztorch.layers.linear import Linear
from eztorch.layers.sequential import Sequential
from eztorch.functions.sigmoid import Sigmoid
from eztorch.functions.softmax import Softmax
from eztorch.optim.sgd import SGD
from eztorch.optim.base import Optimizer


class MLP:
    
    def __init__(self, seq: Sequential) -> None:
        self.model = seq
        self.softmax = Softmax()

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def forward(self, x: FloatArray) -> FloatArray:
        return self.model(x)

    def backwardAndGradientDescent(self, x: FloatArray, y: IntArray, learning_rate: float) -> None:
        self.backward(x, y)
        optimizer = SGD(learning_rate)
        optimizer.step(self.model.parameters(), self.model.grads())

    def probability(self, x: FloatArray) -> FloatArray:
        logits: FloatArray = self.forward(x)
        return self.softmax(logits)

    def classify(self, x: FloatArray) -> IntArray:
        return np.argmax(self.probability(x), axis=1)

    def backward(self, x: FloatArray, y: IntArray) -> None:
        batch_size: int = x.shape[0]
        logits: FloatArray = self.forward(x)
        probs: FloatArray = self.softmax(logits)

        grad_output: FloatArray = probs.copy()
        grad_output[np.arange(batch_size), y] -= 1
        grad_output /= batch_size

        for layer in reversed(self.model.layers):
            grad_output = getattr(layer, "backward")(grad_output, 0.0)

    def train_step(self, x: FloatArray, y: IntArray, optimizer: Optimizer) -> None:
        self.backward(x, y)
        optimizer.step(self.model.parameters(), self.model.grads())