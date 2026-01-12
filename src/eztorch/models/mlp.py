import numpy as np

from eztorch.layers.linear import Linear
from eztorch.layers.sequential import Sequential
from eztorch.functions.sigmoid import Sigmoid
from eztorch.functions.softmax import Softmax


class MLP:
    
    def __init__(self):
        self.model = Sequential([
            Linear(2, 4), Sigmoid(),
            Linear(4, 4), Sigmoid(),
            Linear(4, 2)
        ])
        self.softmax = Softmax()

    def __call__(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)

    def backwardAndGradientDescent(self, x, y, learning_rate):
        batch_size = x.shape[0]
        logits = self.forward(x)
        probs = self.softmax(logits)

        grad_output = probs
        grad_output[range(batch_size), y] -= 1
        grad_output /= batch_size

        for layer in reversed(self.model.layers):
            grad_output = getattr(layer, "backward")(grad_output, learning_rate)

    def probability(self, x):
        logits = self.forward(x)
        return self.softmax(logits)

    def classify(self, x):
        return np.argmax(self.probability(x), axis=1)