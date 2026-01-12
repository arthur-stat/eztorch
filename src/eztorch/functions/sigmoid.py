import numpy as np


class Sigmoid:
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

