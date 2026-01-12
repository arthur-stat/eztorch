import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.output = x
        return self.output

    def predict_proba(self, x):
        logits = self(x)
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0, keepdims=True)

    def paramenters(self):
        return [p for layer in self.layers for p in layer.paramenters()]

