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
        logits_max = np.max(logits, axis=-1, keepdims=True)
        e_x = np.exp(logits - logits_max)
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def parameters(self):
        return [p for layer in self.layers for p in getattr(layer, "parameters", lambda: [])()]