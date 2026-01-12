import numpy as np

from eztorch.typing import FloatArray
from .layer import LayerProtocol


class Sequential:
    
    def __init__(self, layers: list[LayerProtocol]):
        self.layers: list[LayerProtocol] = layers

    def __call__(self, x: FloatArray) -> FloatArray:
        for layer in self.layers:
            x = layer(x)
        self.output: FloatArray = x
        return self.output

    def predict_proba(self, x: FloatArray) -> FloatArray:
        logits: FloatArray = self(x)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        e_x = np.exp(logits - logits_max)
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def parameters(self) -> list[FloatArray]:
        return [p for layer in self.layers for p in getattr(layer, "parameters", lambda: [])()]

    def grads(self) -> list[FloatArray]:
        return [g for layer in self.layers for g in getattr(layer, "grads", lambda: [])()]