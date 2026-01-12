from typing import Optional

from eztorch.typing import FloatArray
from .layer import LayerProtocol


class Residual:

    def __init__(self, block: LayerProtocol, projection: Optional[LayerProtocol] = None) -> None:
        self.block = block
        self.projection = projection

    def __call__(self, x: FloatArray) -> FloatArray:
        self.input: FloatArray = x
        self.skip: FloatArray = self.projection(x) if self.projection is not None else x
        self.out: FloatArray = self.block(x)
        self.output: FloatArray = self.skip + self.out
        return self.output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        grad_block: FloatArray = self._backward_layer(self.block, grad_output)
        grad_skip: FloatArray = self._backward_layer(self.projection, grad_output) if self.projection is not None else grad_output
        return grad_block + grad_skip

    def parameters(self) -> list[FloatArray]:
        params = []
        params.extend(getattr(self.block, "parameters", lambda: [])())
        if self.projection is not None:
            params.extend(getattr(self.projection, "parameters", lambda: [])())
        return params

    def grads(self) -> list[FloatArray]:
        grads = []
        grads.extend(getattr(self.block, "grads", lambda: [])())
        if self.projection is not None:
            grads.extend(getattr(self.projection, "grads", lambda: [])())
        return grads

    def _backward_layer(self, layer: Optional[LayerProtocol], grad_output: FloatArray) -> FloatArray:
        if layer is None:
            return grad_output
        backward = getattr(layer, "backward", None)
        if callable(backward):
            return backward(grad_output)
        layers = getattr(layer, "layers", None)
        if layers is not None:
            g = grad_output
            for l in reversed(layers):
                g = getattr(l, "backward")(g)
            return g
        raise AttributeError("Layer does not implement backward")