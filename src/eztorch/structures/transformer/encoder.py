from eztorch.typing import FloatArray

from eztorch.layers.attention import MultiHeadSelfAttention
from eztorch.layers.linear import Linear
from eztorch.layers.norm import LayerNorm
from eztorch.functions.activations import ReLU


class TransformerEncoderLayer:
    """Single Transformer encoder layer: Self-Attn -> FFN, each with residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = self._as_sequential([
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model),
        ])
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x: FloatArray, mask: FloatArray | None = None) -> FloatArray:
        self._input: FloatArray = x
        attn_out: FloatArray = self.self_attn(x, mask=mask)
        self._attn_out: FloatArray = attn_out
        pre_norm1 = attn_out + x
        norm1_out: FloatArray = self.norm1(pre_norm1)
        self._norm1_out = norm1_out

        ffn_out: FloatArray = self.ffn(norm1_out)
        self._ffn_out = ffn_out
        pre_norm2 = ffn_out + norm1_out
        out: FloatArray = self.norm2(pre_norm2)
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        grad_pre_norm2 = self.norm2.backward(grad_output)

        grad_ffn_out = grad_pre_norm2
        grad_skip_norm1 = grad_pre_norm2

        grad_ffn_in = self.ffn.backward(grad_ffn_out)
        grad_norm1_total = grad_ffn_in + grad_skip_norm1

        grad_pre_norm1 = self.norm1.backward(grad_norm1_total)
        grad_attn_out = grad_pre_norm1
        grad_input_skip = grad_pre_norm1

        grad_input = self.self_attn.backward(grad_attn_out)
        grad_input += grad_input_skip
        return grad_input

    def parameters(self) -> list[FloatArray]:
        params = []
        params.extend(self.self_attn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm2.parameters())
        return params

    def grads(self) -> list[FloatArray]:
        grads = []
        grads.extend(self.self_attn.grads())
        grads.extend(self.norm1.grads())
        grads.extend(self.ffn.grads())
        grads.extend(self.norm2.grads())
        return grads

    def _as_sequential(self, layers: list) -> "SequentialCompat":
        return SequentialCompat(layers)

    def _forward_attn_with_mask(self, x: FloatArray, mask: FloatArray) -> FloatArray:
        return self.self_attn(x, mask)


class SequentialCompat:
    """Minimal sequential wrapper to reuse residual/backward utilities."""

    def __init__(self, layers: list) -> None:
        self.layers = layers

    def __call__(self, x: FloatArray) -> FloatArray:
        for layer in self.layers:
            x = layer(x)
        self.output: FloatArray = x
        return self.output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        g = grad_output
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def parameters(self) -> list[FloatArray]:
        return [p for l in self.layers for p in getattr(l, "parameters", lambda: [])()]

    def grads(self) -> list[FloatArray]:
        return [g for l in self.layers for g in getattr(l, "grads", lambda: [])()]