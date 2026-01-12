import numpy as np
from eztorch.typing import FloatArray

from eztorch.layers.attention import MultiHeadSelfAttention
from eztorch.functions.attention import scaled_dot_product_attention
from eztorch.layers.linear import Linear
from eztorch.layers.norm import LayerNorm
from eztorch.functions.activations import ReLU


class TransformerDecoderLayer:
    """Single Transformer decoder layer: self-attn, cross-attn, FFN; each with residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.ffn = self._as_sequential([
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model),
        ])

    def __call__(self, x: FloatArray, encoder_out: FloatArray, tgt_mask: FloatArray | None = None, memory_mask: FloatArray | None = None) -> FloatArray:
        self._input: FloatArray = x
        self._memory: FloatArray = encoder_out

        self_attn_out: FloatArray = self.self_attn(x, tgt_mask)
        pre_norm1 = self_attn_out + x
        norm1_out: FloatArray = self.norm1(pre_norm1)
        self._norm1_out = norm1_out

        cross_out: FloatArray = self.cross_attn(norm1_out, encoder_out, memory_mask)
        pre_norm2 = cross_out + norm1_out
        norm2_out: FloatArray = self.norm2(pre_norm2)
        self._norm2_out = norm2_out

        ffn_out: FloatArray = self.ffn(norm2_out)
        pre_norm3 = ffn_out + norm2_out
        out: FloatArray = self.norm3(pre_norm3)
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        grad_pre_norm3 = self.norm3.backward(grad_output)

        grad_ffn_out = grad_pre_norm3
        grad_norm2_skip = grad_pre_norm3

        grad_ffn_in = self.ffn.backward(grad_ffn_out)
        grad_norm2_total = grad_ffn_in + grad_norm2_skip

        grad_pre_norm2 = self.norm2.backward(grad_norm2_total)
        grad_cross_out = grad_pre_norm2
        grad_norm1_skip = grad_pre_norm2

        grad_cross_query = self.cross_attn.backward(grad_cross_out)
        grad_norm1_total = grad_cross_query + grad_norm1_skip

        grad_pre_norm1 = self.norm1.backward(grad_norm1_total)
        grad_self_out = grad_pre_norm1
        grad_input_skip = grad_pre_norm1

        grad_input = self.self_attn.backward(grad_self_out)
        grad_input += grad_input_skip
        return grad_input

    def parameters(self) -> list[FloatArray]:
        params = []
        params.extend(self.self_attn.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.cross_attn.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.norm3.parameters())
        return params

    def grads(self) -> list[FloatArray]:
        grads = []
        grads.extend(self.self_attn.grads())
        grads.extend(self.norm1.grads())
        grads.extend(self.cross_attn.grads())
        grads.extend(self.norm2.grads())
        grads.extend(self.ffn.grads())
        grads.extend(self.norm3.grads())
        return grads

    def _as_sequential(self, layers: list) -> "SequentialCompat":
        return SequentialCompat(layers)


class MultiHeadCrossAttention(MultiHeadSelfAttention):
    def __call__(self, query: FloatArray, memory: FloatArray, mask: FloatArray | None = None) -> FloatArray:
        # key/value come from memory; reuse projection weights
        self.memory: FloatArray = memory
        return super().__call__(query, memory, memory, mask)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        # use base backward (computes grads for q/k/v); drop memory grads for now
        grad_query = super().backward(grad_output)
        return grad_query


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