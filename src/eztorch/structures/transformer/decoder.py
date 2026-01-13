import numpy as np

from eztorch.functions.activations import ReLU
from eztorch.layers.attention import MultiHeadSelfAttention
from eztorch.layers.linear import Linear
from eztorch.layers.norm import LayerNorm
from eztorch.layers.dropout import Dropout
from eztorch.typing import FloatArray


class TransformerDecoderLayer:
    """Single Transformer decoder layer: self-attn, cross-attn, FFN; each with residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.dropout1 = Dropout(p=0.1)
        self.norm1 = LayerNorm(d_model)
        self.dropout2 = Dropout(p=0.1)
        self.norm2 = LayerNorm(d_model)
        self.dropout3 = Dropout(p=0.1)
        self.norm3 = LayerNorm(d_model)
        self.ffn = self._as_sequential([
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model),
        ])

    def __call__(self, x: FloatArray, encoder_out: FloatArray, tgt_mask: FloatArray | None = None, memory_mask: FloatArray | None = None) -> FloatArray:
        self._input: FloatArray = x
        self._memory: FloatArray = encoder_out

        self_attn_out: FloatArray = self.self_attn(x, mask=tgt_mask)
        self_attn_out = self.dropout1(self_attn_out)
        pre_norm1 = self_attn_out + x
        norm1_out: FloatArray = self.norm1(pre_norm1)
        self._norm1_out = norm1_out

        cross_out: FloatArray = self.cross_attn(norm1_out, encoder_out, memory_mask)
        cross_out = self.dropout2(cross_out)
        pre_norm2 = cross_out + norm1_out
        norm2_out: FloatArray = self.norm2(pre_norm2)
        self._norm2_out = norm2_out

        ffn_out: FloatArray = self.ffn(norm2_out)
        ffn_out = self.dropout3(ffn_out)
        pre_norm3 = ffn_out + norm2_out
        out: FloatArray = self.norm3(pre_norm3)
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        grad_pre_norm3 = self.norm3.backward(grad_output)

        grad_ffn_out = self.dropout3.backward(grad_pre_norm3)
        grad_norm2_skip = grad_pre_norm3

        grad_ffn_in = self.ffn.backward(grad_ffn_out)
        grad_norm2_total = grad_ffn_in + grad_norm2_skip

        grad_pre_norm2 = self.norm2.backward(grad_norm2_total)
        grad_cross_out = self.dropout2.backward(grad_pre_norm2)
        grad_norm1_skip = grad_pre_norm2

        grad_cross_query = self.cross_attn.backward(grad_cross_out)
        grad_norm1_total = grad_cross_query + grad_norm1_skip

        grad_pre_norm1 = self.norm1.backward(grad_norm1_total)
        grad_self_out = self.dropout1.backward(grad_pre_norm1)
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
        self.grad_W_o[...] = np.einsum("bsi,bsj->ij", self.context_combined, grad_output)
        grad_context_combined: FloatArray = np.einsum("bsi,ij->bsj", grad_output, self.W_o.T)

        grad_context_heads = grad_context_combined.reshape(
            grad_output.shape[0], grad_output.shape[1], self.num_heads, self.d_k
        )
        grad_context_heads = np.transpose(grad_context_heads, (0, 2, 1, 3))

        grad_Q_h = np.zeros_like(self.Q_h)
        grad_K_h = np.zeros_like(self.K_h)
        grad_V_h = np.zeros_like(self.V_h)

        for h in range(self.num_heads):
            grad_attn_output = grad_context_heads[:, h]
            attn = self.attn_weights[h]
            V_h = self.V_h[:, h]
            K_h = self.K_h[:, h]
            Q_h = self.Q_h[:, h]

            grad_V_h[:, h, :, :] = np.einsum("bqk,bqd->bkd", attn, grad_attn_output)
            grad_context = np.einsum("bqd,bkd->bqk", grad_attn_output, V_h)
            grad_scores = attn * (grad_context - np.sum(grad_context * attn, axis=-1, keepdims=True))
            grad_Q_h[:, h, :, :] = np.einsum("bqk,bkd->bqd", grad_scores, K_h) / np.sqrt(self.d_k)
            grad_K_h[:, h, :, :] = np.einsum("bqk,bqd->bkd", grad_scores, Q_h) / np.sqrt(self.d_k)

        grad_Q = self._combine_heads(grad_Q_h)
        grad_K = self._combine_heads(grad_K_h)
        grad_V = self._combine_heads(grad_V_h)

        self.grad_W_q[...] = np.einsum("bsi,bsj->ij", self.input_q, grad_Q)
        self.grad_W_k[...] = np.einsum("bsi,bsj->ij", self.input_k, grad_K)
        self.grad_W_v[...] = np.einsum("bsi,bsj->ij", self.input_v, grad_V)

        grad_input_q = np.einsum("bsi,ij->bsj", grad_Q, self.W_q.T)
        grad_input_k = np.einsum("bsi,ij->bsj", grad_K, self.W_k.T)
        grad_input_v = np.einsum("bsi,ij->bsj", grad_V, self.W_v.T)

        grad_memory = grad_input_k + grad_input_v
        self.grad_memory: FloatArray = grad_memory

        return grad_input_q


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