import numpy as np
from eztorch.typing import FloatArray
from eztorch.functions.attention import scaled_dot_product_attention


class MultiHeadSelfAttention:
    """Multi-head self-attention layer (no bias, no dropout) for transformer-style blocks."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        limit = 1 / np.sqrt(d_model)
        self.W_q: FloatArray = np.random.uniform(-limit, limit, size=(d_model, d_model))
        self.W_k: FloatArray = np.random.uniform(-limit, limit, size=(d_model, d_model))
        self.W_v: FloatArray = np.random.uniform(-limit, limit, size=(d_model, d_model))
        self.W_o: FloatArray = np.random.uniform(-limit, limit, size=(d_model, d_model))

        self.grad_W_q: FloatArray = np.zeros_like(self.W_q)
        self.grad_W_k: FloatArray = np.zeros_like(self.W_k)
        self.grad_W_v: FloatArray = np.zeros_like(self.W_v)
        self.grad_W_o: FloatArray = np.zeros_like(self.W_o)

    def _split_heads(self, x: FloatArray) -> FloatArray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))  # (batch, heads, seq, d_k)

    def _combine_heads(self, x: FloatArray) -> FloatArray:
        batch_size, heads, seq_len, d_k = x.shape
        x = np.transpose(x, (0, 2, 1, 3)).reshape(batch_size, seq_len, heads * d_k)
        return x

    def __call__(self, x: FloatArray, mask: FloatArray | None = None) -> FloatArray:
        self.input: FloatArray = x  # (batch, seq, d_model)

        Q = np.einsum("bsd,dk->bsk", x, self.W_q)
        K = np.einsum("bsd,dk->bsk", x, self.W_k)
        V = np.einsum("bsd,dk->bsk", x, self.W_v)

        Q_h = self._split_heads(Q)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # reshape mask to match heads if provided
        attn_outputs = []
        attn_weights = []
        for h in range(self.num_heads):
            out, attn = scaled_dot_product_attention(Q_h[:, h], K_h[:, h], V_h[:, h], mask)
            attn_outputs.append(out)
            attn_weights.append(attn)

        self.attn_weights: list[FloatArray] = attn_weights
        self.Q_h = Q_h
        self.K_h = K_h
        self.V_h = V_h

        context = np.stack(attn_outputs, axis=1)  # (batch, heads, seq, d_k)
        context_combined = self._combine_heads(context)  # (batch, seq, d_model)

        self.context = context
        self.context_combined = context_combined
        self.Q = Q
        self.K = K
        self.V = V

        output: FloatArray = np.einsum("bsd,dk->bsk", context_combined, self.W_o)
        return output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        self.grad_W_o[...] = np.einsum("bsi,bsj->ij", self.context_combined, grad_output)
        grad_context_combined: FloatArray = np.einsum("bsi,ij->bsj", grad_output, self.W_o.T)

        grad_context_heads = grad_context_combined.reshape(
            grad_output.shape[0], grad_output.shape[1], self.num_heads, self.d_k
        )
        grad_context_heads = np.transpose(grad_context_heads, (0, 2, 1, 3))  # (batch, heads, seq, d_k)

        grad_Q_h = np.zeros_like(self.Q_h)
        grad_K_h = np.zeros_like(self.K_h)
        grad_V_h = np.zeros_like(self.V_h)

        for h in range(self.num_heads):
            grad_attn_output = grad_context_heads[:, h]  # (batch, seq, d_k)
            attn = self.attn_weights[h]  # (batch, seq, seq)
            V_h = self.V_h[:, h]
            K_h = self.K_h[:, h]
            Q_h = self.Q_h[:, h]

            # grad w.r.t. value
            grad_V_h[:, h, :, :] = np.einsum("bqk,bqd->bkd", attn, grad_attn_output)
            grad_context = np.einsum("bqd,bkd->bqk", grad_attn_output, V_h)

            # softmax gradient
            grad_scores = attn * (grad_context - np.sum(grad_context * attn, axis=-1, keepdims=True))

            # grad scores -> Q,K
            grad_Q_h[:, h, :, :] = np.einsum("bqk,bkd->bqd", grad_scores, K_h) / np.sqrt(self.d_k)
            grad_K_h[:, h, :, :] = np.einsum("bqk,bqd->bkd", grad_scores, Q_h) / np.sqrt(self.d_k)

        grad_Q = self._combine_heads(grad_Q_h)
        grad_K = self._combine_heads(grad_K_h)
        grad_V = self._combine_heads(grad_V_h)

        self.grad_W_q[...] = np.einsum("bsi,bsj->ij", self.input, grad_Q)
        self.grad_W_k[...] = np.einsum("bsi,bsj->ij", self.input, grad_K)
        self.grad_W_v[...] = np.einsum("bsi,bsj->ij", self.input, grad_V)

        grad_input = np.zeros_like(self.input)
        grad_input += np.einsum("bsi,ij->bsj", grad_Q, self.W_q.T)
        grad_input += np.einsum("bsi,ij->bsj", grad_K, self.W_k.T)
        grad_input += np.einsum("bsi,ij->bsj", grad_V, self.W_v.T)

        return grad_input

    def parameters(self) -> list[FloatArray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def grads(self) -> list[FloatArray]:
        return [self.grad_W_q, self.grad_W_k, self.grad_W_v, self.grad_W_o]