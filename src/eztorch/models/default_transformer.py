import numpy as np

from eztorch.functions.losses import CrossEntropyLoss
from eztorch.layers.dropout import Dropout
from eztorch.layers.embedding import Embedding
from eztorch.layers.linear import Linear
from eztorch.layers.norm import LayerNorm
from eztorch.structures.transformer.decoder import TransformerDecoderLayer
from eztorch.structures.transformer.encoder import TransformerEncoderLayer
from eztorch.typing import FloatArray, IntArray
from .base import BaseModel


class PositionalEncoding:

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe: FloatArray = pe

    def __call__(self, x: FloatArray) -> FloatArray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class DefaultTransformer(BaseModel):
    """A minimal encoder-decoder Transformer returning logits."""

    def __init__(
            self,
            src_vocab: int,
            tgt_vocab: int,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            num_layers: int = 6,
            max_len: int = 512,
    ) -> None:
        # Use embeddings for token ids directly
        self.d_model = d_model
        self.src_embed = Embedding(src_vocab, d_model)
        self.tgt_embed = Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout_src = Dropout(p=0.1)
        self.dropout_tgt = Dropout(p=0.1)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm_out = LayerNorm(d_model)
        self.generator = Linear(d_model, tgt_vocab)
        self.loss_fn = CrossEntropyLoss()

    def encode(self, src: IntArray, src_mask: FloatArray | None = None) -> FloatArray:
        x = self.src_embed(src) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout_src(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        self._memory = x
        return x

    def decode(self, tgt: IntArray, memory: FloatArray, tgt_mask: FloatArray | None = None,
               memory_mask: FloatArray | None = None) -> FloatArray:
        x = self.tgt_embed(tgt) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout_tgt(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        x = self.norm_out(x)
        logits = self.generator(x)
        return logits

    def __call__(self, src: IntArray, tgt: IntArray, src_mask: FloatArray | None = None,
                 tgt_mask: FloatArray | None = None, memory_mask: FloatArray | None = None) -> FloatArray:
        self._src = src
        self._tgt = tgt
        memory = self.encode(src, src_mask)
        logits = self.decode(tgt, memory, tgt_mask, memory_mask)
        self._logits = logits
        return logits

    def step(self, src: IntArray, tgt_input: IntArray, tgt_labels: IntArray, src_mask: FloatArray | None = None,
             tgt_mask: FloatArray | None = None, memory_mask: FloatArray | None = None) -> tuple[float, FloatArray]:
        logits = self(src, tgt_input, src_mask, tgt_mask, memory_mask)
        # flatten batch*seq for CE
        batch, seq, vocab = logits.shape
        loss, grad = self.loss_fn(logits.reshape(-1, vocab), tgt_labels.reshape(-1))
        grad = grad.reshape(batch, seq, vocab)
        return loss, grad

    def parameters(self) -> list[FloatArray]:
        params: list[FloatArray] = []
        params.extend(self.src_embed.parameters())
        params.extend(self.tgt_embed.parameters())
        for layer in self.encoder_layers:
            params.extend(layer.parameters())
        for layer in self.decoder_layers:
            params.extend(layer.parameters())
        params.extend(self.norm_out.parameters())
        params.extend(self.generator.parameters())
        return params

    def grads(self) -> list[FloatArray]:
        grads: list[FloatArray] = []
        grads.extend(self.src_embed.grads())
        grads.extend(self.tgt_embed.grads())
        for layer in self.encoder_layers:
            grads.extend(layer.grads())
        for layer in self.decoder_layers:
            grads.extend(layer.grads())
        grads.extend(self.norm_out.grads())
        grads.extend(self.generator.grads())
        return grads

    def backward(self, grad_output: FloatArray, src: IntArray, tgt: IntArray, src_mask: FloatArray | None = None,
                 tgt_mask: FloatArray | None = None, memory_mask: FloatArray | None = None) -> None:
        # backprop through generator and norm_out
        grad_gen_in: FloatArray = grad_output
        grad_norm = self.generator.backward(grad_gen_in)
        grad_dec_out = self.norm_out.backward(grad_norm)

        # decode backward
        grad_memory_total = np.zeros_like(self._memory)
        for layer in reversed(self.decoder_layers):
            grad_dec_out = layer.backward(grad_dec_out)
            grad_mem = getattr(layer.cross_attn, "grad_memory", None)
            if grad_mem is not None:
                grad_memory_total += grad_mem
        grad_dec_in = self.dropout_tgt.backward(grad_dec_out)
        _ = self.tgt_embed.backward(grad_dec_in)
        # positional encoding?

        grad_enc = grad_memory_total
        for layer in reversed(self.encoder_layers):
            grad_enc = layer.backward(grad_enc)
        grad_enc_in = self.dropout_src.backward(grad_enc)
        _ = self.src_embed.backward(grad_enc_in)
        # positional encoding?
        return