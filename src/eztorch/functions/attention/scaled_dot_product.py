import numpy as np
from eztorch.typing import FloatArray


def scaled_dot_product_attention(
        query: FloatArray,
        key: FloatArray,
        value: FloatArray,
        mask: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute scaled dot-product attention as described in "Attention Is All You Need".

    The function computes attention scores between queries and keys, applies optional
    masking, and returns the weighted sum of values.

    Args:
        query: Query tensor with shape (batch, seq_q, d_k)
        key: Key tensor with shape (batch, seq_k, d_k)
        value: Value tensor with shape (batch, seq_k, d_v)
        mask: Optional mask tensor broadcastable to (batch, seq_q, seq_k).
              Positions with zero values will be masked (set to -1e9 before softmax).

    Returns:
        tuple[FloatArray, FloatArray]: A tuple of:
            - output: Attended values with shape (batch, seq_q, d_v)
            - attn: Attention weights with shape (batch, seq_q, seq_k)

    Raises:
        ValueError: If input dimensions are incompatible

    Notes:
        - Implements the scaling factor 1/sqrt(d_k) as in the original Transformer paper
        - Uses numerical masking with -1e9 for stability
        - Applies softmax with max subtraction for numerical stability

    Example:
        >>> batch, seq_q, seq_k, d_k, d_v = 2, 10, 12, 64, 32
        >>> query = np.random.randn(batch, seq_q, d_k)
        >>> key = np.random.randn(batch, seq_k, d_k)
        >>> value = np.random.randn(batch, seq_k, d_v)
        >>> output, attn = scaled_dot_product_attention(query, key, value)
        >>> output.shape
        (2, 10, 32)
    """
    d_k = query.shape[-1]
    scores: FloatArray = np.einsum("bqd,bkd->bqk", query, key) / np.sqrt(d_k)

    if mask is not None:
        mask_bool = mask.astype(bool)
        scores = np.where(mask_bool, scores, -1e9)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn: FloatArray = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output: FloatArray = np.einsum("bqk,bkd->bqd", attn, value)
    return output, attn
