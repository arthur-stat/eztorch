import numpy as np
from typing import Sequence, List

from eztorch.text.vectorizer import TextVectorizer
from eztorch.typing import FloatArray


class OneHotVectorizer(TextVectorizer):
    """Simple one-hot vectorizer for token id sequences."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, token_ids: Sequence[int]) -> FloatArray:
        arr = np.zeros((len(token_ids), self.vocab_size), dtype=float)
        for i, idx in enumerate(token_ids):
            if 0 <= idx < self.vocab_size:
                arr[i, idx] = 1.0
        return arr

    def batch(self, batch_ids: Sequence[Sequence[int]]) -> FloatArray:
        return np.stack([self(seq) for seq in batch_ids], axis=0)

    def parameters(self) -> List[FloatArray]:
        return []

    def grads(self) -> List[FloatArray]:
        return []