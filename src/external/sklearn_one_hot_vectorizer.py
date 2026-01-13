from typing import Sequence, List

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from eztorch.text.vectorizer import TextVectorizer
from eztorch.typing import FloatArray


class SklearnOneHotVectorizer(TextVectorizer):
    """One-hot vectorizer backed by scikit-learn's OneHotEncoder.

    Produces shape `(seq_len, vocab_size)` for a token id sequence.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.encoder = OneHotEncoder(
            categories=[np.arange(self.vocab_size)],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=float,
        )
        # Fit once using all categories to avoid repeated fitting on calls
        self.encoder.fit(np.arange(self.vocab_size).reshape(-1, 1))

    def __call__(self, token_ids: Sequence[int]) -> FloatArray:
        arr = np.asarray(token_ids, dtype=np.int64).reshape(-1, 1)
        oh = self.encoder.transform(arr)
        return oh.astype(float)

    def batch(self, batch_ids: Sequence[Sequence[int]]) -> FloatArray:
        return np.stack([self(seq) for seq in batch_ids], axis=0)

    def parameters(self) -> List[FloatArray]:
        return []

    def grads(self) -> List[FloatArray]:
        return []

