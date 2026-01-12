from typing import Protocol, Sequence

from eztorch.typing import FloatArray


class TextVectorizer(Protocol):
    """Maps token ids to vector representations."""

    def __call__(self, token_ids: Sequence[int]) -> FloatArray:
        ...

    def parameters(self) -> list[FloatArray]:
        return []

    def grads(self) -> list[FloatArray]:
        return []