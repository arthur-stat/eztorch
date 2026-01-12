from typing import Protocol, List, Sequence


class Tokenizer(Protocol):
    """Tokenizes text into integer ids."""

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, ids: Sequence[int]) -> str:
        ...

    def batch_encode(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def batch_decode(self, batch_ids: Sequence[Sequence[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]