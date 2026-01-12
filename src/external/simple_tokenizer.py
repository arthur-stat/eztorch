from typing import Dict, List, Sequence

from eztorch.text.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Whitespace tokenizer with a built vocabulary; adds BOS/EOS; PAD=0, BOS=1, EOS=2, UNK=3."""

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, lower: bool = True) -> None:
        self.lower = lower
        self.vocab: Dict[str, int] = {
            "<pad>": self.PAD,
            "<bos>": self.BOS,
            "<eos>": self.EOS,
            "<unk>": self.UNK,
        }
        self.inv_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}

    def build_vocab(self, texts: Sequence[str]) -> None:
        for text in texts:
            for tok in self._tokenize(text):
                if tok not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.inv_vocab[idx] = tok

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.BOS] + [self.vocab.get(tok, self.UNK) for tok in tokens] + [self.EOS]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens: List[str] = []
        for idx in ids:
            if idx in (self.PAD, self.BOS):
                continue
            if idx == self.EOS:
                break
            tokens.append(self.inv_vocab.get(idx, "<unk>"))
        return " ".join(tokens)

    def _tokenize(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        return text.strip().split()