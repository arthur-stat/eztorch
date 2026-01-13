from typing import Dict, List, Sequence

from eztorch.text.tokenizer import Tokenizer


class HFTokenizer(Tokenizer):
    """Tokenizer adapter powered by HuggingFace `tokenizers`.

    - Trains a whitespace-based WordLevel tokenizer with special tokens.
    - Adds BOS/EOS via a post-processor so `encode` returns ids including them.
    - Exposes `vocab` and `inv_vocab` for compatibility with existing demos.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, lower: bool = True) -> None:
        # Delayed init; created in build_vocab
        self._tok = None
        self.lower = lower
        self.vocab: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.inv_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}

    def build_vocab(self, texts: Sequence[str]) -> None:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors

        # Normalize to lowercase if requested
        model = models.WordLevel(unk_token=self.UNK_TOKEN)
        tok = Tokenizer(model)
        if self.lower:
            tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        else:
            tok.normalizer = normalizers.NFKC()
        tok.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.WordLevelTrainer(
            special_tokens=[self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN],
            min_frequency=1,
        )

        tok.train_from_iterator(texts, trainer)

        # Resolve IDs of special tokens from trained vocab
        vocab = tok.get_vocab()
        self.vocab = vocab
        self.inv_vocab = {idx: tok for tok, idx in vocab.items()}
        self.PAD = vocab.get(self.PAD_TOKEN, self.PAD)
        self.BOS = vocab.get(self.BOS_TOKEN, self.BOS)
        self.EOS = vocab.get(self.EOS_TOKEN, self.EOS)
        self.UNK = vocab.get(self.UNK_TOKEN, self.UNK)

        # Add post-processor to inject BOS/EOS during encoding
        tok.post_processor = processors.TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[(self.BOS_TOKEN, self.BOS), (self.EOS_TOKEN, self.EOS)],
        )

        self._tok = tok

    def encode(self, text: str) -> List[int]:
        if self._tok is None:
            raise RuntimeError("HFTokenizer not trained. Call build_vocab(texts) first.")
        enc = self._tok.encode(text)
        return list(enc.ids)

    def decode(self, ids: Sequence[int]) -> str:
        # Mirror SimpleTokenizer semantics: skip PAD/BOS, stop at EOS
        tokens: List[str] = []
        for idx in ids:
            if idx in (self.PAD, self.BOS):
                continue
            if idx == self.EOS:
                break
            tok = self.inv_vocab.get(int(idx), self.UNK_TOKEN)
            # Skip unknown special tokens
            if tok in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                continue
            tokens.append(tok)
        return " ".join(tokens)