import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
for path in (SRC_DIR,):
    if path not in sys.path:
        sys.path.append(path)

import numpy as np

from external.hf_tokenizer import HFTokenizer
from external.sklearn_one_hot_vectorizer import SklearnOneHotVectorizer
from eztorch.models.default_transformer import DefaultTransformer
from eztorch.optim.sgd import SGD
from eztorch.optim.base import zero_grads_inplace


def pad_sequences(seqs, pad_id: int):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


def shift_tgt(seqs):
    tgt_in = [s[:-1] for s in seqs]
    tgt_labels = [s[1:] for s in seqs]
    return tgt_in, tgt_labels


def main():
    train_src_texts = [
        "hatsune miku",
        "hoshino ichika",
        "tenma saki",
        "hinomori shiho",
        "mochizuki honami"
    ]
    train_tgt_texts = [
        "初音 未来",
        "星乃 一歌",
        "天马 咲希",
        "日野森 志步",
        "望月 穗波"
    ]
    test_src_texts = [
        "yoisaki kanade",
    ]

    src_tok = HFTokenizer()
    tgt_tok = HFTokenizer()
    src_tok.build_vocab(train_src_texts + test_src_texts)
    tgt_tok.build_vocab(train_tgt_texts)

    src_ids = [src_tok.encode(t) for t in train_src_texts]
    tgt_ids = [tgt_tok.encode(t) for t in train_tgt_texts]

    src_padded = pad_sequences(src_ids, src_tok.PAD)
    tgt_padded = pad_sequences(tgt_ids, tgt_tok.PAD)
    tgt_in_ids, tgt_labels_ids = shift_tgt(tgt_padded)

    src_vocab = len(src_tok.vocab)
    tgt_vocab = len(tgt_tok.vocab)
    src_vec = SklearnOneHotVectorizer(src_vocab)
    tgt_vec = SklearnOneHotVectorizer(tgt_vocab)

    src_batch = src_vec.batch(src_padded)
    tgt_in_batch = tgt_vec.batch(tgt_in_ids)
    tgt_labels = np.array(tgt_labels_ids, dtype=np.int64)

    model = DefaultTransformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=max(src_batch.shape[1], tgt_in_batch.shape[1]),
    )
    optimizer = SGD(lr=0.1)

    for step in range(300):
        zero_grads_inplace(model.grads())
        loss, grad = model.step(src_batch, tgt_in_batch, tgt_labels)
        model.backward(grad, src_batch, tgt_in_batch)
        optimizer.step(model.parameters(), model.grads())
        if step % 50 == 0:
            print(f"step {step}, loss {loss:.4f}")

    logits = model(src_batch, tgt_in_batch)
    preds = logits.argmax(axis=-1)
    for i, pred in enumerate(preds):
        decoded = tgt_tok.decode(pred.tolist())
        print(f"sample {i} predicted: {decoded}")

    for new_src in test_src_texts:
        new_ids = src_tok.encode(new_src)
        new_padded = pad_sequences([new_ids], src_tok.PAD)
        new_src_batch = src_vec.batch(new_padded)
        bos_pad = [[tgt_tok.BOS] + [tgt_tok.PAD] * (tgt_in_batch.shape[1] - 1)]
        new_tgt_in = tgt_vec.batch(bos_pad)
        new_logits = model(new_src_batch, new_tgt_in)
        new_pred = new_logits.argmax(axis=-1)[0]
        decoded_new = tgt_tok.decode(new_pred.tolist())
        print(f"inference '{new_src}' -> '{decoded_new}'")


if __name__ == "__main__":
    main()